"""
LLM-based cognitive scoring via an OpenAI-compatible endpoint (currently Gemini).

Changes that matter in production:
  1. AsyncOpenAI, not OpenAI. The sync client's .create() call blocks the
     whole event loop while it waits on the network — inside an `async def`
     FastAPI route that stalls every other in-flight request on the worker.
  2. Structured `tips` in the response, not a single ai_insight string.
     "Tips to improve the profile" is the actual product — a paragraph of
     prose isn't something a UI can render as a checklist.
  3. Explicit completion params (max_tokens, reasoning_effort, sampling) set
     from config rather than left as provider defaults — free-tier hosted
     endpoints are exactly where undocumented defaults bite you.
  4. `_ai_degraded` sentinel on the returned dict. main.py checks this before
     writing to the cache — a transient LLM outage should never get cached
     as if it were a real result for 48 hours.
  5. Takes BOTH pinned_repos and recent_repos. A profile with nothing pinned
     used to get a canned "pin some projects" message and a zeroed cognitive
     score — now it falls back to judging the most recent repos instead. And
     even when pinned_repos isn't empty, recent-but-unpinned repos are still
     passed along so the model can flag one worth pinning by name.

Previously ran against NVIDIA NIM's deepseek-v4-flash, which required NIM's
non-standard `extra_body.chat_template_kwargs` to control thinking and was
unreliable on the free tier (frequent 503 "workers busy" under load). Gemini's
OpenAI-compat layer takes `reasoning_effort` as a plain parameter, so that
plumbing is gone — swapping providers again later just means changing the
values in config.py, not this file's logic.
"""

import json
import logging

from openai import AsyncOpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()

_client = AsyncOpenAI(
    base_url=_settings.llm_base_url,
    api_key=_settings.llm_api_key,
    timeout=_settings.llm_request_timeout,
)

_MAX_OTHER_REPOS = 6

_SYSTEM_PROMPT = """\
You are a senior technical recruiter reviewing a GitHub profile for a {role} role.

You are given two lists:
- "pinned_repos": the repos the candidate has chosen to showcase on their profile.
  This list may be EMPTY if the candidate hasn't pinned anything.
- "other_recent_repos": some of the candidate's other recent, non-fork repos that
  are NOT currently pinned. This list may also be empty.

Score the profile's showcase quality:
1. Technical Complexity (0-25): basic tutorials or copy-paste clones score low;
   original systems, non-trivial architecture, or real infrastructure score high.
2. Originality (0-25): standard bootcamp/course clones score low; a unique
   problem or unusual approach scores high.

If pinned_repos is non-empty, base these two scores on pinned_repos — that's what a
recruiter actually sees first. If pinned_repos is EMPTY, base the scores on the
strongest repos in other_recent_repos instead, and say plainly in ai_insight that
the candidate hasn't pinned anything yet and this score reflects their recent work.

Separately, look at other_recent_repos and decide whether any of them would
meaningfully strengthen the profile if pinned — because it's more complex, better
documented, or more relevant to the {role} role than what's currently pinned, or
because nothing is pinned at all. If so, include a tip recommending the candidate
pin it, naming the specific repo. Don't force this if nothing in other_recent_repos
is actually an improvement.

Produce 2-4 concrete, actionable tips total, covering both showcase-quality issues
and any pin recommendation. Every tip must reference something specific you actually
observed (a repo name, a missing README, a missing license, a thin description) — no
generic advice like "write more code" or "contribute to open source."

Return ONLY valid JSON, no markdown fences, exactly matching this shape:
{{
  "complexity_score": <int 0-25>,
  "originality_score": <int 0-25>,
  "cognitive_total": <int, sum of the two above>,
  "ai_insight": "<one paragraph summary of the profile>",
  "tips": [
    {{"issue": "<string>", "action": "<string>", "impact": "high" | "medium" | "low"}}
  ]
}}
"""

_FALLBACK = {
    "complexity_score": 0,
    "originality_score": 0,
    "cognitive_total": 0,
    "ai_insight": "AI evaluation is unavailable right now — showing the deterministic score only.",
    "tips": [],
    "_ai_degraded": True,
}

_README_EXCERPT_CHARS = 1200  # keep token usage predictable regardless of README length


def _strip_json_fences(text: str) -> str:
    """
    response_format=json_object should prevent this, but reasoning models
    occasionally wrap output in ```json fences anyway — strip defensively
    rather than let a stray fence blow up json.loads.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text.removeprefix("json").strip()
    return text


def _extract_readme_text(repo: dict) -> str:
    """
    readme_md / readme_lower come back from the GraphQL query as Blob objects
    (`{"text": "..."}`), or None if the repo has no README at that path — not
    plain strings. Unwrap here so callers just get text.
    """
    readme_obj = repo.get("readme_md") or repo.get("readme_lower")
    if isinstance(readme_obj, dict):
        return readme_obj.get("text") or ""
    return ""


def _prune_repo(repo: dict) -> dict:
    readme = _extract_readme_text(repo)
    topics = [
        node["topic"]["name"]
        for node in ((repo.get("repositoryTopics") or {}).get("nodes") or [])
    ]
    return {
        "name": repo.get("name"),
        "description": repo.get("description"),
        "language": (repo.get("primaryLanguage") or {}).get("name"),
        "has_license": bool(repo.get("licenseInfo")),
        "topics": topics,
        "readme_excerpt": readme[:_README_EXCERPT_CHARS],
    }


def _unpinned_recent(pinned_repos: list[dict], recent_repos: list[dict]) -> list[dict]:
    """
    recent_repos can legitimately include repos that are also pinned (pinned repos
    are usually recently pushed too) — filter those out so the model isn't asked to
    "recommend pinning" something that's already pinned.
    """
    pinned_names = {r.get("name") for r in pinned_repos}
    return [r for r in recent_repos if r.get("name") not in pinned_names][:_MAX_OTHER_REPOS]


async def get_cognitive_score(
    pinned_repos: list[dict], recent_repos: list[dict], target_role: str | None
) -> dict:
    other_recent = _unpinned_recent(pinned_repos, recent_repos)

    if not pinned_repos and not other_recent:
        return {
            **_FALLBACK,
            "ai_insight": "No repositories to analyze yet — push some work and pin your best projects.",
            "_ai_degraded": False,
        }

    payload = {
        "pinned_repos": [_prune_repo(r) for r in pinned_repos],
        "other_recent_repos": [_prune_repo(r) for r in other_recent],
    }
    role_label = target_role or "general software engineering"

    try:
        response = await _client.chat.completions.create(
            model=_settings.llm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT.format(role=role_label)},
                {"role": "user", "content": json.dumps(payload)},
            ],
            temperature=_settings.llm_temperature,
            top_p=_settings.llm_top_p,
            max_tokens=_settings.llm_max_tokens,
            response_format={"type": "json_object"},
            reasoning_effort=_settings.llm_reasoning_effort,
        )
        message = response.choices[0].message

        reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
        if reasoning:
            logger.debug("LLM reasoning trace: %s", reasoning)

        parsed = json.loads(_strip_json_fences(message.content))
        parsed.setdefault("tips", [])
        parsed.setdefault("ai_insight", "")
        parsed.setdefault("_ai_degraded", False)
        return parsed
    except (json.JSONDecodeError, KeyError, IndexError, AttributeError) as exc:
        # Model returned something we couldn't parse as the expected JSON shape.
        logger.warning("LLM response did not match expected shape: %s", exc)
        return _FALLBACK
    except Exception as exc:  # noqa: BLE001 — deliberately broad: any provider/network failure
        logger.warning("LLM call failed: %s", exc)
        return _FALLBACK