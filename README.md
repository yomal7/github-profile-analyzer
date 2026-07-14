# GitHub Profile Analyzer

Scores a GitHub profile (deterministic + LLM-based) and generates recruiter-facing
improvement tips. Built for a job-search SaaS's microservice layer.

## Setup

```bash
cp .env.example .env        # fill in GITHUB_TOKEN and LLM_API_KEY
uv sync
uv run uvicorn app.main:app --reload
```

`GITHUB_TOKEN` just needs `public_repo`/read scopes — it's only reading public data.
`LLM_API_KEY` is used with the default Gemini OpenAI-compatible endpoint.
If you swap providers, update `llm_base_url` and `llm_model` in `app/config.py`.

Logs are written to `logs/app.log` (general application logs) and `logs/http_access.log`
(HTTP request access logs). Both use rotating file handlers

## Try it

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"github_username": "torvalds", "target_role": "backend"}'
```

## What changed vs. the v1 draft

- **GraphQL query** now pulls `contributionsCollection` (real PR/review/commit activity,
  last 12 months), account `createdAt`, and per-pinned-repo `licenseInfo`,
  `hasIssuesEnabled`, `repositoryTopics`, and best-effort README text.
- **Fixed a scoring bug**: the old PR count measured PRs merged *into the user's own
  repos*, not PRs the user actually authored. It now uses
  `contributionsCollection.totalPullRequestContributions`.
- **`ai_service`** now accepts both `pinned_repos` and `recent_repos`. If a profile has
  no pinned repos, the AI scorer falls back to judging the most recent repos instead,
  providing actionable feedback on what to pin. Uses `AsyncOpenAI` against a
  Gemini-compatible endpoint so the LLM call no longer blocks the FastAPI event loop.
  Returns structured `tips` (issue / action / impact) instead of a single prose string.
- **Logging** now writes to two rotating log files:
  - `logs/app.log`: general application logs (INFO/WARNING/ERROR from all modules)
  - `logs/http_access.log`: HTTP request access logs (Apache combined format)
- **HTTP access logging** via `AccessLogMiddleware` — every request is logged with client IP,
  method, path, query string, status code, duration, and user-agent.
- **`github_service`** raises typed exceptions (`GithubUserNotFoundError`,
  `GithubApiError`) instead of throwing raw `KeyError`s or 404-ing on every failure
  mode, and retries transient failures (timeouts, 429, 5xx) with backoff via `tenacity`.
- **Cache keys are normalized** (lowercased username + target_role) so `Torvalds` and
  `torvalds` share a cache entry instead of silently doubling your miss rate.
- Split into `config.py` / `schemas.py` / `services/` / `core/` per the structure
  discussed — `github_service.py`, `scoring_service.py`, and `core/cache.py` are meant
  to be the stable layer.

## Known limitations worth knowing about (not fixed here)

- `profile_cache` is in-process memory (`TTLCache`). Fine for one instance.
- No persistence. If you want users to track score improvement over time.
- **Logging**: `RotatingFileHandler` is not safe across multiple OS processes writing the
  same file concurrently. Fine for `uvicorn --reload`'s single worker, or a single gunicorn
  worker. For multi-worker deployments, either give each worker its own log file (suffix by
  PID), switch to `QueueHandler` + a listener process, or ship logs to a centralized service.
- README matching only tries `README.md` and `readme.md` at the repo root. It will
  miss READMEs with other casings/paths.