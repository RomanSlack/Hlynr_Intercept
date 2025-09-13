### 🔄 Project Awareness & Context & Research
### 🎯 Hlynr Intercept — Project-Specific Rules

- **Canonical refs:** Treat `Unity_RL_Inference_API_Feasibility.md`, `Unity_RL_API_Integration_Feasibility.md`, and `UNITY_DATA_REFERENCE.md` as the single sources of truth for this feature (exclude README from EXAMPLES).
- **Frames & units:** Python side runs in **ENU (right-handed)**; Unity is **left-handed**. Keep **deterministic ENU↔Unity transforms** (versioned), **SI units**, and quaternions **{w,x,y,z}** everywhere.
- **API v1.0 (HTTP/JSON):** Request must include `obs_version`, `vecnorm_stats_id`, and `transform_version`. Response returns `rate_cmd_radps{pitch,yaw,roll}`, `thrust_cmd[0..1]`, `aux[]`, diagnostics (`policy_latency_ms`, `obs_clip_fractions`, optional `value_estimate`), and `safety{clamped,clamp_reason?}`.
- **Determinism:** Same request → **byte-identical** response and JSONL log. Freeze VecNormalize stats by ID; include the IDs in every timestep and summary line.
- **Latency SLOs:** End-to-end (policy + server + serialization) **p50 < 20 ms**, **p95 < 35 ms**.
- **Safety clamps:** Hard-limit angular rates (±`rate_max_radps`) and `thrust_cmd∈[0,1]`; set `clamped=true` and record `clamp_reason` in response and logs.
- **Episode artifacts:** Append-only **JSONL per timestep** (blue/red states, guidance, action, diagnostics, safety, reward/done/info) plus **final summary** (outcome, miss distance, impact time, duration, steps, seed, `vecnorm_stats_id`, `policy_id`). Maintain a `manifest.json` indexing episodes for Unity replay.
- **Testing (must-pass):** (1) Transform round-trip & axis/sign checks, (2) normalization golden-vector → known action, (3) repeated-request determinism, (4) 60 Hz soak without latency spikes/GC.
- **File layout (Python side, minimal expectation):**
src/hlynr_bridge/{server.py,schemas.py,transforms.py,normalize.py,policy.py,clamps.py,episode_logger.py,config.py}
tests/{test_transforms.py,test_schemas.py,test_normalize.py,test_end_to_end.py}
- **Env vars (.env example):** `MODEL_CHECKPOINT`, `VECNORM_STATS_ID`, `OBS_VERSION`, `TRANSFORM_VERSION`, `HOST`, `PORT`, `CORS_ORIGINS`, `LOG_DIR`.
- **Scope guardrails:** Do **not** modify Unity physics/guidance; expose clean interfaces only. Keep Python server stateless per-request.

- **Documentation is a source of truth** - Your knowledge is...third party API's - that information was freshsly scraped and yo
- **Docker & Selftesting** - You must use Docker and you mus...ut fixing anything. You can use Docker with Curl, or just by run
- **check all jina scrapes** - some Jina scrapes fail and ha...again until it works and you get the actual content of the file.
- **Always read `PLANNING.md`** at the start of a new conver...stand the project's architecture, goals, style, and constraints.
- **Check `TASK.md`** before starting a new task. If the tas... isn’t listed, add it with a brief description and today's date.
- **Use consistent naming conventions, file structure, and architecture patterns** as described in `PLANNING.md`.
- **Use Docker commands** whenever executing Python commands, including for unit tests.
- **Set up Docker** Setup a docker instance for development ...ut of Docker so that you can self improve your code and testing.
- **Agents** - Agents should be designed as intelligent huma...t your basic propmts that generate absolute shit. This is absolu
- **Stick to OFFICIAL DOCUMENTATION PAGES ONLY** - For all r...en to you in intitial.md and then create a llm.txt from it in yo
**Create full production ready code**
- **Ultrathink** - Use Ultrathink capabilities before every ...ation and code generation, what informatoin to put into PRD etc.
- **LLM Models** - Always look for the models page from the ...not change models, find the exact model name to use in the code.
- **Always scrape around 30-100 pages in total when doing re...actual page/content. Put the output of each SUCCESFUL Jina scrap
- **Refer to /research/ directory** - Before implementing an...e /research/ directory and use the .md files to ensure you're co
- **Source Code and Run** - Use DIFFUSERS for text to image.... Refrain from changing this unless the requirements say so. - A
- **Clarity & Simplicity** - Choosing clarity, simplicity, and maintainability over cleverness.
- **Use Type Hints** - Always write complete type hints for functions and methods.
- **Docstrings** - Every function and class must have a docstring explaining its purpose and usage.
- **Use Hyphens** - When writing a comment, use hyphens to organize your thoughts across bullets.
- **Use Consistent Naming** - snake_case for variables and functions, PascalCase for classes, and UPPER_CASE for constants.
- **Respect the Folder Structure** - Do not move folders or files unless explicitly requested.
- **Use Environment Variables** - Store sensitive configuration and keys in `.env` and load them with `dotenv`.
- **Log Everything** - Use Python's `logging` module for debug and error logs; never rely on `print` in production code.
- **Write Tests First** - Before writing the implementation, write unit tests, then implement code to pass tests (TDD).
- **Coverage** - Ensure your tests cover at least 90% of the codebase; include edge cases, error handling, and boundary conditions.
- **Use Fixtures** - In tests, use `pytest` fixtures for reusable setup. Keep test data in a `tests/fixtures/` directory.
- **Linting & Formatting** - Use `ruff` for linting and `black` for formatting; enforce them in CI via pre-commit.
- **Pre-Commit Hooks** - Install and run pre-commit hooks for linting, formatting, and static analysis before every commit.
- **CI/CD** - Use GitHub Actions to run tests, linting, and build steps on every push to `main` and for pull requests.
- **Branch Strategy** - Use feature branches with descriptive names (`feature/add-user-auth`) and open PRs early for feedback.
- **Atomic Commits** - Make small, focused commits with clear messages; avoid mixing unrelated changes.
- **Code Review** - Submit PRs for review; reviewers must check for correctness, security, performance, and readability.
- **Dependencies** - Pin exact versions in `requirements.txt` and update with care; use `pip-tools` for managing constraints.
- **Security** - Never commit secrets; run `pip-audit` regularly; validate all inputs at the API boundaries.
- **Performance** - Use profiling tools to identify bottlenecks; optimize only when necessary and with benchmarks.
- **Feature Flags** - Guard incomplete features behind flags; use environment variables or a config file.
- **Feature Toggles** - Keep toggles short-lived and remove them once features are stable.
- **Graceful Degradation** - Handle API failures or missing dependencies gracefully, with retries and fallback behaviors.
- **Documentation** - Update `README.md` and `docs/` for any significant change; keep examples runnable and accurate.
- **Exceptions** - Use custom exceptions for domain-specific errors; never swallow exceptions silently.
- **API Design** - Use RESTful principles; version APIs and avoid breaking changes without deprecation plans.
- **Contracts** - Use Pydantic or Marshmallow for schema validation and type-safe parsing at service boundaries.
- **Monitoring** - Instrument code with metrics (`prometheus_client`); set alerts for error rates and latency.
- **Internationalization** - If applicable, use standard i18n libraries; avoid hard-coded strings.
- **Accessibility** - Ensure UIs (if any) meet accessibility standards; follow WAI-ARIA guidelines.
- **Legal & Compliance** - Confirm licensing for dependencies; ensure exported artifacts meet regulatory needs.
- **Data Privacy** - PII must be encrypted at rest and in transit; redact sensitive data in logs.
- **Backups** - If persistent storage is involved, document backup/restore procedures; test them quarterly.
- **Configuration Management** - Keep config in code; document defaults and overrides; avoid ad-hoc config drift.
- **Idempotence** - Make scripts retryable and safe to run multiple times.
- **Observability** - Standardize logging levels, structure logs as JSON where appropriate; capture correlation IDs.
- **Resiliency** - Use timeouts, retries with exponential backoff, and circuit breakers for remote calls.
- **Task Management** - Keep `TASK.md` updated with status, owner, and next steps for each active task.
- **Time Management** - Prioritize tasks by impact and effort; avoid premature optimization or gold-plating.
- **Meetings** - Before meetings, prepare agendas; after meetings, document decisions and action items in `MEETING_NOTES.md`.
- **Communication** - Prefer clear, concise written communication; document assumptions and alternatives considered.
- **Mentorship** - Share knowledge via code reviews, documentation, and brown-bag sessions; help others grow.
- **Ownership** - Take responsibility for the code you write; fix bugs promptly and write tests to prevent regressions.

- **Always create a module structure that can scale.** Folde... to mergeeverthing in the right mess to the issue management.
- **Use RAG** put all files as basis if not already. Commen...for wrong assumptions about a database when it's in datasources.
- **For wrong logic we want future looking solutions or with...ot the fastest but will not create a mess for the future. Its o
- **Prometheus** - Always include this for monitoring basic a... own using the standard python libs - explain how to integrate it
- **Ensights** - Include a pipeline that gets screens as input...me 10 pure different ideas how to create truly new and OG content
- **Graphs** - Always use graphs and diagrams in the prs, and... on how to improve this. Come up with different original ideas an

> 🚨 Some imports and usage may fail. Use Jira - Include warnings...ises. Create PR with a summary and ask for guidance to resolve.

## Development Guidelines

### 🧱 Code Style
- Follow **PEP 8** and use **black** for formatting and **ruff** for linting.
- Prefer composition over inheritance; keep functions small and focused.
- Avoid global state; pass dependencies explicitly.

### 🧪 Testing Requirements
- Use **pytest** for writing tests; include unit, integration, and e2e tests.
- Mock external dependencies (APIs, databases) to keep tests deterministic.
- Run tests locally and in CI; ensure they're fast and reliable.

### 🔧 Tooling
- Use **pre-commit** hooks for linting and formatting on commit.
- Use **mypy** for optional static typing enforcement.
- Use **Makefile** or **justfile** for common tasks (lint, test, build).

### 🚀 Deployment
- Use Docker for containerization; provide a `Dockerfile` and `docker-compose.yml` if needed.
- Tag releases semantically (e.g., `v1.2.3`); maintain a `CHANGELOG.md`.
- Store environment variables in `.env` and document them in `README.md`.

### 🔐 Security Best Practices
- Validate and sanitize all inputs.
- Use HTTPS for all external communications.
- Keep dependencies up to date; run `pip-audit` or similar regularly.

### 📚 Documentation & Explainability
- **Update `README.md`** when new features are added, dependencies change, or setup steps are modified.
- **Comment non-obvious code** and ensure everything is understandable to a mid-level developer.
- When writing complex logic, **add an inline `# Reason:` comment** explaining the why, not just the what.

### 🧠 AI Behavior Rules
- **Never assume missing context. Ask questions if uncertain.**
- **Never hallucinate libraries or functions** – only use known, verified Python packages.
- **Always confirm file paths and module names** exist before referencing them in code or tests.
- **Never delete or overwrite existing code** unless explicitly instructed to or if part of a task from `TASK.md`.
