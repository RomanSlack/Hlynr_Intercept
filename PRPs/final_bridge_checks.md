You are Claude Code working directly on my local VS Code project. You can read & edit files, run shells, run the server, and execute tests. Please follow the workflow: Explore → Plan → Code → Verify → Commit. Do not skip Explore/Plan. Keep changes minimal and targeted to get the server production-ready for Unity integration tomorrow.

PROJECT CONTEXT (read this first)
- Repo root: /Users/quinnhasse/dev/Hlynr_Intercept
- Server is FastAPI at: src/hlynr_bridge/server.py  (uvicorn target: hlynr_bridge.server:app)
- Schemas/logic: src/hlynr_bridge/{schemas.py,transforms.py,normalize.py,clamps.py,episode_logger.py,config.py,seed_manager.py}
- Legacy training & utils live under src/phase4_rl/ (keep, but the server runs from hlynr_bridge)
- Model artifacts likely here:
  - checkpoints/best_model/best_model.zip
  - vecnorm/test_vecnorm_001.pkl
- Prometheus metrics endpoint must expose hlynr_* metrics in exposition format.
- Healthz must include EXACT 7 keys: ok, policy_loaded, policy_id, vecnorm_stats_id, obs_version, transform_version, seed
- Inference endpoint: POST /v1/inference returns action-only commands (pitch/yaw/roll rate_cmd_radps + thrust_cmd + aux), diagnostics, and safety fields.
- /act must be feature-flag gated OFF by default with ENABLE_LEGACY_ACT=false.

KNOWN GOOD: We already migrated Flask→FastAPI and fixed the ASGI error. /healthz and /metrics return 200, but /healthz fields are null until env/model are set, and metrics counters are zero until requests hit inference.

OBJECTIVE
Fully verify the server works end-to-end with our real model and frozen VecNormalize stats, and fix any remaining wiring issues. Deliver a final PASS/FAIL checklist with concrete evidence. Keep refactors minimal; only fix what’s necessary to make the server production-ready for Unity tomorrow.

STEP 1 — EXPLORE (do not edit yet)
1) Open src/hlynr_bridge/server.py and confirm:
   - FastAPI imports and app = FastAPI()
   - /healthz assembles the 7 required keys from a persistent server_stats (or equivalent) store
   - Seed setup (set_deterministic_seeds) runs at startup and stores seed in server_stats
   - Model loading: determine EXACT env var names required (e.g., MODEL_CHECKPOINT / POLICY_ID / VECNORM_STATS_ID / OBS_VERSION / TRANSFORM_VERSION)
   - Metrics: from prometheus_client import ...; /metrics returns generate_latest() with correct content-type
   - /v1/inference: confirm it uses Pydantic request, normalizes obs via VecNormalize-by-ID, runs deterministic policy inference (deterministic=True, torch threads=1), applies safety clamps, logs JSONL with required version fields
   - Episode logging: verify that per-timestep version fields and final summary include obs_version, vecnorm_stats_id, transform_version, policy_id, seed, steps, outcome, etc. (either via inference_logger or episode_logger)
   - Legacy /act endpoint is gated by ENABLE_LEGACY_ACT env var (default false)
2) Confirm artifact paths exist:
   - checkpoints/best_model/best_model.zip
   - vecnorm/test_vecnorm_001.pkl
3) Confirm sample req.json location if present; otherwise we’ll create one.

STEP 2 — PLAN
Draft a short plan (few bullets) describing:
- Which env vars we must set, with the exact names the code expects
- The commands to start the server on port 5001 (single worker)
- The curl commands to run sanity checks
- The expected JSON shapes/fields we’ll assert
- Any likely failure modes (e.g., bad path names, missing IDs) and how you’ll auto-diagnose them from logs

STEP 3 — RUN & VERIFY (adjust code only if needed)
A) Environment & startup
- In a shell in repo root, set env vars to what server.py expects (use the exact names from EXPLORE). If names differ, adapt accordingly. Example defaults (adjust if code differs):
  export POLICY_ID=best_model_v1
  export MODEL_CHECKPOINT=checkpoints/best_model/best_model.zip
  export VECNORM_STATS_ID=test_vecnorm_001
  export OBS_VERSION=obs_v1.0
  export TRANSFORM_VERSION=tfm_v1.0
  export SEED=42
  export ENABLE_LEGACY_ACT=false
- Start server:
  PYTHONPATH=src uvicorn hlynr_bridge.server:app --host 0.0.0.0 --port 5001 --workers 1 --reload --log-level debug
- If startup fails to load model or vecnorm stats, capture the exact log error and fix the code or env var names/paths minimally (e.g., correct file path, ensure vecnorm loader points to vecnorm/<id>.pkl). Keep changes surgical.

B) Healthz contract
- Run:
  curl -s http://localhost:5001/healthz | jq '{ok,policy_loaded,policy_id,vecnorm_stats_id,obs_version,transform_version,seed}'
- VERIFY: All 7 fields exist; ok:true; policy_loaded:true after model loads; IDs/versions/seed not null.
- If any are null: fix the server to persist those values at startup (server_stats or equivalent), then retest.

C) Prometheus metrics
- Run:
  curl -s http://localhost:5001/metrics | grep -E 'hlynr_requests_total|hlynr_inference_latency_ms|hlynr_safety_clamps_total'
- Initially counts may be zero; that’s fine until we hit inference.

D) Exercise inference path (action-only contract)
- If no req.json exists, create one at repo root using this exact schema (adjust only if our schemas demand it):
{
  "meta": {"episode_id": "test_001", "t": 1.0, "dt": 0.01, "sim_tick": 100},
  "frames": {"frame": "ENU", "unity_lh": true},
  "blue": {
    "pos_m": [0.0, 0.0, 100.0],
    "vel_mps": [100.0, 0.0, 0.0],
    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    "ang_vel_radps": [0.0, 0.0, 0.0],
    "fuel_frac": 0.5
  },
  "red": {
    "pos_m": [1000.0, 0.0, 0.0],
    "vel_mps": [-50.0, 0.0, 0.0],
    "quat_wxyz": [1.0, 0.0, 0.0, 0.0]
  },
  "guidance": {
    "los_unit": [1.0, 0.0, 0.0],
    "los_rate_radps": [0.0, 0.0, 0.0],
    "range_m": 1000.0,
    "closing_speed_mps": 150.0,
    "fov_ok": true,
    "g_limit_ok": true
  },
  "env": {"episode_step": 100, "max_steps": 1000},
  "normalization": {"obs_version": "obs_v1.0", "vecnorm_stats_id": "test_vecnorm_001"}
}
- POST 3x:
  for i in {1..3}; do
    curl -s -X POST http://localhost:5001/v1/inference -H "Content-Type: application/json" -d @req.json > out$i.json
  done
- VERIFY response shape with jq:
  jq '{action:.action.rate_cmd_radps, thrust:.action.thrust_cmd, aux:.action.aux, diagnostics, safety}' out1.json
- Contract checks:
  - action.rate_cmd_radps has pitch/yaw/roll
  - action.thrust_cmd in [0,1]
  - diagnostics includes policy_latency_ms and obs_clip_fractions low/high
  - safety has clamped (bool) and clamp_reason (nullable/enum)
- Determinism sanity: compare only actions across responses (ignore latency fields):
  jq '.action' out1.json > a1.json; jq '.action' out2.json > a2.json; diff a1.json a2.json || true
  (They should match unless stochasticity is enabled; if they don’t, ensure deterministic=True and seeds/threading enforcement.)

E) Metrics after inference
- Run again:
  curl -s http://localhost:5001/metrics | grep -E 'hlynr_requests_total|hlynr_inference_latency_ms|hlynr_safety_clamps_total'
- VERIFY: Counts > 0; histogram/summary show samples; if clamps applied, hlynr_safety_clamps_total increments with canonical reasons.

F) Logging & manifest
- If the server logs inference steps, verify JSONL and manifest:
  - Find the output directory used by episode_logger/inference_logger (from EXPLORE). Example: runs/ or inference_episodes/
  - Verify latest episode JSONL contains per-timestep fields: obs_version, vecnorm_stats_id, transform_version, policy_id; summary line has outcome, miss_distance_m (if provided), impact_time_s (if provided), episode_duration_s, steps, seed, vecnorm_stats_id, policy_id.
  - If server exposes verify_manifest(), run it via a one-liner Python call and show output. Otherwise, write a tiny validation util or grep/jq checks to ensure manifest.json exists and references real files.

G) Legacy endpoint gating
- Ensure ENABLE_LEGACY_ACT=false by default; test:
  curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:5001/act -d '{}'  # expect 404
- If it isn’t 404, gate it behind the env var (minimal fix only).

STEP 4 — FIXES (only if needed)
- If env var names don’t match what code expects, update the README & .env.example and/or adjust server to read the names we used above (single source of truth).
- If healthz fields remain null, ensure server sets and persists them at startup (server_stats["seed"], ["policy_id"], ["vecnorm_stats_id"], ["obs_version"], ["transform_version"]).
- If vecnorm fails to load, ensure normalize.load_vecnorm(stats_id) maps to vecnorm/<stats_id>.pkl and forces eval-only mode.
- If determinism fails, enforce deterministic=True and torch.set_num_threads(1) in inference, and confirm seeds are set once at startup.

STEP 5 — FINAL REPORT & COMMIT
- Produce a concise PASS/FAIL checklist with evidence snippets (command + trimmed output) for:
  [ ] Server start (uvicorn FastAPI) without ASGI errors
  [ ] /healthz 7 keys populated (ok=true, policy_loaded=true, IDs/versions/seed set)
  [ ] /metrics Prometheus with hlynr_* metrics and non-zero counts after inference
  [ ] /v1/inference returns action-only command with diagnostics & safety fields
  [ ] Deterministic actions across repeated identical requests (at least action object matches)
  [ ] Episode JSONL includes required version fields per timestep and complete summary
  [ ] manifest.json exists & correctly indexes episodes
  [ ] /act returns 404 by default (feature-flagged)
- If changes were needed, commit with a clear, small message like:
  fix(server): load policy/vecnorm via env; persist healthz keys; solidify metrics & logging
- If no code changes needed, do not commit; just provide the PASS report.

IMPORTANT NOTES
- Use port 5001 (5000 may be occupied).
- Run the server in the same shell where env vars are exported.
- Keep edits minimal and focused on correctness for tomorrow’s Unity hookup.
- Do not remove or refactor training code under src/phase4_rl; only ensure the server imports from src/hlynr_bridge and runs independently.

Begin with EXPLORE and produce your short PLAN before coding. Then proceed through RUN & VERIFY and only make targeted changes if a check fails.