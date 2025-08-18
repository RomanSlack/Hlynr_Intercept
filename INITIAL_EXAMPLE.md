## FEATURE

- **Unity↔RL Inference API Bridge (Python side)** to serve low-latency control to Unity (extend `bridge_server.py`).
- **Deterministic ENU↔Unity (left-handed) transforms**, versioned and tested.
- **Obs assembly → VecNormalize → PPO policy → post-policy safety clamps** (rate/thrust).
- **Episode JSONL logging + manifest** for deterministic Unity replay.
- **Health/metrics endpoints** with CORS enabled.
- **Action-only, realistic control path**: Unity remains physics-authoritative. Python returns only `rate_cmd_radps{pitch,yaw,roll}` and `thrust_cmd` (no XYZ/rotation teleport).

---

## EXAMPLES

Use these three docs (exclude README) as your canonical references while implementing this feature.

- **`UNITY_DATA_REFERENCE.md` — Episode data contract**
  - Layout: `manifest.json`, `ep_XXXXXX.jsonl`; per-timestep fields (blue/red states, guidance, action, reward, done, info), final summary (outcome, miss distance, impact time, steps, seed, vecnorm id, policy id).
  - Conventions: SI units, quaternions `{w,x,y,z}`, world frame **ENU**; Unity is **left-handed**—apply exact transforms.

- **`Unity_RL_API_Integration_Feasibility.md` — Comms & mapping**
  - Start with **HTTP/JSON**; keep WebSocket/TCP as later options.
  - Map Unity systems → API payloads: `Missile6DOFController` (rate/thrust), `PIDAttitudeController` (rate→torque), `GuidanceProNav`/`SeekerSensor` (LOS/FOV/G-limit).
  - Latency/threading notes for Unity loop; keep Python stateless per request.

- **`Unity_RL_Inference_API_Feasibility.md` — Bridge server plan**
  - Reuse **`bridge_server.py` (Flask+CORS)**; loads **PPO + VecNormalize**; JSON req/resp already in place.
  - Define **v1.0** shapes w/ `obs_version` + `vecnorm_stats_id`; action mapping: `{pitch,yaw,roll} rad/s`, `thrust 0..1`, aux.
  - Targets: **p50 <20 ms**, **p95 <35 ms** end-to-end; add clip-fractions, clamps, and per-step JSONL.

---

## DOCUMENTATION

- Flask (API, CORS): https://flask.palletsprojects.com/
- Pydantic (schema validation): https://docs.pydantic.dev/
- Stable-Baselines3 PPO + VecNormalize: https://stable-baselines3.readthedocs.io/
- JSON Lines (JSONL): https://jsonlines.org/
- Unity coordinate systems (left-handed) & transforms: https://docs.unity3d.com/Manual/CoordinatesInUnity.html
- Quaternion math `{w,x,y,z}` (right-handed ENU → Unity LH) reference: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

---

## OTHER CONSIDERATIONS

- **API (v1.0) — minimal shapes**
  - **Request**: `meta{episode_id,t,dt,sim_tick}`, `frames{frame:"ENU",unity_lh:true}`, `blue{pos_m[3],vel_mps[3],quat_wxyz[4],ang_vel_radps[3],fuel_frac}`, `red{pos_m[3],vel_mps[3],quat_wxyz[4]}`, `guidance{los_unit[3],los_rate_radps[3],range_m,closing_speed_mps,fov_ok,g_limit_ok}`, `env{wind_mps[3]?,noise_std?,episode_step,max_steps}`, `normalization{obs_version,vecnorm_stats_id}`.
  - **Response**: `action{rate_cmd_radps{pitch,yaw,roll},thrust_cmd,aux[]}`, `diagnostics{policy_latency_ms,obs_clip_fractions{low,high},value_estimate?}`, `safety{clamped,clamp_reason?}`.
  *(Keep any examples you already have; just ensure field names & array shapes match this contract.)*
- **Endpoints (v1)**
  - `POST /v1/inference` — main tick endpoint
  - `GET  /healthz` — readiness + loaded IDs
  - `GET  /metrics` — counters + latency percentiles
- **Determinism & logging**
- Include `obs_version`, `vecnorm_stats_id`, **`transform_version`**, and **`policy_id`** in **every** JSONL timestep and in the final summary line.
- Append-only JSONL per step; final summary: `outcome, miss_distance_m, impact_time_s, episode_duration_s, steps, seed, vecnorm_stats_id, policy_id`.
- **Transforms (`transform_version=tfm_v1.0`)**
  - **Do not** reorder quaternion components. Implement a **change-of-basis** from Unity LH to ENU RH:
    - Vectors: `(E,N,U) = (x,z,y)` with a single-axis sign flip to correct handedness.
    - Rotations: convert `{w,x,y,z}` → rotation matrix `R_unity`; apply `R_enu = C · R_unity · C⁻¹` (where `C` encodes the axis permutation + LH→RH); convert back to `{w,x,y,z}` and re-normalize.
  - Provide round-trip unit tests (Unity→ENU→Unity ≈ identity) and golden fixtures.

- **Safety clamps**
- Clamp `{pitch,yaw,roll}` to ±`rate_max_radps`; `thrust_cmd∈[0,1]`; set `clamped=true` with reason (also log).

- **Performance**
- Target **p50 <20 ms**, **p95 <35 ms** (policy + server + serialization). Use monotonic timers for latency.
- **Metrics**: record `policy_latency_ms` using `time.perf_counter()` and export p50/p95 via `/metrics`.

- **Quickstart**
```bash
export FLASK_APP=hlynr_bridge.server:app
flask run --host ${HOST:-0.0.0.0} --port ${PORT:-5000}
	•	Test plan
	•	Transforms round-trip (axis/sign), normalization golden vectors, byte-identical repeat requests, 60 Hz soak (no GC spikes).