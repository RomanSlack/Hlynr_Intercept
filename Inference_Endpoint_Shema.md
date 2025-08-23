Looking at the code, here's exactly what Unity needs to send to the /v1/inference endpoint:

  📡 HTTP Request Structure

  Headers:

  POST /v1/inference HTTP/1.1
  Content-Type: application/json

  Body (JSON):

  {
    "meta": {
      "episode_id": "unity_episode_12345",
      "t": 25.5,
      "dt": 0.02,
      "sim_tick": 1275
    },
    "frames": {
      "frame": "ENU",
      "unity_lh": true
    },
    "blue": {
      "pos_m": [1500.0, 200.0, 1000.0],
      "vel_mps": [180.0, -10.0, 5.0],
      "quat_wxyz": [0.98, 0.05, 0.12, -0.03],
      "ang_vel_radps": [0.1, -0.05, 0.02],
      "fuel_frac": 0.75
    },
    "red": {
      "pos_m": [3000.0, 500.0, 1200.0],
      "vel_mps": [-120.0, -30.0, -8.0],
      "quat_wxyz": [1.0, 0.0, 0.0, 0.0]
    },
    "guidance": {
      "los_unit": [0.85, 0.35, 0.12],
      "los_rate_radps": [0.02, -0.01, 0.005],
      "range_m": 1850.5,
      "closing_speed_mps": 290.0,
      "fov_ok": true,
      "g_limit_ok": true
    },
    "env": {
      "wind_mps": [5.0, 2.0, -1.0],
      "noise_std": 0.015,
      "episode_step": 1275,
      "max_steps": 3000
    },
    "normalization": {
      "obs_version": "obs_v1.0",
      "vecnorm_stats_id": "vecnorm_checkpoints_obs_v1.0_43d32970"
    }
  }

  🔧 Field Descriptions:

  Required Fields:

  - meta.t: Mission time in seconds
  - blue.pos_m: Interceptor position [x, y, z] in meters
  - blue.vel_mps: Interceptor velocity [x, y, z] in m/s
  - blue.fuel_frac: Fuel remaining (0.0-1.0)
  - red.pos_m: Threat position [x, y, z] in meters
  - red.vel_mps: Threat velocity [x, y, z] in m/s
  - guidance.range_m: Distance to target
  - guidance.closing_speed_mps: Approach rate (negative = diverging)
  - normalization.vecnorm_stats_id: Must match server's loaded stats

  Unity C# Example:

  var requestBody = new {
      meta = new {
          episode_id = $"unity_mission_{Time.time}",
          t = missionTime,
          dt = Time.fixedDeltaTime,
          sim_tick = (int)(missionTime / Time.fixedDeltaTime)
      },
      frames = new { frame = "ENU", unity_lh = true },
      blue = new {
          pos_m = new float[] { transform.position.x, transform.position.y, transform.position.z },
          vel_mps = new float[] { rigidbody.velocity.x, rigidbody.velocity.y, rigidbody.velocity.z },
          quat_wxyz = new float[] { transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z },
          ang_vel_radps = new float[] { rigidbody.angularVelocity.x, rigidbody.angularVelocity.y, rigidbody.angularVelocity.z },
          fuel_frac = currentFuel / maxFuel
      },
      red = new {
          pos_m = new float[] { threatTransform.position.x, threatTransform.position.y, threatTransform.position.z },
          vel_mps = new float[] { threatVelocity.x, threatVelocity.y, threatVelocity.z },
          quat_wxyz = new float[] { 1f, 0f, 0f, 0f }
      },
      guidance = new {
          los_unit = losDirection.normalized,
          los_rate_radps = new float[] { 0f, 0.01f, 0f }, // Can be simplified
          range_m = Vector3.Distance(transform.position, threatTransform.position),
          closing_speed_mps = -Vector3.Dot(losDirection.normalized, rigidbody.velocity - threatVelocity),
          fov_ok = true,
          g_limit_ok = true
      },
      env = new {
          wind_mps = new float[] { 0f, 0f, 0f }, // Can be simplified
          noise_std = 0.01,
          episode_step = (int)missionTime,
          max_steps = 9999
      },
      normalization = new {
          obs_version = "obs_v1.0",
          vecnorm_stats_id = "vecnorm_checkpoints_obs_v1.0_43d32970"
      }
  };

  The 422 error you're seeing means some field is missing or has wrong type/format. Make sure all required fields are present and numeric values are proper
  floats/ints!
