#!/bin/bash
# Hlynr Inference Server Startup Script

echo "Starting Hlynr Inference Server..."
echo "Setting environment variables..."

# Set required environment variables
export MODEL_CHECKPOINT="phase4_rl/checkpoints/phase4_easy_final.zip"
export VECNORM_STATS_ID=""  # Leave empty to use legacy file or skip normalization
export OBS_VERSION="obs_v1.0"
export TRANSFORM_VERSION="tfm_v1.0"
export HOST="0.0.0.0"
export PORT="5000"

# Optional settings
export LOG_LEVEL="INFO"
export CORS_ORIGINS='["http://localhost:3000", "http://localhost:8080"]'
export ENABLE_CORS="true"
export RATE_MAX_RADPS="10.0"

echo "Environment variables set:"
echo "  MODEL_CHECKPOINT: $MODEL_CHECKPOINT"
echo "  VECNORM_STATS_ID: $VECNORM_STATS_ID"
echo "  OBS_VERSION: $OBS_VERSION"
echo "  HOST:PORT: $HOST:$PORT"

echo ""
echo "Starting uvicorn server..."
echo "Navigate to: http://localhost:5000/healthz to check health"
echo "Press Ctrl+C to stop the server"
echo ""

# Change to src directory and start server
cd "$(dirname "$0")/src"
uvicorn hlynr_bridge.server:app --host "$HOST" --port "$PORT" --reload