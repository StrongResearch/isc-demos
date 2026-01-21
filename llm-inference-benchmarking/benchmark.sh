#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Configuration
# -----------------------------
PORT=8000
HOST=127.0.0.1
LOG_FILE="vllm.log"
DATASET_ID=""
MODEL="/data/${DATASET_ID}"
TP_SIZE=4
GPUS="0,1,2,3"

SERVER_PID=""

# -----------------------------
# Cleanup handler
# -----------------------------
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Cleaning up vLLM server (PID ${SERVER_PID})..."
    kill "${SERVER_PID}"
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

# -----------------------------
# Start server in background
# -----------------------------
echo "Starting vLLM server..."

CUDA_VISIBLE_DEVICES=${GPUS} \
nohup python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  > "${LOG_FILE}" 2>&1 &

SERVER_PID=$!
echo "vLLM server PID: ${SERVER_PID}"

# -----------------------------
# Wait for server readiness
# -----------------------------
echo "Waiting for server to become ready..."

for i in {1..60}; do
  if curl -sf "http://${HOST}:${PORT}/v1/models" > /dev/null; then
    echo "Server is up!"
    break
  fi
  sleep 1
done

if ! curl -sf "http://${HOST}:${PORT}/v1/models" > /dev/null; then
  echo "ERROR: Server did not start within timeout"
  echo "Last 50 log lines:"
  tail -n 50 "${LOG_FILE}"
  exit 1
fi

# -----------------------------
# Curl a test request
# -----------------------------
echo "Sending test completion request..."

curl -s "http://${HOST}:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"Hello from the same machine\"}
    ]
  }" | jq .

echo "Done."
echo "Logs: ${LOG_FILE}"
