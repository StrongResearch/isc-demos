#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Configuration
# -----------------------------
echo "Configuring environment"

PORT=8000
HOST=localhost
LOG_FILE="/root/isc-demos/llm-inference-benchmarking/vllm.log"
# DATASET_ID="uds-full-titanium-hacksaw-250527" # set as env var
MODEL="/data/${DATASET_ID}"
TP_SIZE=4
GPUS="0,1,2,3"

RESULTS_DIR="/root/isc-demos/llm-inference-benchmarking/results"
RESULT_EXT="json"
POST_RESULT_DELAY=5

SERVER_PID=""
BENCH_PID=""

# make sure results directory exists and is empty
mkdir -p "${RESULTS_DIR}"
rm -rf "${RESULTS_DIR}/*"

# -----------------------------
# Cleanup handler
# -----------------------------
echo "Starting cleanup handler"

cleanup() {
  if [[ -n "${BENCH_PID}" ]] && kill -0 "${BENCH_PID}" 2>/dev/null; then
    echo "Cleaning up benchmarker (PID ${BENCH_PID})..."
    kill -INT "${BENCH_PID}"
    wait "${BENCH_PID}" 2>/dev/null || true
  fi

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
echo "Starting vLLM server"

source /opt/venv/bin/activate
CUDA_VISIBLE_DEVICES=${GPUS} \
nohup python3 -m vllm.entrypoints.openai.api_server \
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
echo "Waiting for server to become ready"

for i in {1..100}; do
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
# echo "Sending test completion request..."

# curl -s "http://${HOST}:${PORT}/v1/chat/completions" \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer test" \
#   -d "{
#     \"model\": \"${MODEL}\",
#     \"messages\": [
#       {\"role\": \"user\", \"content\": \"Hello from the same machine\"}
#     ]
#   }" | jq .

# -----------------------------
# Run inference-benchmarker (background)
# -----------------------------
echo "Starting inference-benchmarker"

# Snapshot existing result files
existing_results=$(ls "${RESULTS_DIR}"/*.${RESULT_EXT} 2>/dev/null || true)

set +e
/root/inference-benchmarker/target/debug/inference-benchmarker \
  --tokenizer-name "${MODEL}/tokenizer.json" \
  --model-name "${MODEL}" \
  --url "http://${HOST}:${PORT}" \
  --profile chat &
set -e

BENCH_PID=$!
echo "Benchmarker PID: ${BENCH_PID}"

# -----------------------------
# Wait for new result file
# -----------------------------
echo "Waiting for benchmark results to be written"

while true; do
  sleep 1
  for f in "${RESULTS_DIR}"/*.${RESULT_EXT}; do
    if ! grep -qxF "$f" <<< "${existing_results}"; then
      echo "Detected new result file: $f"
      echo "Waiting ${POST_RESULT_DELAY}s to ensure file is complete..."
      sleep "${POST_RESULT_DELAY}"

      echo "Stopping benchmarker (PID ${BENCH_PID})..."
      kill -INT "${BENCH_PID}"
      wait "${BENCH_PID}" 2>/dev/null || true
      break 2
    fi
  done
done

# -----------------------------
# Done
# -----------------------------
echo "Done."
echo "Logs: ${LOG_FILE}"


# -----------------------------
# Move results to an experiment artifact
# -----------------------------
echo "Moving results to an artifact"

if [[ -n "$CHECKPOINT_ARTIFACT_PATH" ]]; then
  SAVER_NAME="InferBench_checkpoint_1_force"
  CHECKPOINT_PATH="${CHECKPOINT_ARTIFACT_PATH}/${SAVER_NAME}"
  mkdir -p ${CHECKPOINT_ARTIFACT_PATH}/${SAVER_NAME}
  cp ${RESULTS_DIR}/* ${CHECKPOINT_PATH}/
fi

echo "Sleeping for like 2 minutes to allow time for checkpoint sync"
sleep 120