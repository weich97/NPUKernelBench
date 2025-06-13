#!/bin/bash

# vLLM Server Startup Script for Ascend Environment

# Source Ascend toolkit environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# Configure deterministic behavior
export LCCL_DETERMINISTIC=1
export HCCL_DETERMINISTIC=true
export ATB_MATMUL_SHUFFLE_K_ENABLE=false
export ATB_LLM_LCOC_ENABLE=false
export TASK_QUEUE_ENABLE=2

# Configure vLLM settings
export VLLM_RPC_GET_DATA_TIMEOUT_MS=1800000000
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export TORCH_COMPILE_DEBUG=1
export TORCHDYNAMO_DISABLE=1
export RAY_DEDUP_LOGS=0

# Default configuration
DEFAULT_PORT=5600
DEFAULT_CONFIG_PATH="base_config.yaml"
DEFAULT_LOG_DIR="./vllm_serving_log"
DEFAULT_TENSOR_PARALLEL_SIZE=8
DEFAULT_MAX_MODEL_LEN=32768

echo "--------------------ENVIRONMENT VARIABLES----------------------"
echo "LCCL_DETERMINISTIC=$LCCL_DETERMINISTIC"
echo "HCCL_DETERMINISTIC=$HCCL_DETERMINISTIC"
echo "ATB_MATMUL_SHUFFLE_K_ENABLE=$ATB_MATMUL_SHUFFLE_K_ENABLE"
echo "ATB_LLM_LCOC_ENABLE=$ATB_LLM_LCOC_ENABLE"
echo "TASK_QUEUE_ENABLE=$TASK_QUEUE_ENABLE"
echo "VLLM_RPC_GET_DATA_TIMEOUT_MS=$VLLM_RPC_GET_DATA_TIMEOUT_MS"
echo "VLLM_ALLOW_LONG_MAX_MODEL_LEN=$VLLM_ALLOW_LONG_MAX_MODEL_LEN"
echo "-------------------------------------------------------"

function extract_model_path() {
    local config_path=${1:-$DEFAULT_CONFIG_PATH}

    if [[ ! -f "$config_path" ]]; then
        echo "Error: Configuration file $config_path not found" >&2
        return 1
    fi

    python3 -c "
import yaml
import sys
try:
    with open('$config_path') as f:
        cfg = yaml.safe_load(f)
        print(cfg['chat']['model_path'])
except Exception as e:
    print(f'Error reading config: {e}', file=sys.stderr)
    sys.exit(1)
    "
}

function start_vllm_server() {
    local port=${1:-$DEFAULT_PORT}
    local config_path=${2:-$DEFAULT_CONFIG_PATH}
    local log_dir=${3:-$DEFAULT_LOG_DIR}

    # Create log directory
    mkdir -p "$log_dir"
    local log_path="$log_dir/server_api_0.log"
    echo "Log path: $log_path"

    # Extract model name from config
    local model_name
    model_name=$(extract_model_path "$config_path")
    if [[ $? -ne 0 ]] || [[ -z "$model_name" ]]; then
        echo "Error: Failed to extract model path from configuration" >&2
        return 1
    fi

    echo "Starting vLLM server with model: $model_name"
    echo "Server will be available at: http://localhost:$port"

    # Start vLLM server
    nohup python -m vllm.entrypoints.openai.api_server \
        --model "$model_name" \
        --trust-remote-code \
        --port "$port" \
        --swap-space 10 \
        --gpu-memory-utilization 0.7 \
        --num-scheduler-steps 8 \
        --enable-chunked-prefill \
        --enable-prefix-caching \
        --dtype auto \
        --tensor-parallel-size "$DEFAULT_TENSOR_PARALLEL_SIZE" \
        --max-num-batched-tokens 32768 \
        --max-num-seqs 16 \
        --distributed-executor-backend ray \
        --enforce-eager \
        --enable-reasoning \
        --reasoning-parser deepseek_r1 \
        --max-model-len "$DEFAULT_MAX_MODEL_LEN" \
        > "$log_path" 2>&1 &

    local server_pid=$!
    echo "vLLM server started with PID: $server_pid"
    echo "Monitor logs with: tail -f $log_path"

    return 0
}

function check_server_status() {
    local port=${1:-$DEFAULT_PORT}
    local max_attempts=30
    local attempt=0

    echo "Checking server status on port $port..."

    while [[ $attempt -lt $max_attempts ]]; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo "Server is ready and responding on port $port"
            return 0
        fi

        ((attempt++))
        echo "Attempt $attempt/$max_attempts: Server not ready yet, waiting..."
        sleep 2
    done

    echo "Error: Server failed to start within expected time" >&2
    return 1
}

# Main execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Starting vLLM server..."
    start_vllm_server "$@"

    if [[ $? -eq 0 ]]; then
        check_server_status
    else
        echo "Failed to start vLLM server" >&2
        exit 1
    fi
fi