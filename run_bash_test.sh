#!/bin/bash

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get test file from argument or use default (CP benchmark harness lives here)
TEST_FILE="${1:-tests/test_qwen3_0.6B_fsdp_distributed.py}"

# CP benchmark defaults (only used when TEST_FILE matches the qwen3 fsdp distributed test)
CP_BENCH="${SLIME_TEST_CP_BENCH:-}"
CP_BENCH_NUM_GPUS="${SLIME_TEST_CP_BENCH_NUM_GPUS:-4}"
CP_BENCH_PROMPT_LEN="${SLIME_TEST_CP_BENCH_PROMPT_LEN:-256}"
CP_BENCH_SEQLENS="${SLIME_TEST_CP_BENCH_SEQLENS:-4096,8192}"
CP_BENCH_NUM_ROLLOUT="${SLIME_TEST_CP_BENCH_NUM_ROLLOUT:-10}"
CP_BENCH_WARMUP="${SLIME_TEST_CP_BENCH_WARMUP:-2}"
CP_BENCH_ROLLOUT_BATCH_SIZE="${SLIME_TEST_CP_BENCH_ROLLOUT_BATCH_SIZE:-2}"
CP_BENCH_N_SAMPLES_PER_PROMPT="${SLIME_TEST_CP_BENCH_N_SAMPLES_PER_PROMPT:-2}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running Test in Docker${NC}"
echo -e "${BLUE}Test: $TEST_FILE${NC}"
echo -e "${BLUE}========================================${NC}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if we have enough GPUs available
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "0")
REQUIRED_GPUS=2
if [[ "$TEST_FILE" == "tests/test_qwen3_0.6B_fsdp_distributed.py" ]]; then
    REQUIRED_GPUS="$CP_BENCH_NUM_GPUS"
fi
if [ "$GPU_COUNT" -lt "$REQUIRED_GPUS" ]; then
    echo -e "${RED}Warning: This test requires at least $REQUIRED_GPUS GPUs${NC}"
    echo -e "${RED}Found: $GPU_COUNT GPUs${NC}"
    echo -e "${RED}Continuing anyway...${NC}"
fi

echo -e "${GREEN}Found $GPU_COUNT GPUs${NC}"
echo -e "${GREEN}Mounting local code from: $SCRIPT_DIR${NC}"
echo ""

# Cache model downloads across runs (optional but recommended)
MODELS_CACHE_DIR="${SLIME_DOCKER_MODELS_CACHE_DIR:-$SCRIPT_DIR/.cache/slime_models}"
mkdir -p "$MODELS_CACHE_DIR"
echo -e "${GREEN}Mounting models cache: $MODELS_CACHE_DIR -> /root/models${NC}"
echo ""

# Create temp directory on ephemeral storage if available, otherwise use /tmp
if [ -d "/ephemeral" ]; then
    TMP_DIR="/ephemeral/slime_test_tmp_$$"
    echo -e "${GREEN}Using ephemeral storage for /tmp: $TMP_DIR${NC}"
else
    TMP_DIR="/tmp/slime_test_tmp_$$"
    echo -e "${GREEN}Using host /tmp: $TMP_DIR${NC}"
fi
mkdir -p "$TMP_DIR"

# Keep tmp by default for the CP benchmark so perf JSONLs are not deleted.
KEEP_TMP="${SLIME_KEEP_TMP:-}"
if [[ -z "$KEEP_TMP" && "$TEST_FILE" == "tests/test_qwen3_0.6B_fsdp_distributed.py" ]]; then
    # This script defaults to enabling CP benchmark for this test file.
    if [[ -z "$CP_BENCH" || "$CP_BENCH" == "1" ]]; then
        KEEP_TMP="1"
    fi
fi

# Cleanup function
cleanup() {
    if [[ "${KEEP_TMP:-0}" == "1" ]]; then
        echo -e "${BLUE}Keeping temporary directory: $TMP_DIR${NC}"
        return
    fi
    echo -e "${BLUE}Cleaning up temporary directory: $TMP_DIR${NC}"
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

# Helper to run a command inside the docker image with common mounts/env.
# Extra env vars can be passed as a newline-delimited string of "KEY=VALUE".
run_in_docker() {
    local extra_env_vars="$1"
    local inner_cmd="$2"

    local docker_env_flags=()
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        docker_env_flags+=(-e "$line")
    done <<< "$extra_env_vars"

    local docker_gpus_arg="all"
    if [[ -n "${SLIME_DOCKER_GPU_DEVICES:-}" ]]; then
        docker_gpus_arg="device=${SLIME_DOCKER_GPU_DEVICES}"
        docker_env_flags+=(-e "CUDA_VISIBLE_DEVICES=${SLIME_DOCKER_GPU_DEVICES}")
    fi

    docker run --rm \
        --gpus "$docker_gpus_arg" \
        --ipc=host \
        --shm-size=16g \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v "$SCRIPT_DIR:/workspace/slime" \
        -v "$TMP_DIR:/tmp" \
        -v "$MODELS_CACHE_DIR:/root/models" \
        -e "http_proxy=${http_proxy-}" \
        -e "https_proxy=${https_proxy-}" \
        -e "HTTP_PROXY=${HTTP_PROXY-}" \
        -e "HTTPS_PROXY=${HTTPS_PROXY-}" \
        -e "PYTHONPATH=/root/Megatron-LM/" \
        "${docker_env_flags[@]}" \
        -w /workspace/slime \
        slimerl/slime:latest \
        bash -c "$inner_cmd"
}

# Run Docker container with local code mounted
if [[ "$TEST_FILE" == "tests/test_qwen3_0.6B_fsdp_distributed.py" ]]; then
    # Default to CP benchmark mode when running this test via the helper script.
    if [[ -z "$CP_BENCH" ]]; then
        CP_BENCH="1"
    fi

    if [[ "$CP_BENCH" != "1" ]]; then
        echo -e "${BLUE}CP benchmark disabled (SLIME_TEST_CP_BENCH=$CP_BENCH); running test normally.${NC}"
        run_in_docker "" "
            set -e
            echo \"Installing slime in editable mode...\"
            pip install -e . -q
            echo \"\"
            echo \"=========================================\"
            echo \"Running test: $TEST_FILE\"
            echo \"=========================================\"
            echo \"\"
            python \"$TEST_FILE\"
        "
    else
        echo -e "${BLUE}CP benchmark enabled.${NC}"
        echo -e "${GREEN}Benchmark config:${NC}"
        echo -e "${GREEN}  num_gpus=$CP_BENCH_NUM_GPUS prompt_len=$CP_BENCH_PROMPT_LEN seqlens=$CP_BENCH_SEQLENS${NC}"
        echo -e "${GREEN}  num_rollout=$CP_BENCH_NUM_ROLLOUT warmup=$CP_BENCH_WARMUP rb=$CP_BENCH_ROLLOUT_BATCH_SIZE nsp=$CP_BENCH_N_SAMPLES_PER_PROMPT${NC}"
        echo ""

        run_in_docker \
            "SLIME_TEST_CP_BENCH=1
SLIME_TEST_CP_BENCH_NUM_GPUS=$CP_BENCH_NUM_GPUS
SLIME_TEST_CP_BENCH_PROMPT_LEN=$CP_BENCH_PROMPT_LEN
SLIME_TEST_CP_BENCH_SEQLENS=$CP_BENCH_SEQLENS
SLIME_TEST_CP_BENCH_NUM_ROLLOUT=$CP_BENCH_NUM_ROLLOUT
SLIME_TEST_CP_BENCH_WARMUP=$CP_BENCH_WARMUP
SLIME_TEST_CP_BENCH_ROLLOUT_BATCH_SIZE=$CP_BENCH_ROLLOUT_BATCH_SIZE
SLIME_TEST_CP_BENCH_N_SAMPLES_PER_PROMPT=$CP_BENCH_N_SAMPLES_PER_PROMPT" \
            "
            set -e
            echo \"Installing slime in editable mode...\"
            pip install -e . -q

            IFS=',' read -ra SEQ_ARR <<< \"\$SLIME_TEST_CP_BENCH_SEQLENS\"
            for SEQ in \"\${SEQ_ARR[@]}\"; do
                SEQ=\"\$(echo \"\$SEQ\" | xargs)\"
                [[ -z \"\$SEQ\" ]] && continue

                RESP_LEN=\$((SEQ - SLIME_TEST_CP_BENCH_PROMPT_LEN))
                if [[ \"\$RESP_LEN\" -le 0 ]]; then
                    echo \"Invalid seqlen=\$SEQ with prompt_len=\$SLIME_TEST_CP_BENCH_PROMPT_LEN (response_len would be \$RESP_LEN).\" >&2
                    exit 2
                fi

                export SLIME_TEST_CP_BENCH_DIR=\"/tmp/cp_bench_seqlen_\${SEQ}\"
                export SLIME_TEST_CP_BENCH_RESPONSE_LEN=\"\$RESP_LEN\"

                echo \"\"
                echo \"=========================================\"
                echo \"CP benchmark: total_seqlen=\$SEQ (prompt=\$SLIME_TEST_CP_BENCH_PROMPT_LEN response=\$RESP_LEN)\"
                echo \"Artifacts: \$SLIME_TEST_CP_BENCH_DIR\"
                echo \"=========================================\"
                echo \"\"

                python \"$TEST_FILE\"
            done
            "

        echo -e "${GREEN}Artifacts written under:${NC}"
        IFS=',' read -ra SEQ_ARR <<< "$CP_BENCH_SEQLENS"
        for SEQ in "${SEQ_ARR[@]}"; do
            SEQ="$(echo "$SEQ" | xargs)"
            [[ -z "$SEQ" ]] && continue
            echo -e "${GREEN}  ${NC}$TMP_DIR/cp_bench_seqlen_${SEQ}/perf_cp1.jsonl"
            echo -e "${GREEN}  ${NC}$TMP_DIR/cp_bench_seqlen_${SEQ}/perf_cp2.jsonl"
        done
    fi
else
    run_in_docker "" "
        set -e

        echo \"Installing slime in editable mode...\"
        pip install -e . -q

        echo \"\"
        echo \"=========================================\"
        echo \"Running test: $TEST_FILE\"
        echo \"=========================================\"
        echo \"\"

        python \"$TEST_FILE\"
    "
fi

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Test completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Test failed with exit code: $EXIT_CODE${NC}"
    echo -e "${RED}========================================${NC}"
fi

exit $EXIT_CODE
