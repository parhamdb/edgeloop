#!/usr/bin/env bash
#
# Benchmark: speculative decoding ON vs OFF with llama-server
#
# Uses Ollama's GGUF model blobs so no extra downloads needed.
# Builds llama.cpp from source with CUDA if not already built.
#
# Usage: ./scripts/bench_speculative.sh [--skip-build]
#
# Target model: qwen2.5-coder:7b  (main model)
# Draft model:  qwen2.5-coder:0.5b (speculative draft)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAMA_DIR="$PROJECT_DIR/.llama.cpp"
LLAMA_SERVER="$LLAMA_DIR/build/bin/llama-server"
PORT_NORMAL=8090
PORT_SPEC=8091
N_GPU_LAYERS=99
N_CTX=4096
N_DRAFT=8

# GGUF model paths from Ollama blobs
MODEL_7B="/var/lib/ollama/blobs/sha256-60e05f2100071479f596b964f89f510f057ce397ea22f2833a0cfe029bfc2463"
MODEL_05B="/var/lib/ollama/blobs/sha256-20693aeb02c63304e263a72453b6ab89e1c700a87c6948cac523ac1e6f7cade0"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[bench]${NC} $*"; }
ok()   { echo -e "${GREEN}[bench]${NC} $*"; }
warn() { echo -e "${YELLOW}[bench]${NC} $*"; }
err()  { echo -e "${RED}[bench]${NC} $*" >&2; }

cleanup() {
    log "Cleaning up..."
    kill "$PID_NORMAL" 2>/dev/null || true
    kill "$PID_SPEC" 2>/dev/null || true
    wait "$PID_NORMAL" 2>/dev/null || true
    wait "$PID_SPEC" 2>/dev/null || true
}
trap cleanup EXIT

PID_NORMAL=""
PID_SPEC=""

# --- Build llama.cpp ---

build_llama() {
    if [[ -x "$LLAMA_SERVER" ]]; then
        ok "llama-server already built at $LLAMA_SERVER"
        return
    fi

    log "Cloning llama.cpp..."
    if [[ ! -d "$LLAMA_DIR" ]]; then
        git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
    fi

    log "Building llama.cpp with CUDA..."
    cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
        -DGGML_CUDA=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_CURL=OFF \
        2>&1 | tail -5

    cmake --build "$LLAMA_DIR/build" --target llama-server -j "$(nproc)" 2>&1 | tail -5

    if [[ ! -x "$LLAMA_SERVER" ]]; then
        err "Build failed — llama-server not found at $LLAMA_SERVER"
        exit 1
    fi
    ok "Built llama-server successfully"
}

# --- Verify models exist ---

check_models() {
    if [[ ! -f "$MODEL_7B" ]]; then
        err "Target model not found: $MODEL_7B"
        err "Run: ollama pull qwen2.5-coder:7b"
        exit 1
    fi
    if [[ ! -f "$MODEL_05B" ]]; then
        err "Draft model not found: $MODEL_05B"
        err "Run: ollama pull qwen2.5-coder:0.5b"
        exit 1
    fi
    ok "Models found"
}

# --- Start servers ---

wait_for_server() {
    local port=$1
    local name=$2
    local max_wait=120
    local waited=0
    while ! curl -sf "http://localhost:$port/health" >/dev/null 2>&1; do
        sleep 1
        waited=$((waited + 1))
        if [[ $waited -ge $max_wait ]]; then
            err "$name failed to start within ${max_wait}s"
            exit 1
        fi
    done
    ok "$name ready on port $port (${waited}s)"
}

start_normal() {
    log "Starting llama-server (normal, no speculative decoding)..."
    "$LLAMA_SERVER" \
        --model "$MODEL_7B" \
        --port "$PORT_NORMAL" \
        --n-gpu-layers "$N_GPU_LAYERS" \
        --ctx-size "$N_CTX" \
        --flash-attn on \
        --log-disable \
        >/dev/null 2>&1 &
    PID_NORMAL=$!
    wait_for_server "$PORT_NORMAL" "Normal server"
}

start_speculative() {
    log "Starting llama-server (speculative decoding, draft=$N_DRAFT)..."
    "$LLAMA_SERVER" \
        --model "$MODEL_7B" \
        --model-draft "$MODEL_05B" \
        --port "$PORT_SPEC" \
        --n-gpu-layers "$N_GPU_LAYERS" \
        --ctx-size "$N_CTX" \
        --draft-max "$N_DRAFT" \
        --draft-min 1 \
        --flash-attn on \
        --log-disable \
        >/dev/null 2>&1 &
    PID_SPEC=$!
    wait_for_server "$PORT_SPEC" "Speculative server"
}

# --- Benchmark helpers ---

# Send a completion request and measure time + tokens
bench_request() {
    local port=$1
    local prompt=$2
    local max_tokens=${3:-256}

    local start_ns
    start_ns=$(date +%s%N)

    local response
    response=$(curl -sf "http://localhost:$port/completion" \
        -H "Content-Type: application/json" \
        -d "{
            \"prompt\": \"$prompt\",
            \"n_predict\": $max_tokens,
            \"temperature\": 0.1,
            \"cache_prompt\": true,
            \"stream\": false
        }" 2>/dev/null)

    local end_ns
    end_ns=$(date +%s%N)
    local elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))

    local tokens_predicted
    tokens_predicted=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('timings',{}).get('predicted_n',0))" 2>/dev/null || echo "0")

    local prompt_eval_ms
    prompt_eval_ms=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d.get('timings',{}).get('prompt_ms',0):.1f}\")" 2>/dev/null || echo "0")

    local predicted_ms
    predicted_ms=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d.get('timings',{}).get('predicted_ms',0):.1f}\")" 2>/dev/null || echo "0")

    local predicted_per_s
    predicted_per_s=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d.get('timings',{}).get('predicted_per_second',0):.1f}\")" 2>/dev/null || echo "0")

    local prompt_per_s
    prompt_per_s=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d.get('timings',{}).get('prompt_per_second',0):.1f}\")" 2>/dev/null || echo "0")

    echo "${elapsed_ms}|${tokens_predicted}|${prompt_eval_ms}|${predicted_ms}|${predicted_per_s}|${prompt_per_s}"
}

# Run a test case N times and report stats
run_test() {
    local name=$1
    local port=$2
    local prompt=$3
    local max_tokens=${4:-256}
    local runs=${5:-3}

    local total_ms=0
    local total_tps=0
    local results=()

    # Warmup run (primes the KV cache)
    bench_request "$port" "$prompt" "$max_tokens" >/dev/null 2>&1

    for i in $(seq 1 "$runs"); do
        local result
        result=$(bench_request "$port" "$prompt" "$max_tokens")
        results+=("$result")

        local ms tok prompt_ms pred_ms tps prompt_tps
        IFS='|' read -r ms tok prompt_ms pred_ms tps prompt_tps <<< "$result"
        total_ms=$((total_ms + ms))
        total_tps=$(python3 -c "print(${total_tps} + ${tps})")
    done

    local avg_ms=$((total_ms / runs))
    local avg_tps
    avg_tps=$(python3 -c "print(f'{${total_tps} / ${runs}:.1f}')")

    # Print last run's details
    local ms tok prompt_ms pred_ms tps prompt_tps
    IFS='|' read -r ms tok prompt_ms pred_ms tps prompt_tps <<< "${results[-1]}"

    printf "  %-12s  avg=%4dms  tok/s=%6s  prompt=%sms  gen=%sms  tokens=%s\n" \
        "$name" "$avg_ms" "$avg_tps" "$prompt_ms" "$pred_ms" "$tok"
}

# --- Benchmark suite ---

PROMPTS=(
    "Explain what a hash table is in one paragraph.|256|short_gen"
    "Write a Python function that implements binary search, with docstring and type hints. Include edge cases.|512|code_gen"
    "What are the main differences between TCP and UDP? List 5 differences with explanations.|512|medium_gen"
    "Write a Rust function that reads a TOML file, parses it into a struct with serde, handles errors properly, and includes unit tests.|1024|long_code"
)

run_benchmark_suite() {
    local label=$1
    local port=$2

    echo ""
    echo "=== $label (port $port) ==="
    echo ""

    for entry in "${PROMPTS[@]}"; do
        IFS='|' read -r prompt max_tokens test_name <<< "$entry"
        run_test "$test_name" "$port" "$prompt" "$max_tokens" 3
    done
}

# --- Edgeloop integration benchmark ---

run_edgeloop_bench() {
    local label=$1
    local port=$2

    echo ""
    echo "=== $label — Edgeloop Agent Roundtrip ==="
    echo ""

    # Use the Rust benchmark via llama-server backend
    cd "$PROJECT_DIR"
    LLAMA_SERVER_PORT=$port cargo test --test speculative_bench -- --nocapture 2>&1 | grep -E "^\s+(simple|tool|multi)" || true
}

# --- Main ---

main() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║   Speculative Decoding Benchmark                        ║"
    echo "║   Target: qwen2.5-coder:7b  Draft: qwen2.5-coder:0.5b  ║"
    echo "║   GPU: RTX 4070  Draft tokens: $N_DRAFT                     ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    if [[ "${1:-}" != "--skip-build" ]]; then
        build_llama
    fi
    check_models

    # Stop Ollama to free GPU memory
    warn "Stopping Ollama to free GPU VRAM..."
    sudo systemctl stop ollama 2>/dev/null || true
    sleep 2

    start_normal
    start_speculative

    # Raw llama-server benchmarks
    run_benchmark_suite "NORMAL (no speculative decoding)" "$PORT_NORMAL"
    run_benchmark_suite "SPECULATIVE (draft=$N_DRAFT, qwen2.5-coder:0.5b)" "$PORT_SPEC"

    # Edgeloop agent benchmarks (if test exists)
    if [[ -f "$PROJECT_DIR/tests/speculative_bench.rs" ]]; then
        run_edgeloop_bench "NORMAL" "$PORT_NORMAL"
        run_edgeloop_bench "SPECULATIVE" "$PORT_SPEC"
    fi

    # Comparison summary
    echo ""
    echo "=== COMPARISON SUMMARY ==="
    echo ""
    echo "Run complete. Compare the avg ms and tok/s values above."
    echo "Speculative decoding helps most with long generation tasks"
    echo "where the draft model can predict tokens the target would generate."
    echo ""

    # Restart Ollama
    warn "Restarting Ollama..."
    sudo systemctl start ollama 2>/dev/null || true

    ok "Done!"
}

main "$@"
