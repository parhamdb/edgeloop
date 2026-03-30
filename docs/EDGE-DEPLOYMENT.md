# Edge Deployment Guide

Optimizing edgeloop for maximum speed on edge devices.

## Quick Summary

| Optimization | Where | Impact | Cost |
|-------------|-------|--------|------|
| `--mlock` | llama-server flag | **Cold: 6ms vs 24ms, +15% gen speed** | Locks RAM |
| `-ctk q4_0 -ctv q4_0` | llama-server flag | **75% less KV cache RAM** | None measured |
| `slot_id = 0` | edgeloop config | **Warm: 3ms vs 24ms prefill** | Dedicates slot |
| `n_keep = <sys_tokens>` | edgeloop config | Pins system prompt in cache | None |
| `keep_alive = "-1"` | edgeloop config (Ollama) | Eliminates model reload | Holds GPU RAM |
| Persistent history | Built-in (v0.1) | **84% prefill savings at turn 10** | More RAM per session |
| Stop sequences | Built-in (v0.1) | Prevents over-generation | None |

## llama-server Launch Configs

### Minimal edge device (1-2GB RAM, e.g., Pi 3)

```bash
llama-server -m model-q4.gguf \
  -ngl 0 \              # CPU only (no GPU)
  -c 1024 \             # Small context — saves RAM
  -np 1 \               # Single slot — one agent
  -ctk q4_0 -ctv q4_0 \ # Quantized KV cache — 75% less RAM
  -t 4 \                # Match CPU cores
  --mlock \             # Lock in RAM, prevent swapping
  --no-webui
```

### Mid-range edge (4-8GB, e.g., Pi 4/5, Jetson Nano)

```bash
llama-server -m model-q4.gguf \
  -ngl 99 \             # Offload all layers to GPU
  -c 4096 \             # Full context
  -np 2 \               # 2 slots — 2 concurrent agents
  -ctk q8_0 -ctv q8_0 \ # Quantized KV — 50% less VRAM
  --mlock \
  --no-webui
```

### High-end edge (16GB+, e.g., Jetson Orin/Thor)

```bash
llama-server -m model-q4.gguf \
  -ngl 99 \
  -c 8192 \             # Large context
  -np 4 \               # 4 concurrent agents
  -fa on \              # Flash attention
  --mlock \
  --no-webui
```

## edgeloop Config for Maximum Speed

```toml
transports = ["cli"]
tool_packages = ["tools/filesystem"]

[agent]
system_prompt = "You are helpful."  # Keep SHORT — every token costs prefill
template = "chatml"
max_tokens = 2048                   # Don't over-allocate context
max_iterations = 5                  # Limit tool call loops
temperature = 0.1                   # Lower = faster sampling

[backend]
type = "llama-server"
endpoint = "http://localhost:8080"
slot_id = 0                         # Pin to dedicated cache slot
n_keep = 50                         # Pin system prompt in cache permanently

[transport.cli]
prompt = "> "
```

For Ollama:
```toml
[backend]
type = "ollama"
endpoint = "http://localhost:11434"
model = "qwen3:1.7b"               # Smaller model = faster
keep_alive = "-1"                   # Never unload — eliminates cold start
```

## Model Selection for Edge

| Device RAM | Recommended model | Size | Speed |
|-----------|------------------|------|-------|
| 1-2GB | qwen2.5-0.5b Q4 | 400MB | ~500 tok/s CPU |
| 2-4GB | qwen2.5-1.5b Q4 | 1GB | ~300 tok/s GPU |
| 4-8GB | qwen3:1.7b or qwen2.5-coder:7b Q4 | 1.4-4.7GB | ~100-300 tok/s |
| 8-16GB | qwen2.5:14b Q4 or qwen3.5:9b | 9-6.6GB | ~50-100 tok/s |
| 16GB+ | qwen3.5:35b-a3b (MoE) | 23GB | ~30 tok/s |

## KV Cache Memory Calculator

```
KV cache size = 2 * n_layers * n_kv_heads * head_dim * n_ctx * n_slots * bytes_per_element

For qwen2.5-1.5b, ctx=4096, 4 slots:
  f16:  2 * 28 * 2 * 128 * 4096 * 4 * 2 bytes = 112 MB
  q8_0: 2 * 28 * 2 * 128 * 4096 * 4 * 1 bytes =  56 MB
  q4_0: 2 * 28 * 2 * 128 * 4096 * 4 * 0.5      =  28 MB
```

On a 4GB device: q4_0 KV cache saves 84MB — that's 2% of total RAM freed.

## Reducing Tool Schema Tokens

edgeloop uses compact tool format (~8 tokens per tool). With 15 tools, that's ~120 tokens in the system prompt. To minimize:

1. Keep tool descriptions short (5-10 words)
2. Use short parameter names
3. Only include tools the agent actually needs
4. Use separate tool_packages for different agent roles

## Binary Size Optimization

```bash
# Minimal binary for constrained devices
cargo build --release --no-default-features --features "llama-server,cli-transport"
# Result: ~2MB (just llama-server + CLI, no Ollama/OpenAI/MQTT/WebSocket)

# Further size reduction with nightly Rust
cargo +nightly build --release -Z build-std=std,panic_abort \
  -Z build-std-features=panic_immediate_abort \
  --no-default-features --features "llama-server,cli-transport"
```

## Deployment Checklist

1. Choose model size for your RAM budget (see table above)
2. Quantize KV cache if VRAM is tight (`-ctk q4_0 -ctv q4_0`)
3. Use `--mlock` to prevent swapping
4. Pin agent to a slot (`slot_id = 0`)
5. Set `n_keep` to your system prompt token count
6. Keep system prompt and tool descriptions short
7. Set `max_iterations = 5` (3-5 is enough for most tasks)
8. Use `temperature = 0.1` for tool-calling agents
9. Build edgeloop with only the features you need
