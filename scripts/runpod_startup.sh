#!/usr/bin/env bash
# Minimal restart script for PersonaPlex on RunPod.
# Use this after runpod_setup.sh has already run (models are cached).
# Suitable for RunPod's "Start Command" field.
#
# Auto-detects RunPod persistent volume at /runpod for cached models.
# Auto-sources HF_TOKEN from container environment if set via RunPod UI.
#
# Usage: bash scripts/runpod_startup.sh [--gradio-tunnel] [--cpu-offload] [--port PORT]

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────

PORT=8998
GRADIO_TUNNEL=false
CPU_OFFLOAD=false

# ─── Auto-detect RunPod volume ───────────────────────────────────────────────

if [[ -d "/runpod/huggingface_cache" ]]; then
    CACHE_DIR="/runpod/huggingface_cache"
elif [[ -d "/runpod" ]]; then
    CACHE_DIR="/runpod/huggingface_cache"
else
    CACHE_DIR="/root/.cache"
fi

# ─── Usage ───────────────────────────────────────────────────────────────────

usage() {
    cat <<EOF
PersonaPlex RunPod Startup (restart after pod sleep/wake)

Usage: bash $0 [OPTIONS]

Options:
  --gradio-tunnel   Enable Gradio tunnel for a public URL
  --cpu-offload     Offload model layers to CPU (for GPUs < 24 GB)
  --port PORT       Server port (default: 8998)
  --help            Show this help message

Environment:
  HF_TOKEN          Required. Hugging Face token.
                    (auto-sourced from container env if set via RunPod UI)

This script assumes runpod_setup.sh has been run at least once
(system deps installed, package installed, models cached).
EOF
    exit 0
}

# ─── Parse args ──────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gradio-tunnel) GRADIO_TUNNEL=true; shift ;;
        --cpu-offload)   CPU_OFFLOAD=true; shift ;;
        --port)          PORT="$2"; shift 2 ;;
        --help)          usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ─── Helpers ─────────────────────────────────────────────────────────────────

info()  { echo -e "\033[1;34m[INFO]\033[0m $*"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

# ─── Validate HF_TOKEN (auto-source from container env) ─────────────────────

if [[ -z "${HF_TOKEN:-}" ]] && [[ -f /proc/1/environ ]]; then
    HF_TOKEN="$(tr '\0' '\n' < /proc/1/environ | grep '^HF_TOKEN=' | cut -d= -f2- || true)"
    export HF_TOKEN
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    error "HF_TOKEN is not set."
    echo "  export HF_TOKEN=hf_..."
    exit 1
fi

# ─── Set cache directory ────────────────────────────────────────────────────

export HF_HOME="$CACHE_DIR"

# ─── Launch ──────────────────────────────────────────────────────────────────

SERVER_ARGS=(
    --host 0.0.0.0
    --port "$PORT"
)

if [[ "$GRADIO_TUNNEL" == true ]]; then
    SERVER_ARGS+=(--gradio-tunnel)
fi

if [[ "$CPU_OFFLOAD" == true ]]; then
    SERVER_ARGS+=(--cpu-offload)
fi

info "Starting PersonaPlex server on port $PORT..."
info "  Cache: $CACHE_DIR"
exec python -m moshi.server "${SERVER_ARGS[@]}"
