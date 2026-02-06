#!/usr/bin/env bash
# Full automated setup for PersonaPlex on a RunPod GPU pod.
# Usage: bash scripts/runpod_setup.sh [--gradio-tunnel] [--cpu-offload] [--port PORT]
#
# Prerequisites:
#   - RunPod GPU pod (A10G / L4 24 GB recommended)
#   - HF_TOKEN environment variable set (needs access to nvidia/personaplex-7b-v1)
#
# This script is idempotent — safe to re-run after a pod restart.
# Auto-detects RunPod persistent volume at /runpod for cache & repo storage.

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────

REPO_URL="https://github.com/Usoramara/personaplex.git"
MODEL_REPO="nvidia/personaplex-7b-v1"
PORT=8998
GRADIO_TUNNEL=false
CPU_OFFLOAD=false

# ─── Auto-detect RunPod volume ───────────────────────────────────────────────

if [[ -d "/runpod" ]]; then
    VOLUME_DIR="/runpod"
else
    VOLUME_DIR="/root"
fi
REPO_DIR="$VOLUME_DIR/personaplex"
CACHE_DIR="$VOLUME_DIR/huggingface_cache"

# ─── Usage ───────────────────────────────────────────────────────────────────

usage() {
    cat <<EOF
PersonaPlex RunPod Setup

Usage: bash $0 [OPTIONS]

Options:
  --gradio-tunnel   Enable Gradio tunnel for a public URL
  --cpu-offload     Offload model layers to CPU (for GPUs < 24 GB)
  --port PORT       Server port (default: 8998)
  --help            Show this help message

Environment:
  HF_TOKEN          Required. Hugging Face token with access to $MODEL_REPO
                    (auto-sourced from container env if set via RunPod UI)

Volume:
  If /runpod exists, repo and model cache are stored there (persists across
  pod termination). Otherwise falls back to /root.

Example:
  export HF_TOKEN=hf_...
  bash $0 --gradio-tunnel
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

# ─── Step 1: Validate HF_TOKEN (auto-source from container env) ─────────────

if [[ -z "${HF_TOKEN:-}" ]] && [[ -f /proc/1/environ ]]; then
    HF_TOKEN="$(tr '\0' '\n' < /proc/1/environ | grep '^HF_TOKEN=' | cut -d= -f2- || true)"
    export HF_TOKEN
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    error "HF_TOKEN is not set."
    echo "  Get a token at https://huggingface.co/settings/tokens"
    echo "  Then: export HF_TOKEN=hf_..."
    exit 1
fi
info "HF_TOKEN is set."

# ─── Step 2: Install system dependencies ─────────────────────────────────────

if dpkg -s libopus-dev &>/dev/null; then
    info "System deps already installed (libopus-dev found)."
else
    info "Installing system dependencies..."
    apt-get update -qq
    apt-get install -y --no-install-recommends build-essential pkg-config libopus-dev
    info "System dependencies installed."
fi

# ─── Step 3: Clone the repo ─────────────────────────────────────────────────

# Detect if we're already inside the repo
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/../moshi/moshi/server.py" ]]; then
    REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
    info "Running from inside the repo at $REPO_DIR — skipping clone."
elif [[ -d "$REPO_DIR/moshi" ]]; then
    info "Repo already cloned at $REPO_DIR — pulling latest..."
    git -C "$REPO_DIR" pull --ff-only || true
else
    info "Cloning PersonaPlex to $REPO_DIR..."
    git clone "$REPO_URL" "$REPO_DIR"
    info "Clone complete."
fi

# ─── Step 4: Install Python package ─────────────────────────────────────────

# Install gradio first if tunnel is requested (so moshi's pinned deps take priority)
if [[ "$GRADIO_TUNNEL" == true ]]; then
    info "Installing gradio (required for tunnel)..."
    pip install --quiet gradio
fi

info "Installing moshi-personaplex Python package..."
pip install --quiet "$REPO_DIR/moshi/."
info "Python package installed."

# ─── Step 5: Pre-download model weights ─────────────────────────────────────

export HF_HOME="$CACHE_DIR"

info "Pre-downloading model weights from $MODEL_REPO..."
info "  Cache directory: $CACHE_DIR"
info "  (This may take a few minutes on first run — skips if already cached)"
python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_REPO', token='$HF_TOKEN')"
info "Model weights cached."

# ─── Step 6: Launch the server ───────────────────────────────────────────────

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

echo ""
info "========================================="
info " Starting PersonaPlex server"
info "========================================="
info "  Port: $PORT"
info "  Gradio tunnel: $GRADIO_TUNNEL"
info "  CPU offload: $CPU_OFFLOAD"
info "  Volume: $VOLUME_DIR"
info "  Cache: $CACHE_DIR"
echo ""
info "Access via RunPod proxy:"
info "  https://<POD_ID>-${PORT}.proxy.runpod.net/"
echo ""
if [[ "$GRADIO_TUNNEL" == true ]]; then
    info "A Gradio tunnel URL will appear in the logs below."
    echo ""
fi

exec python -m moshi.server "${SERVER_ARGS[@]}"
