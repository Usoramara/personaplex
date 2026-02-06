#!/usr/bin/env bash
# Full automated setup for PersonaPlex on a RunPod GPU pod.
# Usage: bash scripts/runpod_setup.sh [--gradio-tunnel] [--cpu-offload] [--port PORT]
#
# Prerequisites:
#   - RunPod GPU pod (A10G / L4 24 GB recommended)
#   - HF_TOKEN environment variable set (needs access to nvidia/personaplex-7b-v1)
#
# This script is idempotent — safe to re-run after a pod restart.

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────

REPO_URL="https://github.com/Usoramara/personaplex.git"
REPO_DIR="/root/personaplex"
MODEL_REPO="nvidia/personaplex-7b-v1"
PORT=8998
GRADIO_TUNNEL=false
CPU_OFFLOAD=false
SSL_DIR=""

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

# ─── Step 1: Validate HF_TOKEN ──────────────────────────────────────────────

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

info "Installing moshi-personaplex Python package..."
pip install --quiet "$REPO_DIR/moshi/."
info "Python package installed."

# Install gradio if tunnel is requested
if [[ "$GRADIO_TUNNEL" == true ]]; then
    info "Installing gradio (required for tunnel)..."
    pip install --quiet gradio
fi

# ─── Step 5: Pre-download model weights ─────────────────────────────────────

info "Pre-downloading model weights from $MODEL_REPO..."
info "  (This may take a few minutes on first run — files are cached in /root/.cache)"
huggingface-cli download "$MODEL_REPO" --token "$HF_TOKEN"
info "Model weights cached."

# ─── Step 6: Prepare SSL directory ───────────────────────────────────────────

SSL_DIR="$(mktemp -d)/ssl"
mkdir -p "$SSL_DIR"
info "SSL cert directory: $SSL_DIR"

# ─── Step 7: Launch the server ───────────────────────────────────────────────

SERVER_ARGS=(
    --host 0.0.0.0
    --port "$PORT"
    --ssl "$SSL_DIR"
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
echo ""
info "Access via RunPod proxy:"
info "  https://<POD_ID>-${PORT}.proxy.runpod.net/"
echo ""
if [[ "$GRADIO_TUNNEL" == true ]]; then
    info "A Gradio tunnel URL will appear in the logs below."
    echo ""
fi

exec python -m moshi.server "${SERVER_ARGS[@]}"
