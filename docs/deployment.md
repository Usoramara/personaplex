# Deployment Guide

This guide covers deploying PersonaPlex across different environments, from local development to cloud GPU instances.

## Table of Contents

- [Environment Variables](#environment-variables)
- [Local Setup](#local-setup)
- [Docker Deployment](#docker-deployment)
- [RunPod Deployment](#runpod-deployment)
- [AWS / GCP](#aws--gcp)
- [Gradio Tunnel](#gradio-tunnel)
- [Tested GPU Configurations](#tested-gpu-configurations)
- [Troubleshooting](#troubleshooting)

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | Yes | — | Hugging Face access token. Must accept the [model license](https://huggingface.co/nvidia/personaplex-7b-v1) first. |
| `NO_TORCH_COMPILE` | No | `0` | Set to `1` to disable `torch.compile`. Useful in Docker or when compilation is slow/broken. |
| `CUDA_VISIBLE_DEVICES` | No | all GPUs | Comma-separated GPU indices (e.g., `0` or `0,1`). PersonaPlex uses a single GPU. |

## Local Setup

### Linux with GPU (recommended)

```bash
# 1. Clone the repository
git clone https://github.com/justwybe/Nvidia-natural-conversation-model.git
cd Nvidia-natural-conversation-model

# 2. Install system dependencies
sudo apt install libopus-dev

# 3. Install Python package
pip install moshi/.

# 4. (Blackwell GPUs only) Install compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 5. Set your Hugging Face token
export HF_TOKEN=<YOUR_TOKEN>

# 6a. Launch the server (interactive mode)
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR"

# 6b. Or run offline inference
python -m moshi.offline \
  --voice-prompt "NATF2.pt" \
  --input-wav "assets/test/input_assistant.wav" \
  --seed 42424242 \
  --output-wav "output.wav" \
  --output-text "output.json"
```

### Linux CPU Offload (8 GB+ VRAM)

If your GPU has limited VRAM, use `--cpu-offload` to keep some model layers on CPU:

```bash
pip install accelerate
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR" --cpu-offload
```

### macOS CPU-Only (offline inference only)

The server requires Opus codec streaming which works on macOS, but real-time performance is limited to offline inference:

```bash
brew install opus
pip install moshi/.

# Install CPU-only PyTorch if needed
pip install torch torchvision torchaudio

python -m moshi.offline \
  --device cpu \
  --voice-prompt "NATF2.pt" \
  --input-wav "assets/test/input_assistant.wav" \
  --seed 42424242 \
  --output-wav "output.wav" \
  --output-text "output.json"
```

> **Note:** CPU-only inference is significantly slower than GPU and is only recommended for testing. Expect ~10-30x slower than real-time.

## Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t personaplex .

# Run with GPU access
docker run --gpus all -p 8998:8998 \
  -e HF_TOKEN=$HF_TOKEN \
  -e NO_TORCH_COMPILE=1 \
  -v $(pwd)/.cache:/root/.cache \
  personaplex
```

### Docker Compose

The simplest way to deploy:

```bash
# 1. Clone the repo
git clone https://github.com/justwybe/Nvidia-natural-conversation-model.git
cd Nvidia-natural-conversation-model

# 2. Create .env file with your token
echo "HF_TOKEN=hf_your_token_here" > .env

# 3. Launch
docker compose up
```

The server will be available at `https://localhost:8998`.

### Docker Compose Configuration

The included `docker-compose.yaml` handles:
- GPU reservation (1 NVIDIA GPU)
- Port mapping (8998)
- HF model cache volume (`.cache:/root/.cache`) to avoid re-downloading
- `NO_TORCH_COMPILE=1` for stable Docker builds
- `.env` file for secrets

### Custom Host Binding

To make the server accessible from outside the container on all interfaces:

```bash
docker run --gpus all -p 8998:8998 \
  -e HF_TOKEN=$HF_TOKEN \
  personaplex \
  /app/moshi/.venv/bin/python -m moshi.server --ssl /app/ssl --host 0.0.0.0
```

## RunPod Deployment

Step-by-step guide for deploying on [RunPod](https://www.runpod.io/):

1. **Create a GPU Pod**
   - Template: `RunPod PyTorch 2.x`
   - GPU: A100 80GB (recommended), A10G, or L4
   - Volume: 50 GB+ (for model weights cache)

2. **SSH into the pod**
   ```bash
   ssh root@<POD_IP> -p <PORT>
   ```

3. **Clone and install**
   ```bash
   apt update && apt install -y libopus-dev
   git clone https://github.com/justwybe/Nvidia-natural-conversation-model.git
   cd Nvidia-natural-conversation-model
   pip install moshi/.
   ```

4. **Set your token**
   ```bash
   export HF_TOKEN=<YOUR_TOKEN>
   ```

5. **Launch the server**
   ```bash
   SSL_DIR=$(mktemp -d)
   python -m moshi.server --ssl "$SSL_DIR" --host 0.0.0.0
   ```

   > **Important:** Use `--host 0.0.0.0` so the server binds to all interfaces, making it accessible via RunPod's proxy.

6. **Access via RunPod proxy**
   - Go to your pod's "Connect" tab
   - Use the HTTP proxy URL on port 8998

   Alternatively, use a Gradio tunnel (see [below](#gradio-tunnel)) for a public URL without configuring ports.

## AWS / GCP

### Recommended Instance Types

| Provider | Instance Type | GPU | VRAM | Notes |
|---|---|---|---|---|
| AWS | `g5.2xlarge` | A10G | 24 GB | Good balance of cost and performance |
| AWS | `p4d.24xlarge` | 8x A100 | 40/80 GB | Overkill but fast; use `CUDA_VISIBLE_DEVICES=0` |
| GCP | `a2-highgpu-1g` | A100 | 40 GB | Best single-GPU option |
| GCP | `g2-standard-8` | L4 | 24 GB | Budget option |

### Security Group / Firewall

Open port **8998** (TCP) for the PersonaPlex server:

**AWS Security Group:**
- Type: Custom TCP
- Port Range: 8998
- Source: Your IP or `0.0.0.0/0` (public)

**GCP Firewall Rule:**
```bash
gcloud compute firewall-rules create allow-personaplex \
  --allow tcp:8998 \
  --source-ranges 0.0.0.0/0 \
  --description "PersonaPlex server"
```

### Launch Steps

```bash
# Install deps
sudo apt update && sudo apt install -y libopus-dev
pip install moshi/.

# Set token
export HF_TOKEN=<YOUR_TOKEN>

# Launch (bind to all interfaces for external access)
SSL_DIR=$(mktemp -d)
python -m moshi.server --ssl "$SSL_DIR" --host 0.0.0.0
```

## Gradio Tunnel

For instant public access without configuring ports or firewalls, use the built-in Gradio tunnel:

```bash
SSL_DIR=$(mktemp -d)
python -m moshi.server --ssl "$SSL_DIR" --gradio-tunnel
```

This creates a public `*.gradio.live` URL printed in the console. Useful for:
- Quick demos without port forwarding
- Sharing with collaborators
- RunPod / cloud instances without public IPs

You can optionally set a persistent tunnel token:
```bash
python -m moshi.server --ssl "$SSL_DIR" --gradio-tunnel --gradio-tunnel-token "my-secret-token"
```

> **Note:** Gradio tunnels are rate-limited and intended for demos, not production use.

## Tested GPU Configurations

| GPU | VRAM | Mode | Performance | Notes |
|---|---|---|---|---|
| A100 80GB | 80 GB | Full | Real-time | Recommended for production |
| A100 40GB | 40 GB | Full | Real-time | Recommended for production |
| A10G | 24 GB | Full | Real-time | Good cost/performance ratio |
| L4 | 24 GB | Full | Real-time | Budget cloud option |
| RTX 4090 | 24 GB | Full | Real-time | Best consumer GPU |
| RTX 3080 | 10 GB | CPU Offload | Near real-time | Use `--cpu-offload` |
| T4 | 16 GB | Full | Real-time | Minimum for full GPU mode |
| CPU only | 32 GB+ RAM | CPU | ~10-30x slower | Offline inference only |

> **Minimum VRAM:** 16 GB for full GPU mode, 8 GB with `--cpu-offload`.

## Troubleshooting

### CUDA Out of Memory

**Symptoms:** `torch.cuda.OutOfMemoryError` or `CUDA error: out of memory`

**Solutions:**
1. Use `--cpu-offload` to offload layers to CPU RAM:
   ```bash
   pip install accelerate
   python -m moshi.server --ssl "$SSL_DIR" --cpu-offload
   ```
2. Set `CUDA_VISIBLE_DEVICES` to select a GPU with more memory:
   ```bash
   CUDA_VISIBLE_DEVICES=1 python -m moshi.server --ssl "$SSL_DIR"
   ```
3. Close other GPU processes: `nvidia-smi` to check, `kill <PID>` to free memory.

### SSL Certificate Errors

**Symptoms:** Browser shows "connection not secure" or refuses to connect.

**Solution:** This is expected with self-signed certificates. Click "Advanced" > "Proceed to localhost" in your browser. The `--ssl` flag generates temporary self-signed certs for development.

### Hugging Face Token Issues

**Symptoms:** `401 Unauthorized` or `Repository not found`

**Solutions:**
1. Ensure you've accepted the [model license](https://huggingface.co/nvidia/personaplex-7b-v1)
2. Verify your token: `echo $HF_TOKEN`
3. Try logging in directly: `huggingface-cli login`

### Blackwell GPU Compatibility

**Symptoms:** CUDA errors on RTX 5090, B100, or other Blackwell-architecture GPUs.

**Solution:** Install PyTorch with CUDA 13.0 support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

See [NVIDIA/personaplex#2](https://github.com/NVIDIA/personaplex/issues/2) for details.

### Docker: GPU Not Found

**Symptoms:** `RuntimeError: No CUDA GPUs are available` inside Docker.

**Solutions:**
1. Ensure [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed
2. Use `--gpus all` flag: `docker run --gpus all ...`
3. Verify GPU access inside container: `docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`

### Audio Issues

**Symptoms:** No audio output, garbled audio, or `libopus` errors.

**Solutions:**
1. Install Opus development library:
   ```bash
   # Ubuntu/Debian
   sudo apt install libopus-dev
   # macOS
   brew install opus
   ```
2. For the web client, ensure your browser allows microphone access
3. Check that input WAV files are valid: `ffprobe input.wav`

### Server Shows "Connection in Use"

**Symptoms:** Cannot connect a second client to the server.

**Explanation:** PersonaPlex uses an `asyncio.Lock` that allows only one active connection at a time. Disconnect the current client before connecting a new one, or restart the server.
