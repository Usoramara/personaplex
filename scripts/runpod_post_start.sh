#!/bin/bash
# Auto-start PersonaPlex on RunPod pod boot.
# Called by RunPod's /start.sh as /post_start.sh.
# Models and repo live on the persistent volume at /runpod.

LOG="/var/log/personaplex.log"

echo "[PersonaPlex] post_start.sh triggered at $(date)" | tee "$LOG"

# Clone repo to volume if not present
if [[ ! -d /runpod/personaplex/moshi ]]; then
    echo "[PersonaPlex] Cloning repo to /runpod/personaplex..." | tee -a "$LOG"
    git clone https://github.com/Usoramara/personaplex.git /runpod/personaplex >> "$LOG" 2>&1
fi

# Install deps + launch server (idempotent, uses cached models from volume)
bash /runpod/personaplex/scripts/runpod_setup.sh --gradio-tunnel >> "$LOG" 2>&1 &

echo "[PersonaPlex] Server launching in background. Logs: tail -f $LOG"
