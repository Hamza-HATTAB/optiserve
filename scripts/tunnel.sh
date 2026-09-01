#!/usr/bin/env bash
set -euo pipefail

PORT=${1:-8001}
echo "=========================================================="
echo " OptiServe Cloudflare Tunnel Exposer for Live GPU Demos   "
echo " Target Port: http://localhost:$PORT                      "
echo "=========================================================="

if ! command -v cloudflared &>/dev/null; then
    echo "Notice: cloudflared is not found in PATH."
    echo "To install on Linux:"
    echo "  curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb"
    echo "  sudo dpkg -i cloudflared.deb"
    exit 1
fi

echo "Starting Cloudflare quick tunnel to local RTX 4060 daemon..."
cloudflared tunnel --url "http://localhost:$PORT"
