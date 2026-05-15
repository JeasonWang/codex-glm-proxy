#!/bin/bash
# Stop Codex GLM Proxy

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$PROJECT_DIR/codex-glm-proxy.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Stopping proxy (PID: $PID)..."
        kill "$PID"
        rm -f "$PID_FILE"
        echo "✓ Proxy stopped"
    else
        echo "Proxy process not running (stale PID file)"
        rm -f "$PID_FILE"
    fi
else
    # Try to find and kill by port
    PID=$(lsof -ti:18765 2>/dev/null | head -1)
    if [ -n "$PID" ]; then
        echo "Stopping proxy (PID: $PID)..."
        kill "$PID"
        echo "✓ Proxy stopped"
    else
        echo "Proxy is not running"
    fi
fi
