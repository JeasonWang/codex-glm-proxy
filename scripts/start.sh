#!/bin/bash
# Start Codex GLM Proxy

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="$PROJECT_DIR/codex-glm-proxy.log"
PID_FILE="$PROJECT_DIR/codex-glm-proxy.pid"

# Check if already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Proxy already running (PID: $PID)"
        exit 0
    fi
fi

# Check for API key
if [ -z "$GLM_API_KEY" ]; then
    echo "Error: GLM_API_KEY environment variable is not set"
    echo "Please run: export GLM_API_KEY='your_api_key'"
    exit 1
fi

# Check for aiohttp dependency
python3 -c "import aiohttp" 2>/dev/null || {
    echo "Installing aiohttp..."
    pip3 install -r "$PROJECT_DIR/requirements.txt" || {
        echo "Error: Failed to install aiohttp"
        exit 1
    }
}

# Start proxy
echo "Starting Codex GLM Proxy..."
nohup python3 "$PROJECT_DIR/proxy.py" > "$LOG_FILE" 2>&1 &
PID=$!
echo $PID > "$PID_FILE"
sleep 1

# Verify
if curl -s http://localhost:18765/health > /dev/null 2>&1; then
    echo "✓ Proxy started successfully (PID: $PID)"
    echo "  Health check: http://localhost:18765/health"
    echo "  Log file: $LOG_FILE"
else
    echo "✗ Proxy failed to start. Check log: $LOG_FILE"
    exit 1
fi
