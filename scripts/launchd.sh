#!/bin/bash
# Launchd wrapper: loads .env and starts proxy
DIR="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$DIR/.env" ] && export $(grep -v '^#' "$DIR/.env" | xargs)
exec /opt/homebrew/bin/python3 "$DIR/proxy.py"
