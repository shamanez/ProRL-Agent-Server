#!/usr/bin/env bash
# Stop hook — Emit lightweight cost/telemetry markers.
# Appends one line to .claude/state/costs.jsonl per session stop.
# Non-blocking; pure local log for the /harness-audit command to aggregate later.
set -euo pipefail

root="$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
state="$root/.claude/state"
mkdir -p "$state"
out="$state/costs.jsonl"

ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
branch="$(git -C "$root" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"
# Detect environment signals that affect cost (model, max-thinking, etc.) from common env vars.
model="${CLAUDE_MODEL:-${ANTHROPIC_MODEL:-unknown}}"
profile="${ECC_HOOK_PROFILE:-standard}"

printf '{"ts":"%s","branch":"%s","model":"%s","profile":"%s"}\n' \
  "$ts" "$branch" "$model" "$profile" >> "$out"
exit 0
