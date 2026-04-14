#!/usr/bin/env bash
# Stop hook — Evaluate session for extractable patterns. Non-blocking.
# Writes a pointer to .claude/state/pattern-queue.jsonl; a later /learn-eval run
# (or continuous-learning skill) can process the queue into concrete skills.
set -euo pipefail

root="$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
state="$root/.claude/state"
mkdir -p "$state"
queue="$state/pattern-queue.jsonl"

ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
branch="$(git -C "$root" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"
# Lightweight signal: the pattern extractor only needs a reminder + branch context.
printf '{"ts":"%s","branch":"%s","status":"queued"}\n' "$ts" "$branch" >> "$queue"
exit 0
