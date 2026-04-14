#!/usr/bin/env bash
# Stop hook — Persist a lightweight session summary when ECC_SESSION_PERSIST=1.
# Writes to .claude/state/sessions/<yyyy-mm-dd>-<unix>.json
set -euo pipefail

[ "${ECC_SESSION_PERSIST:-}" = "1" ] || exit 0

input="$(cat 2>/dev/null || true)"
root="$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
dir="$root/.claude/state/sessions"
mkdir -p "$dir"

ts_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
ts_unix="$(date +%s)"
date_d="$(date -u +%Y-%m-%d)"
branch="$(git -C "$root" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"
changed="$(git -C "$root" status --porcelain 2>/dev/null | wc -l | tr -d ' ' || echo 0)"

out="$dir/$date_d-$ts_unix.json"
printf '{\n  "ts": "%s",\n  "branch": "%s",\n  "uncommitted": %s,\n  "raw_len": %s\n}\n' \
  "$ts_iso" "$branch" "$changed" "${#input}" > "$out"
>&2 echo "[stop-session-summary] $out"
exit 0
