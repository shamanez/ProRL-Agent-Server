#!/usr/bin/env bash
# PostToolUse hook — Edit|Write matcher on *.py files.
# Auto-run ruff check --fix and ruff format on the touched file. Fast (<200ms).
# Non-blocking; failures only print a warning.
set -euo pipefail

input="$(cat 2>/dev/null || true)"
path="$(printf '%s' "$input" | /usr/bin/python3 -c 'import json,sys
try:
  d=json.loads(sys.stdin.read())
  ti=d.get("tool_input",{})
  print(ti.get("file_path") or ti.get("path") or "")
except Exception:
  pass
' 2>/dev/null || true)"

[ -z "$path" ] && exit 0
case "$path" in *.py) ;; *) exit 0 ;; esac
[ ! -f "$path" ] && exit 0

# Locate ruff config (prefer dev_config/python/ruff.toml at repo root)
root="$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
cfg="$root/dev_config/python/ruff.toml"
cfg_arg=()
[ -f "$cfg" ] && cfg_arg=(--config "$cfg")

# Prefer poetry ruff if available; fall back to system ruff
if command -v poetry >/dev/null 2>&1 && [ -f "$root/pyproject.toml" ]; then
  ruff="poetry run ruff"
else
  ruff="ruff"
fi

# Run check --fix, then format. Silence success; surface only failures.
if ! $ruff check "${cfg_arg[@]}" --fix --exit-zero "$path" >/tmp/.ruff-check.log 2>&1; then
  >&2 echo "[ruff-post-edit] ruff check failed on $path (see /tmp/.ruff-check.log)"
fi
if ! $ruff format "${cfg_arg[@]}" "$path" >/tmp/.ruff-format.log 2>&1; then
  >&2 echo "[ruff-post-edit] ruff format failed on $path (see /tmp/.ruff-format.log)"
fi
exit 0
