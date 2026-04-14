#!/usr/bin/env bash
# PostToolUse hook — Edit matcher on *.py files.
# Warn (non-blocking) when print(), pdb.set_trace(), or breakpoint() is introduced in an edit.
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

# Inspect final file state (simpler than diffing the edit payload).
hits="$(grep -nE '(\bpdb\.set_trace\s*\(|\bbreakpoint\s*\(\s*\)|^\s*print\s*\()' "$path" 2>/dev/null || true)"
if [ -n "$hits" ]; then
  >&2 echo "[console-warn] $path contains debug statements:"
  printf '%s\n' "$hits" | sed 's/^/    /' >&2
  >&2 echo "[console-warn] Intentional? Use logging, not print(), for non-temporary output."
fi
exit 0
