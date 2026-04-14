#!/usr/bin/env bash
# PreToolUse hook — Bash matcher.
# Block git commands that bypass pre-commit / commit-msg / pre-push hooks.
# Reads tool input JSON from stdin, extracts the command, and exits 2 (block) if dangerous flags are found.
set -euo pipefail

input="$(cat 2>/dev/null || true)"
cmd="$(printf '%s' "$input" | /usr/bin/python3 -c 'import json,sys
try:
  d=json.loads(sys.stdin.read()); print(d.get("tool_input",{}).get("command",""))
except Exception:
  pass
' 2>/dev/null || true)"

[ -z "$cmd" ] && exit 0

# Match git-hook bypass flags when used with git commands
if printf '%s' "$cmd" | grep -Eq '\bgit\b.*(--no-verify|--no-gpg-sign)'; then
  >&2 echo "[block-no-verify] Refused: '$cmd'"
  >&2 echo "[block-no-verify] Reason: --no-verify / --no-gpg-sign bypass pre-commit, commit-msg, and pre-push hooks."
  >&2 echo "[block-no-verify] If a hook is failing, fix the underlying issue instead of skipping it."
  >&2 echo "[block-no-verify] Opt-out (discouraged): ECC_ALLOW_NO_VERIFY=1"
  if [ "${ECC_ALLOW_NO_VERIFY:-}" = "1" ]; then
    >&2 echo "[block-no-verify] ECC_ALLOW_NO_VERIFY=1 — allowing once."
    exit 0
  fi
  exit 2
fi
exit 0
