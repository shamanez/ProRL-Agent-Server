#!/usr/bin/env bash
# PreToolUse hook — Edit|Write|MultiEdit matcher.
# Block edits to linter/formatter/type-checker configs. These are project guardrails;
# modifying them to "make the linter happy" is a common anti-pattern.
# Set ECC_ALLOW_CONFIG_EDIT=1 to bypass once when you genuinely need to update a config.
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

# Guarded paths (match by suffix so absolute-path prefixes don't matter)
case "$path" in
  */dev_config/python/ruff.toml|\
  */dev_config/python/mypy.ini|\
  */dev_config/python/.pre-commit-config.yaml|\
  */.ruff.toml|\
  */ruff.toml|\
  */mypy.ini|\
  */.pre-commit-config.yaml|\
  */.editorconfig)
    if [ "${ECC_ALLOW_CONFIG_EDIT:-}" = "1" ]; then
      >&2 echo "[config-protection] ECC_ALLOW_CONFIG_EDIT=1 — allowing edit to '$path'."
      exit 0
    fi
    >&2 echo "[config-protection] Refused: '$path' is a project-wide lint/type/format config."
    >&2 echo "[config-protection] Fix the code to match the config, don't change the config to match the code."
    >&2 echo "[config-protection] If this update is genuinely intentional, run again with ECC_ALLOW_CONFIG_EDIT=1."
    exit 2
    ;;
esac
exit 0
