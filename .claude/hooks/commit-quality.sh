#!/usr/bin/env bash
# PreToolUse hook — Bash matcher.
# Before `git commit`: warn if lint wasn't run on staged files; surface secret patterns.
# Non-blocking (exit 0) — warnings only.
set -euo pipefail

input="$(cat 2>/dev/null || true)"
cmd="$(printf '%s' "$input" | /usr/bin/python3 -c 'import json,sys
try:
  d=json.loads(sys.stdin.read()); print(d.get("tool_input",{}).get("command",""))
except Exception:
  pass
' 2>/dev/null || true)"

[ -z "$cmd" ] && exit 0
printf '%s' "$cmd" | grep -Eq '\bgit\s+commit\b' || exit 0

# Repo root
root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[ -z "$root" ] && exit 0

# Staged files (respect pathspec)
staged="$(git -C "$root" diff --cached --name-only --diff-filter=ACMR 2>/dev/null || true)"
[ -z "$staged" ] && exit 0

# Warn about secrets in staged diff (lightweight)
diff="$(git -C "$root" diff --cached 2>/dev/null || true)"
if printf '%s' "$diff" | grep -Eiq '(aws_?secret|api[_-]?key|password\s*=\s*["\x27]|BEGIN RSA|BEGIN OPENSSH|ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})'; then
  >&2 echo "[commit-quality] ⚠ Possible secret detected in staged diff — review before committing."
fi

# Warn about forgotten print / pdb / breakpoint in staged .py changes
if printf '%s' "$diff" | grep -E '^\+' | grep -Eiq '(\bpdb\.set_trace\(|\bbreakpoint\(\s*\)|^\+[^#]*\bprint\()'; then
  >&2 echo "[commit-quality] ⚠ New print()/pdb/breakpoint in staged Python. Intentional?"
fi

# Remind to run make lint if any staged file is under lint coverage
if printf '%s' "$staged" | grep -Eq '^(openhands|evaluation|tests|scripts)/'; then
  >&2 echo "[commit-quality] Reminder: run 'make lint' (or 'make lint-scripts' for scripts/) before pushing."
fi

exit 0
