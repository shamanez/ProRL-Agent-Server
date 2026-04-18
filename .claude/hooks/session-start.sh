#!/usr/bin/env bash
# SessionStart hook — 5-line orientation for Claude Code sessions.
# Prints current branch, uncommitted file count, critical unset env vars, and points at CLAUDE.md.
set -euo pipefail

root="$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
cd "$root"

branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '(no git)')"
uncommitted="$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ' || echo 0)"

echo "═══ Session Start ═══"
echo "Branch: $branch   Uncommitted: ${uncommitted} file(s)"

# Report unset environment variables commonly needed by this project (don't set them — just flag)
missing=()
for var in TEST_RUNTIME OH_RUNTIME_SINGULARITY_IMAGE_REPO; do
  if [ -z "${!var:-}" ]; then missing+=("$var"); fi
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "Unset env vars (set if running runtime tests): ${missing[*]}"
fi

echo "Docs: CLAUDE.md  |  Plans: plans-n-solutions/  |  Skills: .claude/skills/"
echo "════════════════════"
exit 0
