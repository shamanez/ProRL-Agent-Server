#!/usr/bin/env bash
# PreCompact hook — Dump a structured summary before context compaction.
# The summary lives at .claude/state/last-precompact.md and is loaded on the next session-start.
set -euo pipefail

root="$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
state_dir="$root/.claude/state"
mkdir -p "$state_dir"
out="$state_dir/last-precompact.md"

branch="$(git -C "$root" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '(no git)')"
recent_files="$(git -C "$root" diff --name-only HEAD~5..HEAD 2>/dev/null | head -20 || true)"
uncommitted="$(git -C "$root" status --porcelain 2>/dev/null || true)"
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

{
  echo "# Pre-compact snapshot — $ts"
  echo ""
  echo "- Branch: \`$branch\`"
  echo ""
  echo "## Recently touched (last 5 commits)"
  if [ -n "$recent_files" ]; then
    echo ""
    printf '%s\n' "$recent_files" | sed 's/^/- /'
  else
    echo ""
    echo "_none_"
  fi
  echo ""
  echo "## Uncommitted"
  if [ -n "$uncommitted" ]; then
    echo ""
    echo '```'
    printf '%s\n' "$uncommitted"
    echo '```'
  else
    echo ""
    echo "_clean_"
  fi
  echo ""
  echo "_Apply the \`strategic-compact\` skill after compaction to decide what must be re-read._"
} > "$out"

>&2 echo "[pre-compact] Snapshot written to $out"
exit 0
