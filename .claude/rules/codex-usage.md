# Codex usage

`/codex:*` slash commands and the `codex-rescue` agent are **diff-scoped** tools. In plan mode or open-ended exploration they hang.

## Rules

- **Never** invoke `/codex:review`, `/codex:adversarial-review`, or the `codex-rescue` agent without a concrete diff in scope.
- Default target is the working-tree diff. Before invoking, verify it is non-trivial:
  - `git status --short --untracked-files=all`
  - `git diff --shortstat` (unstaged)
  - `git diff --shortstat --cached` (staged)
  - Or an explicit branch range: `git diff --shortstat base...HEAD`.
- If the diff is empty or the only changes are a handful of lines with no coherent theme, **don't run codex** — use `code-reviewer` or `python-reviewer` agents instead.
- Prefer background runs for anything beyond ~2 files. Foreground only for tiny diffs where you genuinely need the answer next.

## When to use what

| Situation | Tool |
|---|---|
| I just landed a cut and want diff-scoped review. | `/codex:review` (background) |
| I want adversarial pushback on a committed design. | `/codex:adversarial-review` (background) |
| I want free-form code review in plan mode. | `code-reviewer` or `python-reviewer` agent, not codex. |
| I want to brainstorm architecture. | `planner` or `Plan` agent, not codex. |
| I need a second implementation or a rescue on a stuck task. | `codex-rescue` agent with a **concrete task spec and file scope**. |

## Anti-patterns

- Running `/codex:review` before you have any diff. It will sit forever.
- Passing "review the whole repo" to codex-rescue. Scope it to a path set.
- Chaining codex reviews on overlapping commits. One review per cut.
