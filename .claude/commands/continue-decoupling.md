---
description: Deprecated — points to the active continuation command.
---

# Decoupling milestone is DONE

Stages 0 and 1 are complete. Stage 1 shipped in three cuts: Cut A (local pool smoke, `849314ff`), Cut B (local decoupled trainer, `53949b72`), and Cut C (remote HTTP pool — formerly labelled "Stage 1.5" during development).

**Use `/continue-weight-sync` for the next milestone** (Stage 2 — weight sync + replay buffer). That command has the current environment checks, the right doc-read order, a fresh-session status preamble, and the plan-mode kickoff protocol.

## Pointers

- Current status: [`plans-n-solutions/README.md`](../../plans-n-solutions/README.md)
- Stage 1 umbrella (Cuts A + B, local pool): [`plans-n-solutions/stages/stage1.md`](../../plans-n-solutions/stages/stage1.md)
- Stage 1 Cut C (remote pool) record + rollout stats: [`plans-n-solutions/stages/stage1_remote_pool.md`](../../plans-n-solutions/stages/stage1_remote_pool.md)
- Next-stage plan: [`plans-n-solutions/stages/stage2_weight_sync_and_replay.md`](../../plans-n-solutions/stages/stage2_weight_sync_and_replay.md)
- Next-session kickoff: [`continue-weight-sync.md`](./continue-weight-sync.md)

## Historical reference

The original Stage-1 playbook (phase sequencing, Codex gates, commit discipline) remains at [`plans-n-solutions/stages/stage1_playbook.md`](../../plans-n-solutions/stages/stage1_playbook.md). Treat it as historical — do not re-run it.
