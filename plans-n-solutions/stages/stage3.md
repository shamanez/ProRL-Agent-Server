# Stage 3 - Iterative off-policy publish

**Status: NOT STARTED**

## Plan

After each `save_freq` step, publish HF-format actor weights and force the external vLLM pool to reload them. Mode B from the design doc - simplest weight sync that produces an improving policy.

### Files modified

- `trainer_integration/verl/verl_custom/trainer/ppo/ray_trainer.py` - new `_publish_policy_version()` in `_save_checkpoint`, fan-out `/reload_weights` to all endpoints
- `scripts/serving/vllm_launcher.py` - flesh out `/reload_weights`: kill child, respawn with new model, wait for health
- Launch script - `save_freq=5`, `+publish_on_save=True`

### GPU plan

Same as Stage 2: trainer 0-3, external vLLM 4-7.

### Success criteria

- 20 steps, >= 4 successful `/reload_weights` events
- Each publish causes a drop in `actor/rollout_kl` on the next step
- Rewards trend is not flat (unlike stale-weight Stage 2)
- No "No LLM server addresses added" errors during drain window

### Risks

- Reload time: ~20-60s per Qwen3-4B load on A100
- Partial reload failure leaves replicas on different policy versions
- HF save may not be enabled by default in verl (check Stage 0 checkpoint layout)
- ProRL drain race between `/clear_llm_server` and `/add_llm_server`

## Solution

*Not started yet.*
