#!/bin/bash
# Phase 2 Hydra launcher: fully-async decoupled topology (trainer FSDP
# local, vLLM pool on EC2, ProRL on host) with rank-16 LoRA training,
# bounded replay buffer, continuous rollout producer, and clipped
# temporal importance-sampling correction.
#
# Sibling of run_proagent_qwn3_4B_instruct_weightsync.sh. Differences:
#
#   - EXPERIMENT_NAME=fullasync-replay-prorl (distinct WandB run slot).
#   - No +algorithm.filter_groups.enable hardcoded here — the outer
#     s3_fullasync_docker.sh owns that knob so it can default to False
#     (plain GRPO) and be flipped to True only for the DAPO gate.
#     See plans-n-solutions/stages/full_async.md §5a.
#   - No +replay.* keys hardcoded — outer docker script owns them.
#   - tis_imp_ratio_cap stays at 2 (mechanism shared with Phase 1); the
#     Cut 3 gate is +replay.use_temporal_is, which is set by the docker
#     script.
#
# `set -euo pipefail` so the trainer's exit code propagates to the
# docker launcher (s3_fullasync_docker.sh).
set -euo pipefail

PROJECT_NAME='ProAgent'
EXPERIMENT_NAME='fullasync-replay-prorl'
DATA_PATH="/path/to/data/parquet"
SFT_MODEL_PATH='Qwen/Qwen3-4B-Instruct-2507'
TOKENIZER_PATH='Qwen/Qwen3-4B-Instruct-2507'
CKPT_PATH='/path/to/outputs'


BATCH_SIZE=${BATCH_SIZE:-4}
# Cut 6 — producer batch (DAPO ``requested_batch_size``) is decoupled
# from trainer batch. Default is 4× ``BATCH_SIZE`` so post-warmup the
# replay buffer always carries enough fresh groups for the trainer to
# draw without blocking. Override at run time via
# ``GEN_BATCH_SIZE=...`` in the outer launcher.
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-$((BATCH_SIZE * 4))}
MAX_NUM_ITERS=30
# DAPO-aligned: 8 samples per prompt balances GRPO group-stat signal
# (meaningful with filter_groups on) against rollout cost.
NUM_TRAJ=8
SAVE_FREQ=${SAVE_FREQ:-1}
# 32 OpenHands workers is the sweet spot for a 4-child vLLM pool on
# 4× H100. Empirically the pool saturates to ~100 % GPU util at ~32
# concurrent clients; bumping to 64 made every client-turn slower
# (queue depth on a pool-bound workload, not a host-bound one) and
# regressed first-step wall-clock from 17 min → 47 min (filter=True,
# Progress 3/4 → 0/4). See gotcha §17 in handsoff.md.
OPENHANDS_NUM_WORKERS=32

# DAPO drops KL loss: RLVR rewards are verifiable, no reward-model drift to
# anchor against. Coef/type kept as unused sentinels for readability.
USE_KL_LOSS=False
KL_LOSS_COEF=0.001
KL_LOSS_TYPE=low_var_kl
ENTROPY_COEFF=0
# DAPO "clip-higher": asymmetric clip preserves low-probability token
# exploration (papers/DAPO §3.2). Low bound stays at 0.2.
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28

# GPU_MEM_UTIL is unused for trainer-side vLLM (external pool owns vLLM memory)
# but kept to avoid Hydra removal noise.
GPU_MEM_UTIL=0.8
# rollout_dp_size = world_size / TP_SIZE = 8 / 2 = 4, matching the 4 remote
# endpoints. SP_SIZE=4 for 8-GPU FSDP (DP=2). Bumped from 2 → 4 after the
# Cut-1 max_response_length=16384 raise OOM'd backward at SP=2 (40.36 GiB
# requested on 39.49 GiB A100). Each sequence is now split 4-way along the
# token dim (Ulysses), roughly halving per-GPU activation memory. SP=4
# divides Qwen3-4B's 32 query heads / 8 KV heads / world_size=8 cleanly.
# If 4 still OOMs, escalate to 8 (DP=1, slowest but guaranteed fit).
TP_SIZE=2
NNODES=1
SP_SIZE=4
TEMPERATURE=1.4
TOP_P=0.95

# Sequence length contract (two-knob — see also async_server.py near
# self.total_len init, and the TrajectoryStore constructor in
# ray_trainer.py):
#   data.max_prompt_length=31232:       dataset filter + addend in total_len.
#                                       Never reaches vLLM directly.
#   data.max_response_length=2048:      per-turn vLLM max_output_tokens AND
#                                       addend in total_len. Tightened from
#                                       16384 → 2048: long-tail turns were
#                                       blowing past the publish-boundary
#                                       drain budget; multi-turn budget is
#                                       still bounded by total_len.
#   total_len = 33280:                  plumbed to vLLM as max_model_len; the
#                                       real per-call ceiling. vLLM enforces
#                                       seed + body <= max_model_len.
#   max_starting_message_length=12000:  empirical SWE-Gym (system + task)
#                                       seed cap; rollout-side prompt-slot
#                                       width. Independent knob.
# Replay-store caps mirror this contract: prompt cap =
# max_starting_message_length, response cap = total_len. Wiring at
# ray_trainer.py near TrajectoryStore(...).
python3 -m verl_custom.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=["$DATA_PATH/train.parquet"] \
    data.val_files=["$DATA_PATH/validation.parquet"] \
    data.train_batch_size=$BATCH_SIZE \
    +data.gen_batch_size=$GEN_BATCH_SIZE \
    data.max_prompt_length=31232 \
    data.max_response_length=2048 \
    data.truncation='error' \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    +actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj] \
    +actor_rollout_ref.model.exclude_modules=null \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP_SIZE \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
    actor_rollout_ref.actor.tis_imp_ratio_cap=5 \
    +actor_rollout_ref.actor.use_error_mask=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    +actor_rollout_ref.rollout.logprobs_mode=processed_logprobs \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    actor_rollout_ref.rollout.n=$NUM_TRAJ \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    +actor_rollout_ref.rollout.external_llm_endpoints=[http://${REMOTE_DNS:-ec2-54-145-77-207.compute-1.amazonaws.com}:8100,http://${REMOTE_DNS:-ec2-54-145-77-207.compute-1.amazonaws.com}:8101,http://${REMOTE_DNS:-ec2-54-145-77-207.compute-1.amazonaws.com}:8102,http://${REMOTE_DNS:-ec2-54-145-77-207.compute-1.amazonaws.com}:8103] \
    +actor_rollout_ref.rollout.publish_on_save=True \
    +actor_rollout_ref.rollout.async_manager=openhands \
    +actor_rollout_ref.rollout.max_iterations=$MAX_NUM_ITERS \
    +actor_rollout_ref.rollout.enable_memory_saver=True \
    +actor_rollout_ref.rollout.max_starting_message_length=12000 \
    +actor_rollout_ref.rollout.remove_think_tokens=True \
    +actor_rollout_ref.rollout.openhands_base_url=http://localhost:8006 \
    +actor_rollout_ref.rollout.openhands_num_workers=$OPENHANDS_NUM_WORKERS \
    +actor_rollout_ref.rollout.task_type=swegym \
    +actor_rollout_ref.rollout.chat_template_name=qwen3_chat_template_generation \
    +actor_rollout_ref.rollout.openhands_timeout=1000 \
    +actor_rollout_ref.actor.masking=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    +actor_rollout_ref.rollout.multi_turn.agent=True \
    +actor_rollout_ref.rollout.token_level_generation=True \
    +actor_rollout_ref.rollout.custom_tokenizer=$TOKENIZER_PATH \
    +actor_rollout_ref.rollout.rollout_save_dir=$CKPT_PATH/rollout_data \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    +actor_rollout_ref.rollout.debug=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_manager.type="swebench" \
    +reward_manager.verifier.reward_coef=1.0 \
    +reward_manager.debug=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.resume_mode=auto \
    trainer.max_actor_ckpt_to_keep=10 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NNODES \
    trainer.save_freq=$SAVE_FREQ \
    trainer.val_before_train=False \
    +data.dataloader_num_workers=1 \
    +actor_rollout_ref.exchange_size=500000000 \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    trainer.test_freq=-1 \
    +trainer.enable_pass_k_evaluation=True \
    +trainer.pass_k_problem_id_strategy=input_hash \
    trainer.total_epochs=100 "$@"
