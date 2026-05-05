#!/bin/bash
# Hydra launcher for the FSDP TrainerAdapter — LiveStore consumer only.
#
# BC-15 contract: this script passes NO vLLM endpoints, NO ProRL address,
# and NO parquet paths (BC-14). The trainer's only external connections are:
#   - LiveStore      (gRPC UDS)  — training data via get_batch()
#   - PolicyRegistry (gRPC UDS)  — LoRA publish via publish_policy_version()
#
# All rollout generation belongs to RolloutManager + EnvironmentProvider.
# Start those services first via ops/services/start_all.sh.
set -euo pipefail

PROJECT_NAME='ProAgent'
EXPERIMENT_NAME='fullasync-replay-prorl'
SFT_MODEL_PATH="${SFT_MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
CKPT_PATH="${CKPT_PATH:-/workspace/outputs}"

BATCH_SIZE=${BATCH_SIZE:-4}
# NUM_TRAJ: siblings per group. Must match the group_size used by RolloutManager.
NUM_TRAJ=8
SAVE_FREQ=${SAVE_FREQ:-1}

USE_KL_LOSS=False
KL_LOSS_COEF=0.001
KL_LOSS_TYPE=low_var_kl
ENTROPY_COEFF=0
# DAPO asymmetric clip (papers/DAPO §3.2).
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28

# GPU_MEM_UTIL: local vLLM init param — the vLLM engine is put to sleep
# immediately (async_rollout_manager.sleep()), so this only affects startup
# memory reservation. Overridden by start.sh to 0.45 for A100 headroom.
GPU_MEM_UTIL=0.8
TP_SIZE=2
NNODES=1
# SP=4: Ulysses sequence parallelism for 8-GPU FSDP (DP=2).
SP_SIZE=4
TEMPERATURE=1.4
TOP_P=0.95

python3 -m verl_custom.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=$BATCH_SIZE \
    data.max_prompt_length=31232 \
    data.max_response_length=2048 \
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
    +actor_rollout_ref.actor.masking=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    actor_rollout_ref.rollout.n=$NUM_TRAJ \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    +actor_rollout_ref.rollout.publish_on_save=True \
    +actor_rollout_ref.rollout.max_starting_message_length=12000 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    +actor_rollout_ref.rollout.multi_turn.agent=True \
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
    trainer.test_freq=-1 \
    trainer.total_epochs=100 \
    +actor_rollout_ref.exchange_size=500000000 \
    "$@"
