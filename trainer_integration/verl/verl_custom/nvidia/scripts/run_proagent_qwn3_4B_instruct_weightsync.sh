#!/bin/bash
# Phase 1 sibling of run_proagent_qwn3_4B_instruct_remote_decoupled.sh.
# Same decoupled topology (trainer FSDP local, vLLM pool on EC2, ProRL on
# host), but turns on rank-16 LoRA training and publishes the adapter to
# every pool child via POST /reload_lora after each _save_checkpoint. See
# plans-n-solutions/stages/weight_sync_lora.md.
#
# Deltas vs run_proagent_qwn3_4B_instruct_remote_decoupled.sh:
#   - EXPERIMENT_NAME points at the new WandB project slot.
#   - SAVE_FREQ=5 — publish cadence (design sketch §4).
#   - +actor_rollout_ref.model.lora_rank=16, lora_alpha=32, target_modules
#     covers all Qwen3 attention + MLP projections.
#   - +actor_rollout_ref.rollout.publish_on_save=True flips the ray_trainer
#     _publish_lora_adapter call on.
#
# `set -euo pipefail` so the trainer's exit code propagates to the docker
# launcher (s2_weightsync_docker.sh).
set -euo pipefail

PROJECT_NAME='ProAgent'
EXPERIMENT_NAME='weight-sync-decup-prorl'
DATA_PATH="/path/to/data/parquet"
SFT_MODEL_PATH='Qwen/Qwen3-4B-Instruct-2507'
TOKENIZER_PATH='Qwen/Qwen3-4B-Instruct-2507'
CKPT_PATH='/path/to/outputs'


BATCH_SIZE=4
MAX_NUM_ITERS=30
NUM_TRAJ=4
SAVE_FREQ=5
# See sibling `_remote_decoupled.sh` for the full rationale on worker count.
OPENHANDS_NUM_WORKERS=32

USE_KL_LOSS=True
KL_LOSS_COEF=0.001
KL_LOSS_TYPE=low_var_kl
ENTROPY_COEFF=0
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.2

# GPU_MEM_UTIL is unused for trainer-side vLLM (external pool owns vLLM memory)
# but kept to avoid Hydra removal noise.
GPU_MEM_UTIL=0.8
# rollout_dp_size = world_size / TP_SIZE = 8 / 2 = 4, matching the 4 remote
# endpoints. SP_SIZE=2 matches the Stage 0 baseline for 8-GPU FSDP.
TP_SIZE=2
NNODES=1
SP_SIZE=2
TEMPERATURE=1.4
TOP_P=0.95

python3 -m verl_custom.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=["$DATA_PATH/train.parquet"] \
    data.val_files=["$DATA_PATH/validation.parquet"] \
    data.train_batch_size=$BATCH_SIZE \
    +data.gen_batch_size=$BATCH_SIZE \
    data.max_prompt_length=31232 \
    data.max_response_length=1536 \
    data.truncation='error' \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
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
    actor_rollout_ref.actor.tis_imp_ratio_cap=2 \
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
    +actor_rollout_ref.rollout.external_llm_endpoints=[http://ec2-54-145-77-207.compute-1.amazonaws.com:8100,http://ec2-54-145-77-207.compute-1.amazonaws.com:8101,http://ec2-54-145-77-207.compute-1.amazonaws.com:8102,http://ec2-54-145-77-207.compute-1.amazonaws.com:8103] \
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
    +algorithm.filter_groups.enable=False \
    trainer.test_freq=-1 \
    +trainer.enable_pass_k_evaluation=True \
    +trainer.pass_k_problem_id_strategy=input_hash \
    trainer.total_epochs=100 $@
