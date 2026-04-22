# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

from collections import defaultdict
from pprint import pprint

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.profiler.performance import simple_timer as _timer

from verl_custom.trainer.ppo.core_algos import agg_loss
from verl_custom.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_timing_metrics,
)
from verl_custom.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)


class RayPPOTrainerDAPO(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        # to resolve a bug in vllm
        if self.config.actor_rollout_ref.rollout.mode == 'async':
            self.async_rollout_manager.sleep()
        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get(
            'val_before_train', True
        ):
            val_metrics = self._validate()
            assert val_metrics, f'{val_metrics=}'
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # add tqdm
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc='Training Progress',
        )

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None

        while True:
            metrics = {}

            is_last_step = self.global_steps >= self.total_training_steps

            with _timer('step', timing_raw):
                # generate a batch
                with _timer('gen', timing_raw):
                    assert self.async_rollout_mode
                    self.async_rollout_manager.wake_up()
                    batch = self.async_rollout_manager.generate_sequences_dapo()
                    self.async_rollout_manager.sleep()

                    timing_raw.update(batch.meta_info['timing'])
                    batch.meta_info.pop('timing', None)

                with _timer('reward', timing_raw):
                    # compute scores. Support both model and function-based.
                    if self.use_rm:
                        # we first compute reward model score
                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                        batch = batch.union(reward_tensor)

                    # rule-based rm
                    reward_result = ray.get(self.reward_fn.__call__.remote(batch))
                    reward_extra_infos_dict = {}

                    batch.batch['token_level_scores'] = reward_result['score']
                    batch.batch['token_level_rewards'] = reward_result['reward']

                    if reward_extra_infos_dict:
                        batch.non_tensor_batch.update(
                            {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                        )

                    # Log reward_metrics if available
                    if 'reward_metrics' in reward_result:
                        reward_metrics = reward_result['reward_metrics']
                        if isinstance(reward_metrics, dict):
                            for key, value in reward_metrics.items():
                                metrics[f'reward_metrics/{key}'] = value
                        else:
                            print(
                                f'Warning: reward_metrics is not a dict, got {type(reward_metrics)}'
                            )

                    # compute rewards. apply_kl_penalty if available
                    if self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(
                            batch,
                            kl_ctrl=self.kl_ctrl_in_reward,
                            kl_penalty=self.config.algorithm.kl_penalty,
                        )
                        metrics.update(
                            kl_metrics
                        )  # TODO: This will be cleared if we use multiple genenration batches

                assert self.config.algorithm.filter_groups.enable

                # === Updating ===

                batch.batch['response_mask'] = compute_response_mask(batch)

                # Balance the number of valid tokens across DP ranks.
                # NOTE: This usually changes the order of data in the `batch`,
                # which won't affect the advantage calculation (since it's based on uid),
                # but might affect the loss calculation (due to the change of mini-batching).
                # TODO: Decouple the DP balancing and mini-batching.
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)

                # compute global_valid tokens
                batch.meta_info['global_token_num'] = torch.sum(
                    batch.batch['attention_mask'], dim=-1
                ).tolist()

                # recompute old_log_probs
                with _timer('old_log_prob', timing_raw):
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    entropys = old_log_prob.batch['entropys']
                    response_masks = batch.batch['response_mask']
                    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                    entropy_loss = agg_loss(
                        loss_mat=entropys,
                        loss_mask=response_masks,
                        loss_agg_mode=loss_agg_mode,
                    )
                    old_log_prob_metrics = {
                        'actor/entropy_loss': entropy_loss.detach().item()
                    }
                    metrics.update(old_log_prob_metrics)
                    old_log_prob.batch.pop('entropys')
                    batch = batch.union(old_log_prob)

                    if 'rollout_log_probs' in batch.batch.keys():
                        # TODO: we may want to add diff of probs too.
                        rollout_old_log_probs = batch.batch['rollout_log_probs']
                        actor_old_log_probs = batch.batch['old_log_probs']
                        attention_mask = batch.batch['attention_mask']
                        responses = batch.batch['responses']
                        response_length = responses.size(1)
                        response_mask = attention_mask[:, -response_length:]

                        rollout_probs = torch.exp(rollout_old_log_probs)
                        actor_probs = torch.exp(actor_old_log_probs)
                        rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                        rollout_probs_diff = torch.masked_select(
                            rollout_probs_diff, response_mask.bool()
                        )
                        rollout_probs_diff_max = torch.max(rollout_probs_diff)
                        rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                        rollout_probs_diff_std = torch.std(rollout_probs_diff)
                        metrics.update(
                            {
                                'training/rollout_probs_diff_max': rollout_probs_diff_max.detach().item(),
                                'training/rollout_probs_diff_mean': rollout_probs_diff_mean.detach().item(),
                                'training/rollout_probs_diff_std': rollout_probs_diff_std.detach().item(),
                            }
                        )

                if self.use_reference_policy:
                    # compute reference log_prob
                    with _timer('ref', timing_raw):
                        if not self.ref_in_actor:
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                batch
                            )
                        else:
                            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(
                                batch
                            )
                        batch = batch.union(ref_log_prob)

                # compute values
                if self.use_critic:
                    with _timer('values', timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with _timer('adv', timing_raw):
                    # compute advantages, executed on the driver process

                    norm_adv_by_std_in_grpo = self.config.algorithm.get(
                        'norm_adv_by_std_in_grpo', True
                    )  # GRPO adv normalization factor

                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                        config=self.config.algorithm,
                    )

                # update critic
                if self.use_critic:
                    with _timer('update_critic', timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(
                        critic_output.meta_info['metrics']
                    )
                    metrics.update(critic_output_metrics)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with _timer('update_actor', timing_raw):
                        batch.meta_info['multi_turn'] = (
                            self.config.actor_rollout_ref.rollout.multi_turn.enable
                        )
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(
                        actor_output.meta_info['metrics']
                    )
                    metrics.update(actor_output_metrics)

                # Log rollout generations if enabled
                rollout_data_dir = self.config.trainer.get('rollout_data_dir', None)
                if rollout_data_dir:
                    with _timer('dump_rollout_generations', timing_raw):
                        print(batch.batch.keys())
                        inputs = self.tokenizer.batch_decode(
                            batch.batch['prompts'], skip_special_tokens=True
                        )
                        outputs = self.tokenizer.batch_decode(
                            batch.batch['responses'], skip_special_tokens=True
                        )
                        scores = (
                            batch.batch['token_level_scores'].sum(-1).cpu().tolist()
                        )
                        self._dump_generations(
                            inputs=inputs,
                            outputs=outputs,
                            scores=scores,
                            reward_extra_infos_dict=reward_extra_infos_dict,
                            dump_path=rollout_data_dir,
                        )

                did_save = False
                if self.config.trainer.save_freq > 0 and (
                    is_last_step
                    or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with _timer('save_checkpoint', timing_raw):
                        self._save_checkpoint()
                    did_save = True
                elif self.timeout.check_save():
                    with _timer('save_checkpoint', timing_raw):
                        self._save_checkpoint()
                    did_save = True

                # Mirror RayPPOTrainer.fit publish hook: DAPO's filter_groups
                # path swaps the trainer class, so this block must exist here
                # too or the decoupled LoRA pool never sees a weight update.
                if (
                    did_save
                    and self.config.actor_rollout_ref.rollout.get(
                        'publish_on_save', False
                    )
                    and self.config.actor_rollout_ref.model.get('lora_rank', 0) > 0
                ):
                    import os  # noqa: PLC0415 — autoflake strips top-level if unused at parse time

                    local_global_step_folder = os.path.join(
                        self.config.trainer.default_local_dir,
                        f'global_step_{self.global_steps}',
                    )
                    with _timer('publish_lora', timing_raw):
                        self._publish_lora_adapter(local_global_step_folder)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (
                        is_last_step
                        or self.global_steps % self.config.trainer.test_freq == 0
                        or self.timeout.last_saved
                    )
                ):
                    with _timer('testing', timing_raw):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # training metrics
                metrics.update(
                    {
                        'training/global_step': self.global_steps,
                    }
                )
                # collect metrics
                metrics.update(
                    compute_data_metrics(batch=batch, use_critic=self.use_critic)
                )
                metrics.update(
                    compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                )

                # LoRA weight-sync metrics: publish keys surface only on steps
                # that actually published; staleness is every step (steps
                # since the last successful publish — not policy_version,
                # which is a cardinal not an ordinal).
                if self._last_publish_metrics:
                    metrics.update(self._last_publish_metrics)
                    self._last_publish_metrics = {}
                metrics['rollout/staleness_steps'] = max(
                    0, self.global_steps - self._last_publish_step
                )

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    progress_bar.close()
                    return
