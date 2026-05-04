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

import logging  # noqa: E402 — module logger declared after verl_custom imports
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

_logger = logging.getLogger(__name__)


class RayPPOTrainerDAPO(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def _make_continuous_producer(self):
        """DAPO producer: ``generate_sequences_dapo`` owns its dataloader."""
        from verl_custom.replay.continuous_producer import (  # noqa: PLC0415
            ContinuousRolloutProducer,
        )

        replay_cfg = self.config.replay
        return ContinuousRolloutProducer(
            rollout_manager=self.async_rollout_manager,
            # DAPO pulls prompts from its internal loader; no factory.
            generate_fn=self.async_rollout_manager.generate_sequences_dapo,
            store=self.trajectory_store,
            step_counter=self._step_counter,
            prompts_iter_factory=None,
            poll_interval_s=float(replay_cfg.get('poll_interval_s', 0.05)),
        )

    def _acquire_training_batch_dapo(
        self, metrics: dict, timing_raw: dict
    ) -> 'DataProto':  # noqa: F821 — forward ref, DataProto imported in ray_trainer
        """Produce one DAPO training batch.

        Classic: wake, call ``generate_sequences_dapo``, push+sample.
        Continuous: wait for the store, sample (producer already pushed).
        """
        from verl import DataProto  # noqa: PLC0415

        from verl_custom.replay.continuous_producer import (  # noqa: PLC0415
            wait_until_with_progress,
        )

        # S2 migration: use the store path whenever trajectory_store is set.
        # Previously gated on LIVE_STORE_SOCKET env var AND _producer — that
        # caused the classic in-process path to be taken when
        # CONTINUOUS_PRODUCER=False (external RolloutWorker), because
        # _producer=None and the env var was not forwarded into the container.
        # Fix: trajectory_store being non-None is sufficient; the socket path
        # is already resolved inside LiveStoreClient at construction time.
        _use_store_path = self.trajectory_store is not None
        if _use_store_path:
            if self._producer is not None:
                self._producer.check_background_error()
            n_groups = int(self.config.data.train_batch_size)
            no_progress_timeout_s = float(
                self.config.replay.get('no_progress_timeout_s', 1800.0)
            )
            with _timer('gen', timing_raw):
                if self._producer is not None:
                    # In-process producer path: poll until n_groups fresh
                    # groups are available (busy-poll at 10 ms — acceptable
                    # for the in-process store where gRPC overhead is zero).
                    filled = wait_until_with_progress(
                        lambda: self.trajectory_store.num_fresh_groups(
                            self.global_steps
                        )
                        >= n_groups,
                        self.trajectory_store.total_pushes,
                        no_progress_timeout=no_progress_timeout_s,
                    )
                    if not filled:
                        self._producer.check_background_error()
                        raise RuntimeError(
                            f'Replay store made no forward progress for '
                            f'{no_progress_timeout_s:.1f}s while waiting for '
                            f'{n_groups} fresh groups (current='
                            f'{self.trajectory_store.num_fresh_groups(self.global_steps)} '
                            f'fresh / {self.trajectory_store.num_groups()} total, '
                            f'total_pushes={self.trajectory_store.total_pushes()}). '
                            'Producer is wedged — check pool /health and producer logs.'
                        )
                # External worker path (CONTINUOUS_PRODUCER=False):
                # sample_mini_batch → LiveStoreClient.get_batch blocks
                # server-side until n_groups are ready; NoProgressError is
                # raised by the server after no_progress_timeout_s. No
                # busy-poll needed here — avoid hammering the gRPC socket.
            metrics.update(
                self.trajectory_store.metrics(self.global_steps, suffix='_pre_sample')
            )
            sampled = self.trajectory_store.sample_mini_batch(
                n_groups=n_groups, current_step=self.global_steps
            )
            metrics.update(self.trajectory_store.metrics(self.global_steps))
            metrics.update(
                self.trajectory_store.metrics(self.global_steps, suffix='_post_sample')
            )
            return DataProto.from_dict(
                tensors=sampled.tensors,
                non_tensors=sampled.non_tensors,
                meta_info=sampled.meta_info,
            )

        # Classic path.
        with _timer('gen', timing_raw):
            assert self.async_rollout_mode
            self.async_rollout_manager.wake_up()
            batch = self.async_rollout_manager.generate_sequences_dapo()
            self.async_rollout_manager.sleep()
            timing_raw.update(batch.meta_info['timing'])
            batch.meta_info.pop('timing', None)
        return self._push_and_sample_replay(batch, metrics)

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
        if self.global_steps > 0:
            # Phase 1 bug fix #18: DAPO path was missing the resume-time
            # policy_version sync. The vLLM pool retains its active PV across
            # a trainer restart; without this align, the first post-resume
            # ``/reload_lora`` is rejected as non-monotonic and weight-sync
            # stalls silently. Mirrors ``ray_trainer.py`` fit-time block.
            self.policy_version = self.global_steps
            self.async_rollout_manager.policy_version = self.global_steps

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

        # Cut 4: spin up the DAPO continuous rollout producer. No-op
        # unless ``replay.continuous_producer=True``. Wrapped in
        # try/finally so the daemon thread stops + pool sleeps even on
        # exceptions raised from the training loop.
        self._start_continuous_producer_if_needed()

        try:
            while True:
                metrics = {}

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer('step', timing_raw):
                    # Cut 4: gen + push + sample encapsulated. Classic mode
                    # calls ``generate_sequences_dapo`` inline; continuous
                    # mode waits for the store (producer pushes) and samples.
                    batch = self._acquire_training_batch_dapo(metrics, timing_raw)

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
                                {
                                    k: np.array(v)
                                    for k, v in reward_extra_infos_dict.items()
                                }
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
                        loss_agg_mode = (
                            self.config.actor_rollout_ref.actor.loss_agg_mode
                        )
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
                                ref_log_prob = (
                                    self.actor_rollout_wg.compute_ref_log_prob(batch)
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
                        # Producer + validation share one OpenHands session;
                        # validation's /stop at teardown kills the producer's
                        # in-flight /process. Pause producer around _validate
                        # and resume afterwards. Buffer stays warm across the
                        # pause; K-staleness evicts naturally at next sample.
                        # Gotcha §19: if the producer is mid-asyncio.run the
                        # stop() timeout fires without the thread exiting —
                        # skip validate to avoid concurrent OH dispatch and
                        # let the producer finish its call; retry on the next
                        # save boundary.
                        if self._stop_continuous_producer_if_needed():
                            try:
                                with _timer('testing', timing_raw):
                                    val_metrics: dict = self._validate()
                                    if is_last_step:
                                        last_val_metrics = val_metrics
                                metrics.update(val_metrics)
                            finally:
                                if not is_last_step:
                                    self._start_continuous_producer_if_needed()
                        else:
                            _logger.warning(
                                'step=%d skipping _validate: producer stop '
                                'timed out (still mid-generate_sequences); '
                                'will retry on next save boundary',
                                self.global_steps,
                            )

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
                    # Cut 4: propagate the new step into the producer so
                    # records pushed from now on carry the fresh tag.
                    if self._step_counter is not None:
                        self._step_counter.set(self.global_steps)
                    if self._producer is not None:
                        self._producer.check_background_error()
                    if is_last_step:
                        pprint(f'Final validation metrics: {last_val_metrics}')
                        progress_bar.close()
                        return
        finally:
            self._stop_continuous_producer_if_needed()
