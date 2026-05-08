# Slime vs prime-rl: Deep Technical Analysis

## Executive Summary

The most important finding is that this repository does not yet contain a working Slime trainer adapter. `trainers/slime/scripts/start.sh` is a stub that exits, and the Slime integration exists as a documented contract in `docs/PLUGGING_IN_NEW_TRAINER_OR_ENVIRONMENT.md`. Therefore, the comparison below treats "our Slime system" as the Slime-facing RolloutFabric design: Slime is expected to consume `LiveStore` through a custom rollout function and publish checkpoints through `PolicyRegistry`.

RolloutFabric's strongest idea is its service boundary. It separates dataset ownership, environment execution, hot rollout storage, trainer execution, policy publication, and inference reload into replaceable services. This is cleaner for trainer/environment replacement than prime-rl's vertically integrated design.

prime-rl is stronger as a production training framework. It has a richer asynchronous rollout scheduler, typed Pydantic/TOML config system, broad observability, deployment support, integrated `verifiers` environments, optimized trainer/model code, NCCL/filesystem weight broadcast, and broad tests.

The recommended direction is not to copy prime-rl wholesale. Keep RolloutFabric's contract-first boundary conditions, especially token-ID preservation, per-group policy snapshots, trainer isolation, and hard abort on partial policy fanout. Adapt prime-rl's scheduler discipline, config validation, environment wrapper patterns, metrics/artifact taxonomy, deployment validation, and selected transport fields.

## Our System: Architecture and Big Idea

### Facts

This repository is a six-service RL training fabric for agentic SWE-Bench/OpenHands rollouts. The core data and policy flow is documented in `CLAUDE.md` and `README.md`:

1. `RolloutManager` owns the dataset, reads parquet rows, captures one policy snapshot per group, calls the environment provider, builds `TrainingSample` records, archives rollouts, and pushes accepted groups to `LiveStore`.
2. `EnvironmentProvider` executes task episodes through OpenHands/ProRL over HTTP and uses vLLM for token generation.
3. `InferenceBackend` serves generation and reloads LoRA/policy adapters.
4. `LiveStore` is a bounded hot buffer between rollout production and trainer consumption.
5. `TrainerAdapter` consumes `LiveStore`, trains locally, saves adapters/checkpoints, and publishes policy versions.
6. `PolicyRegistry` fans out adapter reloads to the inference pool and becomes the source of truth for the latest policy version.

Relevant source paths:

- Architecture and invariants: `CLAUDE.md`, `README.md`, `plans-n-solutions/rollout_fabric.md`
- Rollout loop: `core/rollout_fabric/rollout_manager/loop.py`
- Dataset loader: `core/rollout_fabric/rollout_manager/dataloader.py`
- Environment HTTP client: `core/rollout_fabric/rollout_manager/prorl_client.py`
- Episode/sample conversion: `core/rollout_fabric/rollout_manager/episode_builder.py`
- Hot store: `core/rollout_fabric/live_store/store_core.py`, `server.py`, `client.py`, `codec.py`
- Policy publication: `core/rollout_fabric/policy_registry/server.py`, `client.py`, `fanout.py`
- Protocols and wire schemas: `core/rollout_fabric/schemas/protocols/`, `core/rollout_fabric/schemas/proto/`
- Current VERL adapter pattern: `trainers/verl/verl_custom/fabric_adapter/live_store_batch.py`, `trainers/verl/verl_custom/fabric_adapter/pad.py`

The big idea is trainer and environment pluggability through typed service contracts. The trainer should not become the hidden orchestrator. In particular, `BC-14` says `RolloutManager` owns the parquet dataloader, and `BC-15` says the trainer connects only to `LiveStore` and `PolicyRegistry`.

### Slime Status

There is no implemented Slime adapter in this repo. `trainers/slime/scripts/start.sh` is a failing placeholder. The actionable Slime design is in `docs/PLUGGING_IN_NEW_TRAINER_OR_ENVIRONMENT.md`, which says to integrate through Slime's custom rollout function path:

- Replace Slime's inline rollout generation with a function that calls `LiveStoreClient.get_batch()`.
- Map fabric `TrainingSample` records into Slime `Sample` keys such as `input_ids`, `output_ids`, `rewards`, and `loss_mask`.
- Hook policy publishing after Slime saves a model via `PolicyRegistryClient.publish_policy_version()`.

So any claim that "Slime is integrated" would be inaccurate. The correct claim is: RolloutFabric has a Slime-compatible boundary design, but the Slime adapter remains follow-up implementation work.

## Boundary Conditions in Slime

### Where They Are Defined

Boundary conditions are defined primarily in `plans-n-solutions/rollout_fabric.md`, summarized in `CLAUDE.md` and `README.md`, and partially enforced through tests under `tests/invariants/`, `tests/contracts/`, and `tests/slots/`.

The most important Slime-facing conditions are:

| BC | Meaning | Slime-facing effect |
|---|---|---|
| `BC-0` | One `PolicyVersionSnapshot` per group dispatch | All sibling rollouts in a GRPO/DAPO group must be generated by the same behavior policy. |
| `BC-1` | Token IDs as integers on every wire | Slime adapter must not decode and re-tokenize samples. |
| `BC-2` | `push_group` is atomic | Slime/trainer never sees partial groups. |
| `BC-3` | `get_batch` pops under lock before returning | A consumed rollout group cannot be trained twice. |
| `BC-4` | `get_batch` waits on fresh groups | Trainer blocks until usable groups exist. |
| `BC-5` | No-progress detector | Trainer fails when rollout production is wedged rather than spinning forever. |
| `BC-9` | `endpoints_failed > 0` is a hard abort | Slime must abort if any inference endpoint fails adapter reload. |
| `BC-11` | LiveStore wire is unpadded | Slime adapter must pad/pack locally. |
| `BC-13` | RolloutManager imports no VERL/OpenHands code | Trainer/environment frameworks stay out of fabric core. |
| `BC-14` | RolloutManager owns parquet data | Slime must not receive parquet paths or own data sampling. |
| `BC-15` | Trainer connects only to LiveStore and PolicyRegistry | Slime must not talk to ProRL, vLLM, or the dataset. |
| `BC-16` | Start trainer only after at least one group exists | Avoid burning no-progress timeout during rollout warm-up. |

### How They Are Represented

Training records are represented by `TrainingSample` in `core/rollout_fabric/schemas/training_sample.py`. The record carries:

- `prompt_token_ids` and `response_token_ids`
- `response_loss_mask`
- optional behavior logprobs
- `reward` and `raw_reward`
- `behavior_policy_version`
- `created_at_step`
- task/environment/verifier provenance

The protobuf wire schema in `core/rollout_fabric/schemas/proto/live_store.proto` stores token arrays as packed bytes. `core/rollout_fabric/live_store/codec.py` handles conversion between Python integer lists and packed wire bytes. This is the concrete enforcement point for `BC-1`.

Policy state is represented by `PolicyVersionSnapshot` and `PolicyVersionCache` in `core/rollout_fabric/schemas/policy_version.py`. The rollout loop captures one snapshot before dispatching sibling episodes.

### How They Are Applied

`RolloutManagerLoop._run_one_group()` in `core/rollout_fabric/rollout_manager/loop.py` is the key application point:

- It takes one policy snapshot for the group.
- It dispatches sibling episodes under that snapshot.
- It builds `TrainingSample` records with `behavior_policy_version` and `created_at_step`.
- It archives all episodes before filtering.
- It filters unusable groups, including zero-variance reward groups.
- It pushes accepted groups to `LiveStore`.

`StoreCore.get_batch()` in `core/rollout_fabric/live_store/store_core.py` applies the hot-buffer constraints:

- It evicts stale groups using server-side `staleness_cutoff_k`.
- It blocks until enough fresh groups exist.
- It raises `NoProgressError` if no producer push arrives within the no-progress timeout.
- It pops selected groups before returning samples.

`PolicyRegistryClient.publish_policy_version()` in `core/rollout_fabric/policy_registry/client.py` applies the hard abort gate:

- It calls the registry with a 600 second deadline.
- If the registry response is not successful, it raises `PublishFailedError`.
- A Slime adapter must not catch `PublishFailedError` and continue training.

`PolicyRegistryServicer.PublishPolicyVersion()` in `core/rollout_fabric/policy_registry/server.py` fans out `/reload_lora` through `fanout_to_pool()` before committing the new version to SQLite and notifying subscribers.

### How They Affect Runtime And Training Behavior

The boundary design protects group-relative RL correctness. GRPO/DAPO-style advantages assume sibling completions are comparable. If sibling episodes are generated under mixed policies, the relative advantage denominator is no longer clean. `BC-0` avoids this by stamping all siblings with one behavior policy version.

`BC-1` protects KL/entropy stability. Token-in/token-out behavior avoids the common failure where decoded text is re-tokenized with shifted boundaries, causing actor/reference or behavior/current policy logprob mismatch.

`BC-3` and `BC-4` make `LiveStore` a hot handoff buffer, not a replay system. The trainer gets a group once, and the store pops it before returning. This reduces duplicate training on the same rollout.

`BC-9` prevents a warm buffer from hiding broken inference reload. If one vLLM child fails reload and the trainer continues, future rollout data can mix policy versions while metrics still look superficially healthy.

For Slime specifically, these constraints mean the Slime integration should be a consumer shim, not a new rollout owner. The custom Slime rollout function should call `LiveStoreClient.get_batch()` and convert samples into Slime's native format. The Slime checkpoint/save path should publish through `PolicyRegistry`, then abort on publish failure.

### Edge Cases And Assumptions

Several source-level caveats matter:

- Slime is not implemented, so the Slime boundary has not been validated against real Slime `Sample` objects or Ray execution.
- `StoreCore` is documented as FIFO, but `get_batch()` randomly samples group indexes before popping them. Capacity eviction is FIFO via `deque(maxlen=...)`, but training consumption is random.
- `LiveStoreClient.get_batch()` sends `staleness_cutoff_k=0`, while `LiveStoreServicer.GetBatch()` uses server-side configuration. This is operationally defensible, but the request field is misleading.
- `PolicyRegistryServicer.PublishPolicyVersion()` can return success even if the manifest write fails. Since the rollout manager can use file polling, this can leave rollout production on an old policy even after vLLM reload succeeds.
- No-progress timeout values are documented inconsistently: LiveStore defaults to 1800 seconds, while some trainer-side comments reference a 5400 second RPC timeout.
- Some invariant tests appear stale. For example, trainer adapter boundary tests scan an older `trainer_adapters/` layout, while active code lives under `trainers/`.

### Strengths

- Strongly documented and test-oriented boundary conditions.
- Narrow trainer contract.
- Token-level wire representation.
- Group-level policy version stamping.
- Clear separation between hot buffer (`LiveStore`) and durable archive (`ReplayArchive`).
- Hard reload abort semantics.

### Weaknesses

- Slime remains unimplemented.
- Config and launch validation are spread across scripts/env vars rather than one typed run schema.
- Runtime observability is more fragmented than prime-rl.
- Some docs and tests have drifted from current code.

## prime-rl: Architecture and Big Idea

### Facts

prime-rl is a vertically integrated async RL framework. It includes:

- Launcher entrypoints: `/tmp/prime-rl/src/prime_rl/entrypoints/rl.py`
- Orchestrator: `/tmp/prime-rl/src/prime_rl/orchestrator/orchestrator.py`
- Scheduler: `/tmp/prime-rl/src/prime_rl/orchestrator/scheduler.py`
- Environment abstraction: `/tmp/prime-rl/src/prime_rl/orchestrator/envs.py`
- Buffer: `/tmp/prime-rl/src/prime_rl/orchestrator/buffer.py`
- Transport: `/tmp/prime-rl/src/prime_rl/transport/`
- Trainer loop: `/tmp/prime-rl/src/prime_rl/trainer/rl/train.py`
- Loss: `/tmp/prime-rl/src/prime_rl/trainer/rl/loss.py`
- Weight broadcast: `/tmp/prime-rl/src/prime_rl/trainer/rl/broadcast/`
- Config schemas: `/tmp/prime-rl/packages/prime-rl-configs/src/prime_rl/configs/`

The big idea is high-throughput async RL at large GPU scale. Its README explicitly targets 1000+ GPU training, 1T+ MoE models, FSDP2, vLLM, FP8 inference, disaggregated inference, expert/context parallelism, Slurm/Kubernetes, SFT/RL/eval workflows, and native `verifiers` environment integration.

### Runtime Flow

The local `rl` entrypoint writes resolved subconfigs, assigns GPUs, starts inference, optional teacher inference, orchestrator, and trainer subprocesses, then supervises them.

The orchestrator is a high-level owner of rollout production:

1. Installs/loads configured `verifiers` environments.
2. Starts environment servers.
3. Sets up rollout inference clients.
4. Builds a `Buffer`.
5. Creates a `Scheduler`.
6. Generates train batches.
7. Computes advantages.
8. Applies filters.
9. Saves rollout artifacts.
10. Converts trajectories into `TrainingSample`.
11. Optionally computes teacher logprobs.
12. Sends `TrainingBatch` to the trainer.

The scheduler keeps rollout requests in flight, pins a group to one inference client, updates inference weights when checkpoints appear, tracks async/off-policy level, cancels stale rollout groups, and returns completed groups.

The trainer consumes batches, packs micro-batches, runs distributed forward/backward, applies loss, steps optimizer/scheduler, checkpoints, broadcasts weights, and logs throughput/MFU/memory/loss metrics.

### Design Choices And Assumptions

prime-rl centralizes algorithmic responsibility in the orchestrator and trainer:

- Advantages are computed in the orchestrator (`orchestrator/advantage.py`).
- Rollout filters are applied before sending trainable samples.
- The trainer's default loss is currently DPPO+KL-style in `trainer/rl/loss.py`, while `docs/async.md` describes an AIPO-style objective. That is a source/doc drift worth noting.
- The transport types include algorithm-specific fields such as `advantage`, teacher logprobs, temperatures, routed experts, multimodal payloads, and SFT/RL loss flags.

Environment handling is standardized around `verifiers`. `Env` wraps `vf.load_environment()`, starts a ZMQ env server if needed, and exposes `run_rollout()` and `run_group()`. The buffer validates datasets, injects `example_id` when absent, and supports env-ratio sampling and difficulty filtering.

Config is a major strength. The separate `prime-rl-configs` package defines typed Pydantic schemas for RL, trainer, orchestrator, inference, SFT, shared logging, checkpointing, deployment, tokenizer, and weight broadcast. Validators catch cross-component mismatch early, such as model name, tokenizer template, checkpoint interval, max steps, max async level, output directory, and weight broadcast type.

## Side-by-Side Comparison

| Area | RolloutFabric / Slime-facing system | prime-rl |
|---|---|---|
| architecture | Contract-first six-service fabric. Dataset, environment, buffer, trainer, policy registry, and inference are separate. | Vertically integrated framework with launcher, orchestrator, trainer, inference, envs, configs, model code, and observability in one repo. |
| training loop | Trainer consumes unpadded samples from `LiveStore`; rollout generation is external to trainer. Current implemented adapter is VERL, not Slime. | Orchestrator generates batches, computes advantages/filters, sends `TrainingBatch`; trainer packs and trains. More complete production loop. |
| environment abstraction | Bespoke EnvironmentProvider contract over HTTP; RolloutManager uses plain `httpx` and imports no env framework. | Standardized `verifiers` abstraction with env servers, dataset validation, `run_rollout`, and `run_group`. |
| boundary/constraint handling | Explicit BC table, token IDs on wire, per-group policy snapshot, pop-on-sample, trainer isolation, hard abort on partial reload. | Async/off-policy constraints encoded in scheduler with `max_async_level`, `max_off_policy_steps`, stale cancellation, and checkpoint barriers. Less trainer-agnostic. |
| configuration | Mostly env vars, shell scripts, docs, and service-specific settings. | Strong typed Pydantic/TOML/CLI/env schemas with cross-component validators and dry-run config generation. |
| extensibility | Better for swapping trainers/environments if adapters honor protocols. Slime not implemented yet. | Better for adding model/training features inside the prime-rl ecosystem; less clean for replacing major roles independently. |
| distributed execution/scaling | Service split can scale rollout production independently, but current implementation is narrower and operationally bespoke. | Strong GPU scaling stack: FSDP2, EP/CP, FP8, NCCL/filesystem broadcast, Slurm/K8s, disaggregated inference, multi-run LoRA. |
| observability/debugging | Health gates, LiveStore metrics, PolicyRegistry publish metrics, tests, runbooks. Fragmented across scripts/services. | W&B/Prime monitor, Prometheus/health servers, event-loop lag, inference metrics, rollout artifacts, sample/distribution logs, throughput/MFU/memory metrics. |
| developer experience | Clear invariants and small core protocols, but setup is script/env heavy and Slime path is not yet real. | Strong examples, config schemas, docs, tests, `uv` workflow, dry-run launcher, many ready configs. More complex codebase. |
| test coverage or validation approach | Boundary/invariant-focused tests: token wire, group integrity, pop-on-sample, registry abort gate, adapter boundaries. Some drift exists. | Broad unit/integration coverage for configs, scheduler, advantage/filter/buffer, model paths, GPU training, LoRA/MoE. |

## Where Our System Is Better

### Facts

RolloutFabric has a cleaner boundary for trainer replacement. The trainer does not own the dataset, environment execution, or inference endpoints. This is explicitly encoded by `BC-14` and `BC-15` in `plans-n-solutions/rollout_fabric.md` and by the `TrainerAdapter` protocol in `core/rollout_fabric/schemas/protocols/trainer_adapter.py`.

The token wire contract is stricter. `TrainingSample` plus protobuf/codec logic keeps token IDs as integers and avoids text re-tokenization. prime-rl also transports token IDs, but its transport schema includes more algorithm and model-specific fields.

Policy version consistency is more explicit. RolloutFabric stamps every group with a behavior policy version and captures one snapshot per group. This is central to `BC-0`.

The hard abort gate for inference reload is clearer. `PolicyRegistryClient.publish_policy_version()` raises `PublishFailedError` if the registry reports failure. This makes mixed-version inference pool states harder to ignore.

The service split is better aligned with the stated goal of plugging in Slime, VERL, ROLL, or another trainer. prime-rl is more cohesive, but the orchestrator owns many decisions that would be hard to preserve across arbitrary trainer frameworks.

### Opinion

The most valuable part of this repo is not the current VERL patch or planned Slime shim. It is the boundary model. Copying prime-rl's vertical orchestration wholesale would erase that advantage.

## Where prime-rl Is Better

### Facts

prime-rl has a more mature async scheduler. `Scheduler.generate_batch()` keeps rollouts in flight, handles policy update tasks, uses checkpoint readiness barriers, cancels stale groups, refills naturally, tracks off-policy metrics, and supports both fixed sample count and token batch targets.

prime-rl has a far stronger configuration system. `RLConfig`, `OrchestratorConfig`, `TrainerConfig`, and shared validation in `prime-rl-configs` catch mismatch before launch. This is a real operational advantage over env-var/script orchestration.

prime-rl has stronger environment abstraction. `Env`, `TrainEnv`, `EvalEnv`, and `Buffer` provide a consistent path for installing/loading environments, validating datasets, assigning example IDs, running single or group rollouts, and mixing multiple environments.

prime-rl has better observability. It logs orchestrator metrics, trainer metrics, inference metrics, event-loop lag, rollout samples, distributions, benchmark results, Prometheus metrics, and health endpoints.

prime-rl has more complete large-model training infrastructure. The trainer includes FSDP/custom model support, MoE paths, context/expert parallelism, LoRA, multimodal handling, routed experts, teacher logprobs, checkpointing, NCCL/filesystem broadcast, Slurm, and Kubernetes support.

prime-rl's tests are broader across framework behavior, configs, scheduler, models, and integration paths.

### Opinion

prime-rl is currently a better end-to-end product for running large async RL jobs, provided the user accepts its architectural assumptions. RolloutFabric is better as a pluggability boundary, but less complete as a training platform.

## What We Should Adapt From prime-rl

### High Impact / Low Effort

1. Add a typed run config layer.

   Create a lightweight `rollout_fabric.configs` package with Pydantic models for service sockets, dataset paths, policy ID, environment ID, LiveStore sizing/staleness, no-progress timeout, inference endpoints, registry DB/manifest paths, trainer adapter settings, and startup gates. Generate service env files from this config. Use prime-rl's `packages/prime-rl-configs/src/prime_rl/utils/validation.py` as the model for cross-component validation.

   File pointers: `ops/services/start_all.sh`, `docs/TRAINING_OPERATIONS.md`, `core/rollout_fabric/`.

2. Add config validation for boundary conditions.

   Validate before launch that trainer config has no parquet path, ProRL URL, or vLLM URL; RolloutManager has no trainer imports; token mode is token-ID mode; no-progress timeout and RPC timeout are coherent; `POLICY_ID` and `ENVIRONMENT_ID` match across services.

   File pointers: `core/rollout_fabric/schemas/protocols/PLUGGING_IN.md`, `tests/invariants/`.

3. Improve metric taxonomy and artifacts.

   Standardize metric names for rollout throughput, group acceptance/filtering, stale drops, store fill, no-progress, policy publish latency, endpoints OK/failed, off-policy age, response length, solve rates, and trainer wait time. Save per-step rollout JSONL artifacts with a bounded retention policy.

   File pointers: `core/rollout_fabric/live_store/store_core.py`, `core/rollout_fabric/replay_archive/`, `ops/`.

4. Fix local drift before building more.

   Update docs/tests around FIFO vs random LiveStore sampling, request-level staleness cutoff, manifest write behavior, Slime stub paths, and timeout values.

   File pointers: `core/rollout_fabric/live_store/store_core.py`, `core/rollout_fabric/live_store/client.py`, `core/rollout_fabric/live_store/server.py`, `core/rollout_fabric/policy_registry/server.py`, `tests/invariants/test_trainer_adapter_boundary.py`, `trainers/slime/scripts/start.sh`.

### High Impact / High Effort

1. Build the actual Slime adapter.

   Implement `trainers/slime/slime_custom/fabric_adapter/slime_rollout_fn.py` and a `pack_for_slime()` converter. Add a Slime save hook or wrapper that publishes through `PolicyRegistryClient`. Add a smoke test with fake Slime sample objects if full Slime is too heavy for CI.

   File pointers: `docs/PLUGGING_IN_NEW_TRAINER_OR_ENVIRONMENT.md`, `trainers/slime/`, `core/rollout_fabric/live_store/client.py`, `core/rollout_fabric/policy_registry/client.py`.

2. Adapt prime-rl-style continuous scheduling without collapsing boundaries.

   Add a scheduler inside `RolloutManager` that keeps target in-flight groups, tracks group state, has explicit stale cancellation, tracks off-policy level, and retries empty/failed rollouts. Keep advantage computation and trainer-specific packing out of the fabric core.

   File pointers: `core/rollout_fabric/rollout_manager/loop.py`, compare `/tmp/prime-rl/src/prime_rl/orchestrator/scheduler.py`.

3. Generalize EnvironmentProvider wrappers.

   Keep the HTTP/protocol boundary, but add an adapter layer similar to prime-rl's `Env` wrapper: dataset validation, example ID injection, `run_rollout`/`run_group` semantics, and group-scoring metadata. Do not import `verifiers` or OpenHands into `RolloutManager` core.

   File pointers: `core/rollout_fabric/rollout_manager/prorl_client.py`, `core/rollout_fabric/schemas/protocols/environment_provider.py`, compare `/tmp/prime-rl/src/prime_rl/orchestrator/envs.py`.

4. Add stronger weight-broadcast backends.

   PolicyRegistry currently fans out HTTP reloads. prime-rl supports filesystem and NCCL-style weight broadcast paths. Adapting a filesystem marker path, and later an NCCL fast path, could improve large-model update latency.

   File pointers: `core/rollout_fabric/policy_registry/fanout.py`, `inference/vllm/`, compare `/tmp/prime-rl/src/prime_rl/trainer/rl/broadcast/`.

### Nice-To-Have

1. Add W&B/Prometheus integration for all services.

   prime-rl's monitor and metrics server are more complete than current scripts. A small shared metrics helper in `core` would make service metrics easier to consume.

2. Add benchmark mode.

   prime-rl keeps benchmark baselines and exports benchmark JSON. A fabric benchmark for rollout throughput, LiveStore latency, policy publish latency, and trainer wait time would make regressions visible.

3. Add multi-environment sampling ratios.

   prime-rl's `Buffer` supports natural distribution or configured ratios. A similar dataset/env mixing layer could be useful once the fabric moves beyond one SWE dataset.

4. Add VLM/teacher-logprob extensions only when needed.

   prime-rl's transport supports multimodal fields, teacher logprobs, routed experts, and temperatures. These are valuable, but adding all of them now would widen the hot contract prematurely.

## Risks and Non-Recommendations

Do not move advantage computation into RolloutFabric core by default. prime-rl computes advantages in the orchestrator, but that couples the rollout producer to algorithm semantics. RolloutFabric should preserve trainer-pluggability unless we intentionally define a trainer-independent advantage protocol.

Do not let the trainer own data loading again. Giving Slime parquet paths or environment addresses would violate `BC-14` and `BC-15` and recreate the hidden-orchestrator problem this fabric is designed to avoid.

Do not copy prime-rl's single orchestrator shape wholesale. Its scheduler is strong, but the vertical role would weaken RolloutFabric's independent service scaling and replaceability.

Do not adopt richer transport fields without versioning the wire contract. Fields like teacher logprobs, routed experts, multimodal bytes, and temperatures are useful but should be optional, versioned extensions to `TrainingSample`, not implicit new requirements for all trainers.

Do not soften `BC-9`. prime-rl's checkpoint/broadcast paths are sophisticated, but RolloutFabric should keep the hard abort principle: a partial inference reload is a correctness failure, not degraded mode.

Do not ignore prime-rl doc/code drift. Its `docs/async.md` describes an AIPO-style objective, while current `trainer/rl/loss.py` implements DPPO+KL-style behavior. When adapting ideas, source code should win over docs.

## Suggested Follow-Up Work

1. Implement the minimal Slime adapter.

   Add `trainers/slime/slime_custom/fabric_adapter/slime_rollout_fn.py`, `pack_for_slime()`, and a post-save publisher. Update `trainers/slime/scripts/start.sh` from stub to runnable entrypoint.

2. Add Slime boundary tests.

   Add tests that fake Slime `Sample` conversion and verify integer token IDs, unpadded input handling, group integrity, no parquet/ProRL/vLLM configuration, and hard abort on `PublishFailedError`.

3. Fix drift in existing invariant tests.

   Update `tests/invariants/test_trainer_adapter_boundary.py` to scan the actual `trainers/` layout. Reconcile stale `trainer_adapters/` references.

4. Decide and document LiveStore sampling semantics.

   Either update docs to say capacity is FIFO but `get_batch()` samples random groups, or change `StoreCore.get_batch()` to true FIFO consumption. The current mixed behavior is confusing.

5. Clarify staleness cutoff ownership.

   Remove or honor request-level `staleness_cutoff_k` in the gRPC API. If server-side config owns staleness, document that and stop sending a misleading client value.

6. Make manifest write failure policy explicit.

   If file polling remains active, manifest write failure should likely make publish fail or trigger a fallback subscription path. Review `PolicyRegistryServicer.PublishPolicyVersion()`.

7. Add a typed fabric run config.

   Start with Pydantic models for the current service env vars and validations for BC-13/14/15/16. Generate the env needed by `ops/services/start_all.sh`.

8. Add prime-rl-style scheduler metrics.

   Track in-flight groups, group retry counts, stale cancellations, async/off-policy age, wait-for-policy time, update-policy time, and empty/errored rollout rates in `RolloutManagerLoop`.

9. Add rollout artifacts and standardized metrics.

   Save compact per-step/group JSONL artifacts and expose metric keys consistent across RolloutManager, LiveStore, PolicyRegistry, and trainer adapters.

10. Prototype a filesystem weight publication backend.

   Compare current `PolicyRegistry` fanout with prime-rl's marker-file `STABLE` approach. A hybrid path could reduce reload ambiguity while preserving the hard abort gate.

