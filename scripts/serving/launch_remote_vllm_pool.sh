#!/bin/bash
# Stage 1 Cut C — remote vLLM pool orchestrator.
#
# Brings up 4 × _vllm_child.py instances on the remote `vllm-instance` host,
# one per GPU, bound to the same 8100..8103 ports the trainer already expects.
# No SSH tunnel, no supervisor on the remote — direct HTTP over the public
# EC2 DNS name. User is responsible for the EC2 security-group rule that
# allows inbound 8100-8103 from the trainer box's public IP.
#
# Subcommands:
#   bootstrap   One-time: ssh-keyscan, rsync _vllm_child.py + runner +
#               requirements into ~/vllm-pool/, create venv with python3.12,
#               pip install deps, huggingface-cli download Qwen3-4B.
#   start       Launch 4 backgrounded nohup runners on remote GPUs 0-3.
#               Each writes ~/vllm-pool/pid-<port>.pid and
#               ~/vllm-pool/child-<port>.log. Waits for /health 200 per port
#               via curl through the public DNS.
#   stop        SSH-kill the 4 PIDs. Rsync remote child logs back to
#               /tmp/vllm-child-<port>.log so validate_run.py finds them at
#               the Stage-1 path.
#   publish     Fan POST /reload_lora to all 4 children in parallel with a
#               tar.gz of {adapter_model.safetensors, adapter_config.json}.
#               Used manually before the trainer-side hook lands in Cut 3;
#               partial failure → exit 1 (mixed-version rollouts are a
#               correctness bug, see plans-n-solutions/stages/weight_sync_lora.md).
#
# Usage:
#   bash scripts/serving/launch_remote_vllm_pool.sh bootstrap
#   bash scripts/serving/launch_remote_vllm_pool.sh start
#   bash scripts/serving/launch_remote_vllm_pool.sh stop
#   bash scripts/serving/launch_remote_vllm_pool.sh publish <adapter_dir> --policy-version <N>
set -eo pipefail

# -----------------------------------------------------------------------------
# Configuration — override via env if the topology changes.
# -----------------------------------------------------------------------------
REMOTE_HOST="${REMOTE_HOST:-vllm-instance}"
REMOTE_DNS="${REMOTE_DNS:-ec2-54-145-77-207.compute-1.amazonaws.com}"
REMOTE_POOL_DIR="${REMOTE_POOL_DIR:-/home/ec2-user/vllm-pool}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/usr/local/bin/python3.12}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-36864}"

# GPU -> port mapping. Must stay aligned with external_llm_endpoints in
# run_proagent_qwn3_4B_instruct_weightsync.sh.
GPUS=(0 1 2 3)
PORTS=(8100 8101 8102 8103)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_CHILD="${REPO_ROOT}/scripts/serving/_vllm_child.py"
LOCAL_RUNNER="${REPO_ROOT}/scripts/serving/_remote_vllm_runner.sh"
LOCAL_REQS="${REPO_ROOT}/scripts/serving/requirements-remote.txt"

SSH_BASE=(ssh -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE_HOST")

usage() {
  cat <<EOF >&2
usage: $0 <bootstrap|start|stop|publish>

  publish <adapter_dir> --policy-version <N>
      Fan POST /reload_lora to all ${#PORTS[@]} children. <adapter_dir> must
      contain adapter_model.safetensors and adapter_config.json (PEFT layout
      emitted by verl's _save_checkpoint when lora_rank>0).

Environment overrides:
  REMOTE_HOST     SSH alias (default: vllm-instance)
  REMOTE_DNS     Public DNS for health probe (default: ec2-...amazonaws.com)
  REMOTE_POOL_DIR Remote working directory (default: /home/ec2-user/vllm-pool)
  REMOTE_PYTHON   Remote python3.12 path (default: /usr/local/bin/python3.12)
  MODEL           HF repo id (default: Qwen/Qwen3-4B-Instruct-2507)
  GPU_MEM_UTIL    vLLM --gpu-memory-utilization (default: 0.85)
  MAX_MODEL_LEN   vLLM --max-model-len (default: 36864; covers trainer
                  data.max_prompt_length (31232) + data.max_response_length
                  (4096) = 35328 with slack)
EOF
  exit 2
}

# -----------------------------------------------------------------------------
# bootstrap
# -----------------------------------------------------------------------------
do_bootstrap() {
  echo "[bootstrap] target: $REMOTE_HOST ($REMOTE_DNS)"

  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[bootstrap] ERROR: HF_TOKEN not set. Source /home/ubuntu/.prorl_creds.env first." >&2
    exit 3
  fi

  # 1. Ensure host key is cached. Fail fast if SSH itself breaks.
  if ! "${SSH_BASE[@]}" true; then
    echo "[bootstrap] ERROR: SSH to $REMOTE_HOST failed. If this is a first" >&2
    echo "[bootstrap]        connect, run (after confirming the fingerprint):" >&2
    echo "[bootstrap]          ssh-keyscan -H $REMOTE_DNS >> ~/.ssh/known_hosts" >&2
    exit 3
  fi

  # 2. Remote GPU + driver sanity.
  GPU_LINES=$("${SSH_BASE[@]}" nvidia-smi --query-gpu=memory.total,driver_version --format=csv,noheader)
  echo "[bootstrap] remote GPUs:"
  echo "$GPU_LINES" | sed 's/^/[bootstrap]   /'
  if [[ $(echo "$GPU_LINES" | wc -l) -lt 4 ]]; then
    echo "[bootstrap] ERROR: expected 4 GPUs on remote; got $(echo "$GPU_LINES" | wc -l)" >&2
    exit 3
  fi

  # 3. Confirm python3.12 is present.
  if ! "${SSH_BASE[@]}" "$REMOTE_PYTHON" --version; then
    echo "[bootstrap] ERROR: $REMOTE_PYTHON missing on remote." >&2
    exit 3
  fi

  # 4. mkdir + rsync runner/child/requirements.
  "${SSH_BASE[@]}" mkdir -p "$REMOTE_POOL_DIR"
  echo "[bootstrap] rsync payload to $REMOTE_HOST:$REMOTE_POOL_DIR/"
  rsync -a -e "ssh -o BatchMode=yes" \
    "$LOCAL_CHILD" "$LOCAL_RUNNER" "$LOCAL_REQS" \
    "$REMOTE_HOST:$REMOTE_POOL_DIR/"
  "${SSH_BASE[@]}" chmod +x "$REMOTE_POOL_DIR/_remote_vllm_runner.sh"

  # 5. Create venv + install deps (idempotent — pip skips already-installed).
  echo "[bootstrap] ensure venv + pip install (this can take ~10 GB / several minutes on first run)"
  # shellcheck disable=SC2087
  "${SSH_BASE[@]}" bash <<EOF
set -eo pipefail
cd "$REMOTE_POOL_DIR"
if [[ ! -d venv ]]; then
  "$REMOTE_PYTHON" -m venv venv
fi
source venv/bin/activate
pip install --quiet --upgrade pip wheel
pip install --quiet -r requirements-remote.txt
python -c "import vllm; print('[bootstrap] vllm', vllm.__version__)"
EOF

  # 6. HF weights download. HF_TOKEN forwarded via `env … ssh`.
  # HF_HOME pinned inside the pool dir: the default ~/.cache/huggingface on
  # this AMI is root-owned from a prior install, so we route all HF I/O to a
  # user-owned cache. _remote_vllm_runner.sh must export the same HF_HOME at
  # runtime or vLLM will look in the wrong place.
  echo "[bootstrap] huggingface-cli download $MODEL (skipped if cached)"
  # shellcheck disable=SC2087
  HF_TOKEN="$HF_TOKEN" "${SSH_BASE[@]}" env HF_TOKEN="$HF_TOKEN" bash <<EOF
set -eo pipefail
source "$REMOTE_POOL_DIR/venv/bin/activate"
export HF_HOME="$REMOTE_POOL_DIR/hf-cache"
mkdir -p "\$HF_HOME"
hf download "$MODEL" --quiet
EOF

  echo "[bootstrap] DONE. next: open EC2 SG inbound 8100-8103 from trainer public IP, then run '$0 start'"
}

# -----------------------------------------------------------------------------
# start
# -----------------------------------------------------------------------------
do_start() {
  echo "[start] launching ${#PORTS[@]} remote vLLM children on $REMOTE_HOST"

  # Pairs as a single shell-safe arg-string: "gpu:port gpu:port ..."
  PAIRS=""
  for i in "${!PORTS[@]}"; do
    PAIRS+="${GPUS[$i]}:${PORTS[$i]} "
  done

  # shellcheck disable=SC2087
  "${SSH_BASE[@]}" bash <<EOF
set -eo pipefail
cd "$REMOTE_POOL_DIR"
# Pre-sweep any orphaned VLLM::EngineCore workers from a previous aborted run.
# Without this, a fresh start can race a stuck engine holding the port.
pkill -f "VLLM::EngineCore" 2>/dev/null || true
for pair in $PAIRS; do
  gpu="\${pair%:*}"
  port="\${pair#*:}"
  logf="$REMOTE_POOL_DIR/child-\$port.log"
  pidf="$REMOTE_POOL_DIR/pid-\$port.pid"
  # Reap any previous child on this port (idempotent restart).
  if [[ -f "\$pidf" ]] && kill -0 "\$(cat "\$pidf")" 2>/dev/null; then
    echo "[start]   killing stale pid=\$(cat "\$pidf") on :\$port"
    kill "\$(cat "\$pidf")" 2>/dev/null || \
      echo "[start]   WARN: kill \$(cat "\$pidf") failed (EPERM?); pkill sweep will handle" >&2
    sleep 2
  fi
  echo "[start]   gpu=\$gpu port=\$port log=\$logf"
  REMOTE_POOL_DIR="$REMOTE_POOL_DIR" \
  nohup bash "$REMOTE_POOL_DIR/_remote_vllm_runner.sh" \
    "\$gpu" "\$port" "$MODEL" "$GPU_MEM_UTIL" "$MAX_MODEL_LEN" \
    > "\$logf" 2>&1 &
  disown || true
done
EOF

  echo "[start] waiting for /health 200 on each endpoint (timeout 300 s per port)"
  for port in "${PORTS[@]}"; do
    url="http://${REMOTE_DNS}:${port}/health"
    deadline=$(( $(date +%s) + 300 ))
    while :; do
      if curl -sf --max-time 5 "$url" >/dev/null 2>&1; then
        echo "[start]   :$port OK"
        break
      fi
      if (( $(date +%s) >= deadline )); then
        echo "[start] ERROR: $url did not become healthy within 300 s" >&2
        echo "[start]        tail remote log:" >&2
        "${SSH_BASE[@]}" "tail -n 40 $REMOTE_POOL_DIR/child-${port}.log" >&2 || true
        exit 1
      fi
      sleep 5
    done
  done

  echo "[start] all ${#PORTS[@]} endpoints healthy. register with ProRL (host side):"
  for port in "${PORTS[@]}"; do
    printf '  curl -sX POST http://localhost:8006/add_llm_server -H "Content-Type: application/json" -d "{\\"url\\":\\"http://%s:%s\\"}"\n' \
      "$REMOTE_DNS" "$port"
  done
}

# -----------------------------------------------------------------------------
# stop
# -----------------------------------------------------------------------------
do_stop() {
  echo "[stop] killing remote vLLM children"
  # shellcheck disable=SC2087
  "${SSH_BASE[@]}" bash <<EOF
set -eo pipefail
# Distinguish "pool dir legitimately missing" (benign: nothing to tear down)
# from "cd failed for another reason" (EIO, EACCES) so the latter is not
# masked as success.
if [[ ! -d "$REMOTE_POOL_DIR" ]]; then
  echo "[stop] pool dir absent, nothing to tear down" >&2
  exit 0
fi
cd "$REMOTE_POOL_DIR"
for port in ${PORTS[*]}; do
  pidf="pid-\$port.pid"
  if [[ -f "\$pidf" ]]; then
    pid=\$(cat "\$pidf")
    if kill -0 "\$pid" 2>/dev/null; then
      echo "[stop]   kill pid=\$pid (:\$port)"
      kill "\$pid" 2>/dev/null || \
        echo "[stop]   WARN: kill \$pid failed (EPERM?); pkill sweep will handle" >&2
    fi
    rm -f "\$pidf"
  fi
done
# Belt-and-suspenders sweep. vLLM 0.18 spawns "VLLM::EngineCore" worker
# subprocesses that re-parent to init (PPID=1) when the API-server parent
# exits, so a _vllm_child.py-only sweep leaves them pinning GPU memory.
pkill -f "_vllm_child.py" 2>/dev/null || true
pkill -f "VLLM::EngineCore" 2>/dev/null || true
EOF

  # Rsync remote logs to the trainer-box paths validate_run.py expects.
  # Teardown-time copy is the single source of truth for Stage-1-Cut-C validator
  # compatibility; see plan file § "validate_run.py log-path wrinkle".
  echo "[stop] rsync remote child logs -> /tmp/vllm-child-<port>.log"
  rsync_fails=0
  for port in "${PORTS[@]}"; do
    if ! rsync -a -e "ssh -o BatchMode=yes" \
        "$REMOTE_HOST:$REMOTE_POOL_DIR/child-${port}.log" \
        "/tmp/vllm-child-${port}.log" 2>/dev/null; then
      echo "[stop]   (no remote log for :$port)" >&2
      rsync_fails=$((rsync_fails + 1))
    fi
  done
  if (( rsync_fails > 1 )); then
    echo "[stop] WARNING: $rsync_fails/${#PORTS[@]} log rsyncs failed — check SSH/network" >&2
  fi
  echo "[stop] done."
}

# -----------------------------------------------------------------------------
# publish — fan POST /reload_lora to all children in parallel
# -----------------------------------------------------------------------------
do_publish() {
  local adapter_dir="${1:-}"
  shift || true
  local policy_version=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --policy-version) policy_version="$2"; shift 2;;
      *) echo "[publish] ERROR: unknown arg: $1" >&2; exit 2;;
    esac
  done

  if [[ -z "$adapter_dir" ]]; then
    echo "[publish] ERROR: missing <adapter_dir>" >&2
    echo "[publish] usage: $0 publish <adapter_dir> --policy-version <N>" >&2
    exit 2
  fi
  if [[ -z "$policy_version" ]]; then
    echo "[publish] ERROR: --policy-version <N> is required" >&2
    exit 2
  fi
  if ! [[ "$policy_version" =~ ^[0-9]+$ ]] || (( policy_version < 1 )); then
    echo "[publish] ERROR: policy_version must be a positive integer, got '$policy_version'" >&2
    exit 2
  fi
  if [[ ! -d "$adapter_dir" ]]; then
    echo "[publish] ERROR: adapter_dir not a directory: $adapter_dir" >&2
    exit 2
  fi
  for f in adapter_model.safetensors adapter_config.json; do
    if [[ ! -f "$adapter_dir/$f" ]]; then
      echo "[publish] ERROR: missing required file: $adapter_dir/$f" >&2
      exit 2
    fi
  done

  # Tar into a tmp file. Clean up on any exit path.
  local tarball
  tarball="$(mktemp -t "lora_publish_pv${policy_version}_XXXXXX.tgz")"
  local response_dir
  response_dir="$(mktemp -d -t "lora_publish_resp_XXXXXX")"
  trap 'rm -f "$tarball"; rm -rf "$response_dir"' EXIT

  echo "[publish] policy_version=$policy_version adapter_dir=$adapter_dir"
  tar czf "$tarball" -C "$adapter_dir" adapter_model.safetensors adapter_config.json
  local adapter_bytes
  adapter_bytes=$(stat -c %s "$tarball" 2>/dev/null || stat -f %z "$tarball")
  echo "[publish] tarball=$tarball bytes=$adapter_bytes"

  # Fan-out in parallel. Each curl writes:
  #   response body  -> $response_dir/body-<port>.json
  #   status metrics -> $response_dir/meta-<port>.txt   (http=<code> ttfb=... total=...)
  # Exit status of curl is captured via `wait $pid; echo $? > $response_dir/rc-<port>`.
  local pids=()
  local t_start_s
  t_start_s=$(date +%s)
  for port in "${PORTS[@]}"; do
    (
      # Disable errexit inside the subshell so a non-2xx / network failure
      # does not abort before we record the curl exit code for the tally.
      set +e
      curl -sS -m 60 -X POST "http://${REMOTE_DNS}:${port}/reload_lora" \
        -F "adapter=@${tarball}" \
        -F "policy_version=${policy_version}" \
        -o "$response_dir/body-${port}.json" \
        -w "http=%{http_code} ttfb=%{time_starttransfer} total=%{time_total}\n" \
        > "$response_dir/meta-${port}.txt" 2>"$response_dir/err-${port}.txt"
      echo $? > "$response_dir/rc-${port}"
    ) &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || true
  done
  local t_end_s
  t_end_s=$(date +%s)

  # Tally results.
  local ok=0
  local failed=0
  local failed_ports=()
  for port in "${PORTS[@]}"; do
    local rc meta body http
    rc=$(cat "$response_dir/rc-${port}" 2>/dev/null || echo "?")
    meta=$(cat "$response_dir/meta-${port}.txt" 2>/dev/null || echo "")
    body=$(cat "$response_dir/body-${port}.json" 2>/dev/null || echo "")
    http=$(printf '%s' "$meta" | sed -n 's/.*http=\([0-9]*\).*/\1/p')
    if [[ "$rc" == "0" && "$http" == "200" ]]; then
      ok=$((ok + 1))
      echo "[publish]   :${port} OK  ${meta}  body=${body}"
    else
      failed=$((failed + 1))
      failed_ports+=("$port")
      local err
      err=$(cat "$response_dir/err-${port}.txt" 2>/dev/null || echo "")
      echo "[publish]   :${port} FAIL rc=${rc} ${meta}  body=${body}  err=${err}" >&2
    fi
  done

  echo "[publish] ${ok}/${#PORTS[@]} endpoints OK  wall=$((t_end_s - t_start_s))s  policy_version=${policy_version}"
  if (( failed > 0 )); then
    echo "[publish] ABORT: ${failed} endpoint(s) failed: ${failed_ports[*]}" >&2
    echo "[publish]        mixed-version rollout batches are a correctness bug; operator must investigate before re-running." >&2
    exit 1
  fi
}

# -----------------------------------------------------------------------------
# dispatch
# -----------------------------------------------------------------------------
case "${1:-}" in
  bootstrap) do_bootstrap;;
  start) do_start;;
  stop) do_stop;;
  publish) shift; do_publish "$@";;
  *) usage;;
esac
