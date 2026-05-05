# vLLM — InferenceBackend

vLLM pool implementation of the `InferenceBackend` protocol (4× GPU, ports 8100-8103).

**Install** (on the remote EC2 inference machine):
```bash
pip install -r inference/vllm/scripts/requirements-remote.txt
```

**Start**:
```bash
bash inference/vllm/scripts/launch_remote_vllm_pool.sh start
```

Health gate: `GET http://<host>:810N/health` → 200 for each port.

See root `CLAUDE.md` for the full startup sequence and boundary conditions.
