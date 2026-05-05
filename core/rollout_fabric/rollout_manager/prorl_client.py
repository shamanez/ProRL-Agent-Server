"""HTTP client for the EnvironmentProvider (ProRL :8006).

**No OpenHands imports.** The worker calls ProRL's existing HTTP API
via plain ``httpx`` requests. ProRL remains unchanged (frozen through S4).

HTTP contract (POST /process):
  Request: {"instance": {..., "policy_version": N},
            "sampling_params": {..., "token_level_generation": true}}
  Response: {"messages": [{role, content, token_ids?, logprobs?}, ...],
             "resolved": bool, "success": bool, "finish": bool,
             "error": str|null, "reward": float|null}

The ``token_level_generation=True`` sampling param tells ProRL's LLM
clients (qwen3.py, qwen2_5_vl.py) to communicate with vLLM in token IDs
and include them in the response — preserving §3.1 token-in/token-out.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class ProRLEpisodeResult:
    """Parsed result of one ProRL /process call."""

    def __init__(self, raw: dict[str, Any], instance_id: str) -> None:
        self.instance_id = instance_id
        self.messages: list[dict[str, Any]] = raw.get('messages', [])
        self.resolved: bool = bool(raw.get('resolved', False))
        self.success: bool = bool(raw.get('success', False))
        self.finish: bool = bool(raw.get('finish', True))
        self.error: str | None = raw.get('error') or None
        # Reward is computed by ProRL's eval stage. Fall back to
        # float(resolved) if not present (simple binary reward).
        raw_reward = raw.get('reward')
        self.reward: float = (
            float(raw_reward) if raw_reward is not None else float(self.resolved)
        )
        self.raw: dict[str, Any] = raw


class ProRLClient:
    """Thin HTTP client for ProRL's /process endpoint.

    Parameters
    ----------
    base_url:
        ProRL server base URL, e.g. ``http://localhost:8006``.
    default_sampling_params:
        Sampling parameters merged with per-call overrides.
        Must include ``"token_level_generation": true`` for token IDs
        and logprobs to be present in the response (§3.1).
    timeout_s:
        Per-request timeout. ProRL's global timeout is ~300s; set
        this to ``global_timeout * 2 + 60`` for safety.
    """

    def __init__(
        self,
        base_url: str,
        *,
        default_sampling_params: dict[str, Any] | None = None,
        timeout_s: float = 660.0,
    ) -> None:
        self._base = base_url.rstrip('/')
        self._default_params: dict[str, Any] = {
            'token_level_generation': True,  # §3.1 — token IDs + logprobs in response
            'custom_tokenizer': 'Qwen/Qwen3-4B-Instruct-2507',  # required by llm.py for token-level generation
            'temperature': 0.7,
            'top_p': 0.95,
            'max_output_tokens': 2048,
            'max_iterations': 20,
        }
        if default_sampling_params:
            self._default_params.update(default_sampling_params)
        self._client = httpx.Client(timeout=timeout_s)

    def run_episode(
        self,
        instance: dict[str, Any],
        policy_version: int,
        sampling_params: dict[str, Any] | None = None,
    ) -> ProRLEpisodeResult:
        """Run one complete episode via ProRL.

        ``policy_version`` is stamped on the instance so ProRL's LLM
        clients route to ``/v{N}/generate`` on the vLLM pool (§3.4
        pinning protocol — one trajectory sees one policy).
        """
        inst = dict(instance)
        inst['policy_version'] = int(policy_version)
        params = dict(self._default_params)
        if sampling_params:
            params.update(sampling_params)
        payload = {'instance': inst, 'sampling_params': params}
        instance_id = str(inst.get('instance_id', inst.get('trajectory_id', '?')))
        logger.debug('ProRL /process instance_id=%s pv=%d', instance_id, policy_version)
        resp = self._client.post(f'{self._base}/process', json=payload)
        resp.raise_for_status()
        return ProRLEpisodeResult(resp.json(), instance_id)

    def health(self) -> bool:
        try:
            r = self._client.get(f'{self._base}/health', timeout=5.0)
            return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def close(self) -> None:
        self._client.close()
