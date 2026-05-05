"""VERL-specific packing of LiveStore samples into a SampledMiniBatch.

Extracted from LiveStoreClient so core has zero trainer imports (BC-13 extended).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from verl_custom.fabric_adapter.pad import pack_unpadded_groups

if TYPE_CHECKING:
    from rollout_fabric.live_store.client import LiveStoreClient


def sample_mini_batch(
    client: LiveStoreClient,
    n_groups: int,
    current_step: int,
):
    """Call get_batch then pack into a VERL SampledMiniBatch."""
    samples = client.get_batch(
        n_groups=n_groups,
        current_step=current_step,
        timeout_ms=5_400_000,  # 90 min — matches no_progress_timeout_s=5400
    )
    return pack_unpadded_groups(
        samples,
        pad_token_id=client._pad_token_id,
        prompt_length_cap=client._prompt_cap,
        response_length_cap=client._response_cap,
        current_step=current_step,
    )
