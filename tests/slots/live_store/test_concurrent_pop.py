"""S1 — concurrent samplers see disjoint groups under gRPC.

§3.6 (pop-on-sample) under the gRPC boundary. Two clients drawing
2 groups each from a buffer of 8 must never share a sample_uid.
"""

from __future__ import annotations

import threading

import pytest

from .conftest import make_sample

pytestmark = pytest.mark.invariant


def test_two_clients_disjoint(live_store_server) -> None:
    from live_store.client import LiveStoreClient

    _, socket = live_store_server
    common_kwargs = dict(
        policy_id='qwen3-4b-skyrl',
        environment_id='swe_agent',
        environment_version='v1',
        verifier_version='v1',
        split='train',
    )

    # Use one writer client and two reader clients.
    writer = LiveStoreClient(socket, **common_kwargs)
    for i in range(8):
        writer.push_group([make_sample(sample_uid=f's{i}', group_uid=f'g{i}')])

    seen: list[set[str]] = [set(), set()]
    barrier = threading.Barrier(2)

    def draw(idx: int) -> None:
        cli = LiveStoreClient(socket, **common_kwargs)
        try:
            barrier.wait()
            samples = cli.get_batch(n_groups=2, current_step=0, timeout_ms=2_000)
            seen[idx] = {s.group_uid for s in samples}
        finally:
            cli.close()

    threads = [threading.Thread(target=draw, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    writer.close()

    assert seen[0] and seen[1]
    assert seen[0].isdisjoint(seen[1]), (
        f'concurrent gRPC samplers shared groups: {seen[0] & seen[1]}'
    )
