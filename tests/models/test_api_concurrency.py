"""Concurrent generation across gen_kwargs groups (API models, no network)."""

import asyncio

import pytest

from lm_eval.api.instance import Instance
from lm_eval.models.openai_completions import LocalCompletionsAPI


def _api(num_concurrent: int) -> LocalCompletionsAPI:
    return LocalCompletionsAPI(
        base_url="http://test-url.com",
        model="test-model",
        tokenizer_backend=None,
        tokenized_requests=False,
        num_concurrent=num_concurrent,
    )


def _instances(spec: list[tuple[str, int]]) -> list[Instance]:
    """One generate_until instance per (context, max_gen_toks); max_gen_toks sets the group."""
    return [
        Instance(
            request_type="generate_until",
            doc={},
            arguments=(ctx, {"max_gen_toks": max_gen_toks, "until": []}),
            idx=i,
        )
        for i, (ctx, max_gen_toks) in enumerate(spec)
    ]


class FakeServer:
    """Stands in for `amodel_call`: holds a semaphore slot like the real one, no HTTP."""

    def __init__(self, delays: dict[str, float] | None = None):
        self.delays = delays or {}
        self.in_flight = 0
        self.max_in_flight = 0
        self.start_order: list[str] = []
        self.events: dict[str, asyncio.Event] = {}

    def event(self, ctx: str) -> asyncio.Event:
        return self.events.setdefault(ctx, asyncio.Event())

    async def __call__(self, *, session, sem, messages, gen_kwargs=None, **kwargs):
        async with sem:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            try:
                out = []
                for ctx in messages:
                    self.start_order.append(ctx)
                    self.event(ctx).set()
                    wait_for = self.delays.get(ctx)
                    if isinstance(wait_for, str):
                        # Block until request `wait_for` has started.
                        await asyncio.wait_for(self.event(wait_for).wait(), timeout=5)
                    else:
                        await asyncio.sleep(wait_for or 0.01)
                    out.append(f"{ctx}@{gen_kwargs['max_gen_toks']}")
                return out
            finally:
                self.in_flight -= 1


def test_groups_overlap_instead_of_waiting(monkeypatch):
    # Group 100 holds a request that only finishes once group 200's request has started:
    # with one event loop per group, group 200 would wait for group 100 and the fake
    # server's wait would time out, so completing at all shows the groups overlap.
    api = _api(num_concurrent=2)
    server = FakeServer(delays={"slow-a": "b"})
    monkeypatch.setattr(api, "amodel_call", server)
    res = api.generate_until(_instances([("slow-a", 100), ("a", 100), ("b", 200)]))
    assert res == ["slow-a@100", "a@100", "b@200"]


def test_num_concurrent_is_a_global_limit_reached_across_groups(monkeypatch):
    # 3 groups of 3 requests, 4 slots: the limit is filled across groups and never exceeded.
    api = _api(num_concurrent=4)
    server = FakeServer(delays={f"g{g}-{i}": 0.05 for g in range(3) for i in range(3)})
    monkeypatch.setattr(api, "amodel_call", server)
    spec = [(f"g{g}-{i}", 100 * (g + 1)) for g in range(3) for i in range(3)]
    res = api.generate_until(_instances(spec))
    assert res == [f"{ctx}@{n}" for ctx, n in spec]
    assert server.max_in_flight == 4


def test_results_follow_request_order_with_interleaved_groups(monkeypatch):
    api = _api(num_concurrent=3)
    server = FakeServer()
    monkeypatch.setattr(api, "amodel_call", server)
    spec = [("q1", 10), ("long question 2", 20), ("q3", 10), ("q4", 30), ("question 5", 20)]
    res = api.generate_until(_instances(spec))
    assert res == [f"{ctx}@{n}" for ctx, n in spec]


@pytest.mark.parametrize("num_concurrent", [2, 3])
def test_requests_start_in_group_order(monkeypatch, num_concurrent):
    # Groups in order of first appearance, longest context first within a group.
    api = _api(num_concurrent=num_concurrent)
    server = FakeServer()
    monkeypatch.setattr(api, "amodel_call", server)
    spec = [("a", 1), ("bbb", 2), ("aaaa", 1), ("b", 2), ("aa", 1)]
    api.generate_until(_instances(spec))
    assert server.start_order == ["aaaa", "aa", "a", "bbb", "b"]


def test_get_batched_requests_is_unchanged(monkeypatch):
    # The loglikelihood path still uses get_batched_requests (one event loop per call).
    api = _api(num_concurrent=2)
    server = FakeServer()
    monkeypatch.setattr(api, "amodel_call", server)
    out = asyncio.run(
        api.get_batched_requests(
            ["x", "y"], [("x", None), ("y", None)], gen_kwargs={"max_gen_toks": 5}
        )
    )
    assert out == [["x@5"], ["y@5"]]
