"""Retrieval-only eval must detach the query rewriter (zero LLM calls)."""

from data_engineering_copilot.cli import _disable_rewrites_for_eval


class _FakeRewriter:
    async def async_rewrite(self, q):  # pragma: no cover - must never be called
        raise AssertionError("rewrite called during eval")

    async def expand_queries(self, q, n):  # pragma: no cover
        raise AssertionError("expand called during eval")


class _FakeService:
    def __init__(self):
        self.query_rewriter = _FakeRewriter()


def test_disable_rewrites_detaches_rewriter():
    svc = _FakeService()
    mode = _disable_rewrites_for_eval(svc)
    assert mode == "disabled"
    assert svc.query_rewriter is None


def test_disable_rewrites_tolerates_missing_attr():
    class _Bare: ...

    svc = _Bare()
    assert _disable_rewrites_for_eval(svc) == "disabled"
