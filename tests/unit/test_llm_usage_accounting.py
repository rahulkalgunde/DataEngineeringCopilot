from data_engineering_copilot.domain.models import LLMUsage
from data_engineering_copilot.infrastructure.provider_fallback import (
    FallbackChainConfig,
    ProviderConfig,
    ProviderFallbackChain,
    ProviderHealthRegistry,
    UsageLedger,
)


def test_record_and_snapshot():
    UsageLedger.reset()
    UsageLedger.record("judge", {"calls": 1, "prompt_tokens": 10, "completion_tokens": 5})
    UsageLedger.record("judge", {"calls": 1, "prompt_tokens": 7, "completion_tokens": 3})
    UsageLedger.record("generator", {"calls": 1, "prompt_tokens": 100, "completion_tokens": 50})
    snap = UsageLedger.snapshot()
    assert snap["judge"] == {"calls": 2, "prompt_tokens": 17, "completion_tokens": 8}
    assert snap["generator"] == {"calls": 1, "prompt_tokens": 100, "completion_tokens": 50}


def test_reset_clears():
    UsageLedger.record("x", {"calls": 1, "prompt_tokens": 1, "completion_tokens": 1})
    UsageLedger.reset()
    assert UsageLedger.snapshot() == {}


class _FailingClient:
    model = "fail-model"

    async def call(self, request: str) -> str:
        raise RuntimeError("provider down")

    async def close(self) -> None: ...

    @property
    def last_usage(self):
        return LLMUsage(prompt_tokens=999, completion_tokens=999)  # stale decoy


class _ServingClient:
    model = "serve-model"

    def __init__(self, usage: LLMUsage | None = None) -> None:
        self.last_usage = usage or LLMUsage(prompt_tokens=12, completion_tokens=34)

    async def call(self, request: str) -> str:
        return "ok"

    async def close(self) -> None: ...


class _SilentInlineUsageClient:
    """Serves fine but never updates last_usage; reports usage inline on the result."""

    model = "silent-model"

    def __init__(self) -> None:
        self.last_usage = LLMUsage(prompt_tokens=0, completion_tokens=0)

    async def call(self, request: str) -> object:
        class _Result:
            text = "ok"
            usage = LLMUsage(prompt_tokens=50, completion_tokens=9)

            def __str__(self) -> str:
                return self.text

        return _Result()

    async def close(self) -> None: ...


def _chain(providers: list[ProviderConfig]) -> ProviderFallbackChain:
    health = ProviderHealthRegistry()
    for p in providers:
        health.register_provider(p.name, [p.client.model])
    return ProviderFallbackChain(config=FallbackChainConfig(providers=providers), health=health)


async def test_generate_records_tokens_from_serving_provider_not_first():
    """Tokens must come from the client that SERVED (2nd), not stale first-provider usage."""
    UsageLedger.reset()
    chain = _chain(
        [
            ProviderConfig(name="broken", client=_FailingClient()),
            ProviderConfig(name="worker", client=_ServingClient()),
        ]
    )
    out = await chain.generate("hi")
    assert out == "ok"
    snap = UsageLedger.snapshot()
    assert snap["llm"] == {"calls": 1, "prompt_tokens": 12, "completion_tokens": 34}


async def test_generate_sums_tokens_across_both_providers():
    """Both providers report tokens across two calls → ledger totals are the sum."""
    UsageLedger.reset()
    alpha = _ServingClient(LLMUsage(prompt_tokens=12, completion_tokens=34))
    beta = _ServingClient(LLMUsage(prompt_tokens=20, completion_tokens=8))
    health = ProviderHealthRegistry()
    health.register_provider("alpha", [alpha.model])
    health.register_provider("beta", [beta.model])
    chain = ProviderFallbackChain(
        config=FallbackChainConfig(
            providers=[
                ProviderConfig(name="alpha", client=alpha),
                ProviderConfig(name="beta", client=beta),
            ]
        ),
        health=health,
    )
    await chain.generate("q1")  # served by alpha
    health.mark_provider_cooldown("alpha", 60)
    await chain.generate("q2")  # alpha skipped (cooldown), beta serves
    snap = UsageLedger.snapshot()
    assert snap["llm"] == {"calls": 2, "prompt_tokens": 32, "completion_tokens": 42}


async def test_generate_falls_back_to_inline_result_usage_when_last_usage_zero():
    """Provider omitted the usage block (last_usage stays 0) — tokens recovered from result.usage."""
    UsageLedger.reset()
    chain = _chain([ProviderConfig(name="quiet", client=_SilentInlineUsageClient())])
    out = await chain.generate("hi")
    assert str(out) == "ok"
    snap = UsageLedger.snapshot()
    assert snap["llm"] == {"calls": 1, "prompt_tokens": 50, "completion_tokens": 9}
