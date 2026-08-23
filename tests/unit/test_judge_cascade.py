from data_engineering_copilot.evaluation.generation_eval import _judge_call_with_retry, resolve_judge


class _Cloud:
    def __init__(self):
        self.calls = 0

    async def generate(self, _):
        self.calls += 1
        return '{"score": 0.95}'


async def _drive(judge):
    return await _judge_call_with_retry(judge, "p", 0.0, 1.0, max_retries=1)


async def test_in_band_escalates_to_cloud():
    class _L:
        async def generate(self, _):
            return '{"score": 0.9}'

    cloud = _Cloud()
    judge = resolve_judge(local=_L(), cloud=cloud, enabled=True, threshold=0.85, band=0.15)
    s = await _drive(judge)
    assert cloud.calls == 1
    assert abs(s - 0.95) < 1e-9


async def test_out_of_band_uses_local():
    class _L:
        async def generate(self, _):
            return '{"score": 0.3}'

    cloud = _Cloud()
    judge = resolve_judge(local=_L(), cloud=cloud, enabled=True, threshold=0.85, band=0.15)
    s = await _drive(judge)
    assert cloud.calls == 0
    assert abs(s - 0.3) < 1e-9


async def test_disabled_uses_primary_directly():
    class _Local:
        async def generate(self, _):
            return '{"score": 0.4}'

    class _Cloud2:
        def __init__(self):
            self.calls = 0

        async def generate(self, _):
            self.calls += 1
            return '{"score": 0.7}'

    cloud2 = _Cloud2()
    judge = resolve_judge(local=_Local(), cloud=cloud2, enabled=False, threshold=0.85, band=0.15)
    s = await _drive(judge)
    assert cloud2.calls == 1
    assert abs(s - 0.7) < 1e-9
