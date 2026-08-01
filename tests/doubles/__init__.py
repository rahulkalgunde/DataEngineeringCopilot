"""Deterministic test doubles (see the individual modules for contracts).

Conventions:
- Doubles live in ``tests/doubles/`` and are importable from anywhere in the
  test suite; never re-define them inline in test modules.
- Behavior is explicit and offline — no network, no infra, no magic.
- ``tests/unit/test_doubles_contract.py`` pins each double to its protocol so
  the real implementations and the doubles cannot silently drift apart.
"""

from tests.doubles.embedder import StubEmbedder
from tests.doubles.frontier import InMemoryFrontierDB
from tests.doubles.llm import STUB_ANSWER, STUB_GAP_ANSWER, StaticLLM, StubLLM
from tests.doubles.redis import _StubRedis
from tests.doubles.vector_store import InMemoryVectorStore

__all__ = [
    "STUB_ANSWER",
    "STUB_GAP_ANSWER",
    "InMemoryFrontierDB",
    "InMemoryVectorStore",
    "StaticLLM",
    "StubEmbedder",
    "StubLLM",
    "_StubRedis",
]
