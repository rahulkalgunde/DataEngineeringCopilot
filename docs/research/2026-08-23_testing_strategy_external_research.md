# Testing Strategy for a Production Python RAG App — External Research (2026-08-23)

**Scope**: Test-layering for a Python RAG service (unit / integration / e2e), pytest mechanics, async, parallelism, coverage, and anti-patterns — sourced from primary docs only.
**Sources consulted**: pytest docs (marks, fixtures, monkeypatch, flaky), pytest-asyncio, pytest-xdist, coverage.py, testcontainers-python, RESPX, mutmut, Martin Fowler bliki/articles, Google Testing Blog.

---

## TL;DR Checklist

1. **Keep the pyramid shape.** Google's first-guess split is 70% unit / 20% integration / 10% e2e; exact mix varies but must stay pyramid-shaped. Avoid the "ice cream cone" (mostly e2e) and "hourglass" (units + e2e, no integration middle) anti-patterns.
2. **Classify tests by speed + scope + failure isolation, not by name.** Unit = small piece in isolation, fast (<0.1s is the bar; "one tenth of a second is considered slow"), reliable, isolates failures. Integration = a *small group* of units (often two) or one boundary to an external collaborator. E2E = whole deployed system through a user-facing interface.
3. **Test behavior through public interfaces, not internals.** Don't test private methods (a design smell → split the class) and don't test trivial code (getters/setters).
4. **Integration tests are narrow: one integration point at a time**, placed at every serialization/deserialization boundary (HTTP APIs, DB read/write, queues, filesystem). Run external dependencies locally — testcontainers gives you real Postgres/Redis/Qdrant per test run via context managers.
5. **Prefer a real engine over in-memory fakes when dialect matters.** The Practical Test Pyramid calls in-memory-DB-for-tests "risky business" because tests then run against a different database than production; testcontainers removes that tradeoff.
6. **E2E stays minimal and serial**: translate only core user journeys into e2e tests; expect false positives ("notoriously flaky"); reduce count "to a bare minimum."
7. **Register all custom markers and enforce `--strict-markers`** so mistyped markers error instead of silently warning.
8. **Isolate config/env hermetically**: use `monkeypatch.setenv/delenv`/`setitem` (auto-undone after each test) or explicit settings objects built from kwargs; ambient env vars leaking into tests are the classic hermeticity break.
9. **Prefer dependency injection over global patching** — pytest's own docs: "a safer long-term pattern is to make dependencies explicit so they can be passed into the code under test instead of patched globally." When patching, patch where the name is *used*, not where it's defined.
10. **Pin test-double contracts against the real thing.** Fakes drift from reality; contract testing against both fake and real server keeps doubles faithful (the documented cure for brittle Wiremock-style stubs).
11. **Coverage: enable branch coverage (`--branch`), treat *uncovered* code as the signal**, not the percentage. Google bands: 60% acceptable / 75% commendable / 90% exemplary; don't obsess past ~90%. Gate *new/changed* code (~90–99%) rather than mandating repo-wide numbers.
12. **Complement coverage with mutation testing** on critical modules (mutmut): coverage guarantees execution, not assertion strength; mutation score measures whether tests detect behavioral changes. Scope it (mutmut's `max_stack_depth`) to avoid "incidentally tested" functions slowing runs.
13. **pytest-asyncio: pick a mode deliberately.** `strict` is shipped default (explicit markers); `auto` is the docs' recommended default for asyncio-only projects. Keep neighboring tests on the same `loop_scope`; default function-scoped loops give max isolation.
14. **Async tests run sequentially inside their event loop** — total time adds up (~2s+2s=4s); this is intentional isolation. Parallelism comes from xdist processes, not concurrency within a loop.
15. **xdist: use `--dist worksteal` when test durations vary widely** (documented as better load handling with equal-or-better fixture reuse than default `load`); use `--dist loadgroup` + `@pytest.mark.xdist_group` to pin shared-resource-affine tests to one worker; `-n 0` reproduces xdist-order failures serially.
16. **Retry policy: rerun plugins mitigate but never fix flakes** (official docs list them under mitigation). Quarantine (`xfail strict=False`) is "rather dangerous to use permanently." Root-cause state leaks instead; random-order plugins expose order dependence.
17. **Skip-vs-fail semantics**: skip (with conditions/markers) means a precondition is absent (Docker unavailable, model not pulled) — it must be visible and counted, not silent. A broken behavior always fails. If CI gates only on unit tests, monitor integration results with "extra vigilance" (official docs).
18. **Push tests down the pyramid.** When a higher-level test finds a bug, replicate it as a unit test first; drop higher-level tests whose conditions lower levels already cover ("avoid test duplication").

---

## 1. Pyramid layering: ratios, definitions, where real-infra integration fits

**Ratios.** Google's Testing Blog recommends "as a good first guess... a 70/20/10 split: 70% unit tests, 20% integration tests, and 10% end-to-end tests," while stressing the exact mix is team-specific but should "retain that pyramid shape."
Source: <https://testing.googleblog.com/2015/04/just-say-no-to-more-end-to-end-tests.html>

Fowler's bliki frames the essential point as "many more low-level Unit Tests than high level BroadStackTests running through a GUI," with subcutaneous (service/API-level) tests as the intermediate layer. He also flags the escape hatch: the pyramid rests on the assumption that broad-stack tests are slow/brittle/expensive — *"If my high level tests are fast, reliable, and cheap to modify - then lower-level tests aren't needed."*
Sources: <https://martinfowler.com/bliki/TestPyramid.html>

Ham Vocke's Practical Test Pyramid distills Cohn's original to two rules: (1) write tests with different granularity, (2) the more high-level, the fewer. Layer names are explicitly non-canonical ("service test... hard to grasp"; teams should agree on their own consistent vocabulary).
Source: <https://martinfowler.com/articles/practical-test-pyramid.html>

**Definitions that matter operationally:**

| Layer | Definition (per sources) | Feedback properties |
|---|---|---|
| Unit | Small piece of product tested in isolation; may be solitary (all collaborators doubled) or sociable (real in-memory collaborators allowed) — both legitimate | Fast ("one tenth of a second is considered slow"), reliable, isolates failures |
| Integration | "A small group of units, often two units"; Vocke narrows further: *one integration point at a time* with external collaborators replaced or run locally | Verifies coherent wiring; slower, needs external parts |
| E2E | Whole system as deployed, driven like a user (UI or public API) | Biggest confidence, worst feedback loop |

Sources: <https://testing.googleblog.com/2015/04/just-say-no-to-more-end-to-end-tests.html>, <https://martinfowler.com/articles/practical-test-pyramid.html>, <https://martinfowler.com/bliki/TestPyramid.html>

**Mocks vs real infra.** The value of an integration test comes from exercising the *real* collaborator: "start a database, connect your application..., trigger a function..., check the expected data has been written." Mocking the database would make it a unit test wearing an integration costume. For third-party services you can't run locally, run a fake (Wiremock-equivalent) — but see §8 on keeping fakes honest. Never hit production systems from automated tests.
Source: <https://martinfowler.com/articles/practical-test-pyramid.html>

Google's failure table (their composite e2e disaster sketch) shows why layer placement matters economically: sign-in broke → nearly all e2e tests failed; root-causing took days; partner-team failures ruined results. The same bug caught by a focused integration or unit test fails exactly one test pointing at one module.
Source: <https://testing.googleblog.com/2015/04/just-say-no-to-more-end-to-end-tests.html>

## 2. Unit-testing mechanics per pytest docs

- **Fixtures are the DI substrate**: tests request fixtures by parameter name; fixtures can request other fixtures; return values are cached per test so each test gets fresh, isolated state by default. Source: <https://docs.pytest.org/en/stable/how-to/fixtures.html>
- **Scopes**: `function` (default) → destroyed end of test; also `class`, `module`, `package`, `session`. Widen scope only for expensive resources (the docs' own example is network connections). **Dynamic scope** (`scope=<callable>`) is documented as "especially useful... like spawning a docker container" — e.g., session-scoped containers locally, function-scoped in CI. Source: <https://docs.pytest.org/en/stable/how-to/fixtures.html>
- **Safe teardown structure**: prefer `yield` fixtures; keep each fixture to **one state-changing action** plus its teardown, so a mid-setup failure never strands state. Sources: <https://docs.pytest.org/en/stable/how-to/fixtures.html>
- **Parametrization** (via `@pytest.mark.parametrize` or parametrized fixtures) multiplies cases without copy-paste; parametrized fixtures re-run all dependent tests per param; `pytest.param(..., marks=...)` attaches marks (e.g. skip) per case; `ids=` controls test IDs used by `-k`. Sources: <https://docs.pytest.org/en/stable/how-to/fixtures.html>, <https://docs.pytest.org/en/stable/how-to/mark.html>
- **Strict markers**: unregistered marks always warn; with `strict_markers`/`--strict-markers` they become errors — recommended enforcement pattern in the docs. Register vocabulary in config with descriptions. Source: <https://docs.pytest.org/en/stable/how-to/mark.html>
- **monkeypatch vs DI doubles**: monkeypatch provides auto-reverting `setattr/setitem/setenv/delenv/chdir/syspath_prepend` and `context()` for scoped patches. Two doc-mandated disciplines: (a) *patch the reference your code uses* ("if your module does `from os import getcwd`, patch `mymodule.getcwd` rather than `os.getcwd`"); (b) *prefer making dependencies explicit* over global patching for code you control. Autouse fixtures (e.g., deleting `requests.sessions.Session.request`) are the sanctioned way to guarantee no test can reach the network. Source: <https://docs.pytest.org/en/stable/how-to/monkeypatch.html>
- **AAA structure** ("Arrange, Act, Assert") is recommended across all layers for readability. Source: <https://martinfowler.com/articles/practical-test-pyramid.html>

## 3. Integration testing patterns

**testcontainers-python** spins up real dependencies per test run: `with PostgresContainer("postgres:16") as postgres:` → `get_connection_url()` → real SQL against the pinned image version. Key documented properties:

- Context-manager lifecycle gives deterministic teardown; container images are pinned by tag (reproducibility).
- Since v4, install via extras (`testcontainers[postgres]`, etc.) — drivers are *your* dependency, deliberately not bundled.
- Cleanup is enforced by **ryuk** (a reaper sidecar; configurable/disablable via `TESTCONTAINERS_RYUK_*`). Ryuk is the safety net that prevents leaked containers when a test process dies.
- CI specifics are documented: Docker-in-Docker needs the docker client + daemon socket/`DOCKER_HOST`; private registries need `DOCKER_AUTH_CONFIG`.
Source: <https://testcontainers-python.readthedocs.io/en/latest/>

**Health gating**: a container that has started is not a service that is ready. Pattern: start container → poll its readiness (port accept, health endpoint, `SELECT 1`) before tests touch it — the same discipline as production readiness probes. With session-scoped fixtures (dynamic scope, §2), pay startup cost once per worker/run.

**Skip-vs-fail in CI**: infra-dependent suites should *skip with loud reasons* when the platform genuinely can't run them (no Docker), and *fail* when the infra should exist but misbehaves — otherwise red builds get normalized away. Official docs support splitting suites (unit-only CI gate) but warn this lets build-breaking code merge unless integration results are monitored with "extra vigilance." Source: <https://docs.pytest.org/en/stable/explanation/flaky.html>

**Marking conventions**: encode cost and requirements as registered markers (`slow`, `integration`, requires-docker) selectable via `-m`, so CI legs compose (hermetic leg vs Docker leg vs Ollama-heavy leg). Unregistered marker typos are caught by `--strict-markers`. Sources: <https://docs.pytest.org/en/stable/how-to/mark.html>

**Where HTTP mocking sits**: for outbound HTTP in-process, RESPX mocks at the httpx transport layer (`respx_mock` fixture, route patterns, `assert_all_called=True` default catches routes you defined but never exercised) — or without patching via `httpx.MockTransport(router.handler)` injected as DI. That keeps "external API" tests hermetic and fast while real-infra tests cover actual wire behavior. Source: <https://lundberg.github.io/respx/guide/>

## 4. E2E / smoke testing

- **Justification**: keep a *small* number of e2e tests to verify the system as a whole — Google's own pyramid still reserves ~10%; Fowler: high-level tests are a "second line of test defense," and a high-level failure implies a missing lower-level test (write the unit-level replication first). Sources: <https://testing.googleblog.com/2015/04/just-say-no-to-more-end-to-end-tests.html>, <https://martinfowler.com/bliki/TestPyramid.html>
- **Selection rule**: convert "high-value interactions users will have... core value of your product" journeys into e2e; everything more "will likely be more painful than helpful." Source: <https://martinfowler.com/articles/practical-test-pyramid.html>
- **Execution**: e2e suites share mutable system state (deployed stack, data), so run them serially or partition state carefully; parallel e2e is a classic self-interference generator. Pipeline ordering follows *speed*: fast/narrow stages early, broad/slow last ("defining stages... driven by their speed and scope"). Sources: <https://martinfowler.com/articles/practical-test-pyramid.html>
- **Flake control**: pytest's flaky page attributes flakes to uncontrolled system state ("higher level tests are more likely to be flaky as they rely on more state"), overly strict assertions (use `pytest.approx`), and thread-safety misuse. Mitigations ranked by the docs: fix state isolation > rewrite at lower level or delete if redundant > quarantine > rerun plugins. Rerun plugins (`pytest-rerunfailures`, `pytest-flakefinder`) "mitigate the negative effects... by giving them additional chances to pass" — i.e., an acknowledged mitigation, not a cure. Randomizers (`pytest-randomly`, `pytest-random-order`) actively expose order dependence. Source: <https://docs.pytest.org/en/stable/explanation/flaky.html>
- **Quarantine debate**: `xfail(strict=False)` works as manual quarantine but is "rather dangerous to use permanently" — quarantined suites rot silently. Source: <https://docs.pytest.org/en/stable/explanation/flaky.html>

## 5. Coverage & mutation testing

- **Branch > line**: statement coverage reports the `my_partial_fn` example fully covered even though the false-path jump never happens; `coverage run --branch` tracks line-pair transitions and reports partial branches (e.g., missing `4->2` loop-back). Percentages blend statements + branch destinations as "executions ÷ opportunities." Excluded code (`# pragma: no cover`) drops the corresponding branch from consideration entirely. Sources: <https://coverage.readthedocs.io/en/latest/branch.html>
- **What the number means**: Google's position — coverage "is not a perfect measure of test quality... a lossy and indirect metric"; high coverage "does not guarantee that the covered lines have been tested *correctly*"; the greatest value is showing "**what's not covered**." There is no universal ideal number: Google offers 60% acceptable / 75% commendable / 90% exemplary, warns against top-down mandates (checkbox effect), and says gains past a point are logarithmic ("we should not be obsessing on how to get from 90% to 95%"). Per-commit/new-code targets of 99% are called reasonable with 90% a good floor. Integration/e2e coverage is largely *incidental*, not deliberate. Sources: <https://testing.googleblog.com/2020/08/code-coverage-best-practices.html>
- **Mutation testing as the complement**: mutation tools (mutmut; cosmic-ray is the other major Python option) make subtle source changes (`<`→`<=`, `break`→`continue`, integer ±1) and check whether any test fails; survivors = executed-but-unasserted code. Google's coverage post explicitly names mutation testing as "a better technique to assess whether you're adequately exercising the lines your tests cover, and adequately asserting on failures." Practical mutmut guidance: it caches incrementally, maps which tests exercise which functions, and offers `max_stack_depth` because "incidentally tested functions lead to slow mutation testing... and bad test suites." Treat it as a periodic audit of critical modules (parsers, retrieval logic), not a CI-per-commit gate. Sources: <https://testing.googleblog.com/2020/08/code-coverage-best-practices.html>, <https://github.com/boxed/mutmut>

## 6. Async testing (pytest-asyncio)

- **Loop scoping model**: pytest-asyncio provides one event loop per pytest collector; by default each test gets the narrowest (function-collector) loop "for the highest level of isolation between tests." Sharing ancestor loops is opt-in via `@pytest.mark.asyncio(loop_scope="module")` etc. Docs advise: "It is highly recommended for neighboring tests to use the same event loop scope" — mixed scopes in one module are discouraged as hard to follow. Source: <https://pytest-asyncio.readthedocs.io/en/latest/concepts.html>
- **Modes**: `strict` (default — only marked tests/decorated fixtures handled; chosen so plugin ecosystems coexist) vs `auto` (plugin claims all async tests/fixtures; described as "the recommended default" for asyncio-only projects). Pick one project-wide; mixing causes silently-skipped async tests. Source: <https://pytest-asyncio.readthedocs.io/en/latest/concepts.html>
- **No intra-loop concurrency**: async tests run sequentially even when written with sleeps (2s + 2s = ~4s total) — "intentional and important for maintaining test isolation." Speed comes from process parallelism (§7), not spawning concurrent tests. Async fixtures work under either mode but in strict mode require `@pytest_asyncio.fixture` specifically. Source: <https://pytest-asyncio.readthedocs.io/en/latest/concepts.html>

## 7. Parallelism (pytest-xdist)

- **Process counts**: `-n auto` = physical cores, `-n logical` = logical cores (Python ≥3.13 or psutil), `-n 0` disables distribution entirely (the documented way to reproduce order-dependent failures serially), `--maxprocesses` caps workers. Overridable via `PYTEST_XDIST_AUTO_NUM_WORKERS` or the `pytest_xdist_auto_num_workers` hook (hook > env var > interpreter flags). Source: <https://pytest-xdist.readthedocs.io/en/latest/distribution.html>
- **Distribution modes**: `load` (default; no order guarantee), `loadscope` (whole modules/classes to one worker — good for expensive module/class fixtures), `loadfile`, `loadgroup` (**`xdist_group` mark pins tests sharing a group name to the same worker** — the documented answer for tests sharing a container/port/database), `worksteal`, `no`.
- **worksteal**: initial even split, then workers that run low steal from others' queues (needs ≥2 pending items); documented as handling "tests with significantly differing duration better" with "similar or better reuse of fixtures" than `load` — a strong default for heterogeneous suites. Source: <https://pytest-xdist.readthedocs.io/en/latest/distribution.html>
- **Hazards**: the flaky-docs list for parallel failures is exactly the shared-state checklist — a test failing to clean up, dependence on a previous test's data, global-state mutation ("Tests that modify global state typically cannot be run in parallel"). Fixtures themselves are the cleanup mechanism (§2). Also note pytest itself is single-threaded; spawned threads need joining, and `pytest.raises`/`pytest.warns` are not thread-safe. Sources: <https://docs.pytest.org/en/stable/explanation/flaky.html>, <https://pytest-xdist.readthedocs.io/en/latest/distribution.html>

## 8. Anti-patterns catalog

| Anti-pattern | Why it hurts | Source-backed correction |
|---|---|---|
| Ice cream cone (mostly e2e) | Slow builds, flaky signals, days-to-root-cause failures | 70/20/10 shape; think smaller not larger |
| Hourglass (units + e2e, thin middle) | Wiring bugs surface only in slowest layer | Add narrow integration tests at boundaries |
| Leaky ambient env/config in tests | Non-hermetic: local `.env`/shell vars change outcomes; order-dependent CI | Build settings from explicit kwargs; `monkeypatch.setenv/delenv`; autouse guard deleting network entrypoints |
| Order dependence | Flakes under xdist; hidden coupling | Fresh fixtures per test; `-n 0` to reproduce; random-order plugins to expose |
| Over-mocking / drifting fakes | Fake answers ≠ real behavior; tests pass, prod breaks | Keep doubles faithful via contract checks against the real implementation (Vocke's cure for the Wiremock dilemma); prefer DI seams over patching |
| Patching where defined instead of where used | Fragile patches, breaks under refactors | "patch `mymodule.getcwd` rather than `os.getcwd`" |
| Testing privates / trivial code | Refactor-coupled tests; wasted maintenance | Test public observable behavior; split oversized classes; skip getters/setters |
| Slow "unit" suite (hitting disk/net/DB) | Developers stop running it pre-commit | Stub outermost slow collaborators; thousands of unit tests should run in minutes |
| Integration test pretending to be unit (mocked DB, named "integration") | False confidence about real wiring | Real collaborator via testcontainers, or rename honestly |
| Coverage-chasing | Copy-pasted low-value tests, technical debt | Measure uncovered risk instead; mutation-score critical modules |
| Permanent quarantine / reflexive retries | Silent rot; masked genuine failures | Time-boxed quarantine; root-cause state leaks; docs call permanent xfail-nonstrict "dangerous" |
| In-memory DB standing in for prod engine | Dialect/behavior divergence | Real engine in a container ("risky business" otherwise) |
| Redundant high-level duplicates of covered conditions | Slower suite, double maintenance | "Push your tests as far down the test pyramid as you can" |

Sources for rows: <https://testing.googleblog.com/2015/04/just-say-no-to-more-end-to-end-tests.html>, <https://martinfowler.com/articles/practical-test-pyramid.html>, <https://docs.pytest.org/en/stable/explanation/flaky.html>, <https://docs.pytest.org/en/stable/how-to/monkeypatch.html>, <https://testing.googleblog.com/2020/08/code-coverage-best-practices.html>, <https://martinfowler.com/bliki/TestPyramid.html>

---

## Full source list

1. Martin Fowler, *Test Pyramid* (bliki, 2012) — https://martinfowler.com/bliki/TestPyramid.html
2. Ham Vocke, *The Practical Test Pyramid* (2018) — https://martinfowler.com/articles/practical-test-pyramid.html
3. Mike Wacker, *Just Say No to More End-to-End Tests*, Google Testing Blog (2015) — https://testing.googleblog.com/2015/04/just-say-no-to-more-end-to-end-tests.html
4. Arguelles/Ivanković/Bender, *Code Coverage Best Practices*, Google Testing Blog (2020) — https://testing.googleblog.com/2020/08/code-coverage-best-practices.html
5. pytest docs: marks — https://docs.pytest.org/en/stable/how-to/mark.html
6. pytest docs: fixtures — https://docs.pytest.org/en/stable/how-to/fixtures.html
7. pytest docs: monkeypatch — https://docs.pytest.org/en/stable/how-to/monkeypatch.html
8. pytest docs: flaky tests explanation — https://docs.pytest.org/en/stable/explanation/flaky.html
9. pytest-asyncio concepts — https://pytest-asyncio.readthedocs.io/en/latest/concepts.html
10. pytest-xdist distribution — https://pytest-xdist.readthedocs.io/en/latest/distribution.html
11. coverage.py branch coverage — https://coverage.readthedocs.io/en/latest/branch.html
12. testcontainers-python — https://testcontainers-python.readthedocs.io/en/latest/
13. RESPX user guide — https://lundberg.github.io/respx/guide/
14. mutmut README — https://github.com/boxed/mutmut
