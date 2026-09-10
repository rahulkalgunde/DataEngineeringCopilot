# Planner-Worker Architecture via SDD

## Status
Proposed — 2026-09-10

## Context
Single-session sequential execution creates a babysitting loop where the human becomes the bottleneck. The agent runs everything inline, burning context tokens on sequential tasks that could be parallelized. Existing RULE 16 in `opencode.json` describes subagent dispatch but lacks the workflow infrastructure to make it reliable.

## Decision
Adopt the `subagent-driven-development` (SDD) skill as the primary orchestration pattern. Decompose feature work into independent worker tasks, dispatch ephemeral subagents, and integrate results through a verified pipeline.

## Architecture

### 1. Task Decomposition Protocol
Decision tree for breaking features into independent worker tasks:
```
Feature Request
  ├── Can it be decomposed by file?
  │   ├── YES → One task per file (workers edit different files)
  │   └── NO → Can it be decomposed by layer?
  │       ├── YES → One task per layer (domain → infrastructure → services)
  │       └── NO → Can it be decomposed by concern?
  │           ├── YES → One task per concern (schema + logic + tests)
  │           └── NO → Too coupled for parallel dispatch → sequential in planner
  └── Cross-cutting concerns (DI wiring, settings, CLI)
      └── LAST: planner handles after all workers complete
```

**Rules:**
- Workers must NOT edit the same file (if they would, split differently or go sequential)
- Each task touches 1-3 files max (if more, decompose further)
- Cross-cutting concerns (factory.py, settings.py, cli.py) are planner-only
- Dependency order is explicit: "Task B depends on Task A's output"

### 2. Task Brief Format
Self-contained files in `.superpowers/sdd/<plan>/task-N-brief.md`:
```markdown
# Task N: <Short Description>

## Objective
<One sentence: what to build/fix>

## Files to Modify
- `path/to/file.py` — <what changes in this file>

## Interface Contract
<What this task's output must provide to dependent tasks>
- Function/class signatures that must be exposed
- Return types and invariants
- Error handling patterns

## Existing Pattern to Follow
<Code snippet from existing codebase showing the exact pattern to replicate>

## Verification
<Exact command to run after implementation>

## Constraints
- Do NOT modify files outside the listed scope
- Use `make_settings()` for test settings (never `AppSettings()` directly)
- Use `StubEmbedder`/`StubLLM` for test doubles (never real providers)
- Follow frozen dataclass patterns for domain models
```

### 3. Worker Dispatch Protocol
```
1. Record BASE commit: git rev-parse HEAD
2. Create workspace: .superpowers/sdd/<plan-basename>/
3. Write task briefs to workspace (one file per task)
4. Dispatch independent tasks in PARALLEL (same response = parallel)
5. Dispatch dependent tasks SEQUENTIALLY (after dependency completes)
6. Each worker: reads brief → reads source files → implements → verifies → writes report
7. Planner: reads report → reviews → decides fix loop or next task
```

**Fix loop (up to 5 rounds):**
- Rounds 1-3: Resume original worker with specific fix instructions
- Rounds 4-5: Dispatch fresh worker on more capable model
- Circuit breaker at round 5: Planner adjudicates remaining findings

### 4. Shared State Mechanism
Disk-based artifacts in `.superpowers/sdd/<plan-basename>/`:
- `progress.md` — Ledger: decisions, completions, fix rounds
- `task-N-brief.md` — Input for worker N
- `task-N-report.md` — Output from worker N
- `review-package.md` — Final branch review

**Dependency resolution:** Task briefs for dependent tasks include the interface contract from the dependency. Planner verifies the interface contract is satisfied before dispatching dependent tasks.

**Conflict prevention:** Workers must NOT edit the same file (enforced by decomposition protocol). Cross-cutting concerns are planner-only.

### 5. Convention Enforcement
- **Pattern snippets:** Briefs include actual code snippets from the codebase showing the exact pattern to follow
- **Verification commands:** `ruff check`, `ruff format`, `pyright`, targeted `pytest` catch convention violations automatically
- **Convention checklist:** Explicit checklist in every brief (make_settings, frozen dataclasses, test doubles, etc.)

### 6. Verification Protocol
Three layers:
1. **Worker self-verification:** Worker runs Tier 1 checks after implementation
2. **Planner review:** Planner dispatches reviewer subagent with brief + report + diff; two verdicts required (spec-compliance + task-quality)
3. **Integration verification:** Tier 2 full gate after all tasks complete

### 7. Integration Protocol
After all workers complete:
1. Read all worker reports
2. Verify no file conflicts
3. Run integration verification (Tier 2)
4. Handle cross-cutting concerns (factory, settings, CLI)
5. Atomic commits per task
6. Update plan checkboxes → COMPLETE
7. Save session context

## Consequences
**Positive:**
- Eliminates babysitting loop — planner dispatches, workers execute independently
- Parallel execution of independent tasks (3-5x speedup for multi-file features)
- Convention compliance enforced by verification, not worker memory
- Recovery from failures via fix loops and circuit breakers

**Negative:**
- More ceremony for simple tasks (single-file fixes still go through decomposition)
- Requires planner to decompose accurately (bad decomposition = conflicts)
- Disk-based state adds file management overhead

## Verification
- Test the workflow on a small feature (2-3 tasks)
- Verify workers can read briefs, implement, and write reports
- Verify planner can integrate results without conflicts
- Verify fix loop handles worker failures

## Next Steps
1. Write spec (complete)
2. Commit spec (complete)
3. Spec self-review (complete)
4. User reviews spec
5. Transition to implementation via `writing-plans` skill
