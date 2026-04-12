# Core Platform Contracts And Benchmarks

Use the downstream-inspired contract suite and PIT benchmark harness as the
minimum regression gate for public-surface work in the core platform epic.

## Contract suite

The contract tests are intentionally modeled on the three downstream usage
patterns that shaped the roadmap:

- `tests/contracts/test_nowcast_pit_contract.py`
  covers ref-period PIT snapshots and panel-building in a nowcast-style flow
- `tests/contracts/test_volatility_dataset_contract.py`
  covers adapter-backed `DataContext` loading plus notebook-style volatility
  dataset assembly
- `tests/contracts/test_operations_contract.py`
  covers release-aware health reporting and archive fetch planning
- `tests/contracts/test_pit_benchmarks.py`
  smoke-tests the benchmark harness interface so performance tracking itself
  stays callable

Run the full slice with:

```bash
python -m pytest tests/contracts
```

## When this gate is required

Run `tests/contracts` whenever a change touches:

- `alphaforge.time`
- `alphaforge.pit`
- `alphaforge.data.context`
- `alphaforge.features.dataset_*`
- built-in market templates
- release-aware health reporting or archive fetch planning
- any compatibility shim or migration surface that downstream repos still use

Future deprecations should cite the specific contract file that covers the
behavior they are changing.

## PIT benchmark harness

The PIT benchmark harness lives in `benchmarks/pit.py` and exercises the two
highest-value regression paths from the roadmap:

- `PITAccessor.snapshot_ref(...)`
- `PITAccessor.build_snapshot_panel_long(...)`

Run it with:

```bash
python -m benchmarks.pit
```

Optional knobs:

- `--iterations`
- `--periods`
- `--series-count`
- `--revisions-per-period`

Treat the benchmark output as a baseline snapshot, not as a cross-machine SLA.
Compare runs on the same developer machine or CI runner class before drawing
performance conclusions.

## Current baseline snapshot

Baseline captured from:

```bash
python -m benchmarks.pit --iterations 5 --periods 40 --series-count 3 --revisions-per-period 2
```

| Captured at (UTC) | Rows | `snapshot_ref(...)` median | `build_snapshot_panel_long(...)` median |
| --- | ---: | ---: | ---: |
| `2026-04-05T03:35:01.847144+00:00` | 240 | 5.724 ms | 11.518 ms |

## Migration and deprecation workflow

Before removing a compatibility surface or tightening a public contract:

1. run the targeted subsystem tests
2. run `python -m pytest tests/contracts`
3. rerun `python -m benchmarks.pit` if the change touches PIT retrieval paths
4. update `doc/plan/migration_note.md` with the downstream impact
5. update `doc/plan/active__post-migration-plan.md` if a temporary bridge was added

That discipline keeps migration work anchored to explicit checks instead of
repo-local assumptions.
