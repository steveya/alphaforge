# Public Web Source Authoring

Use `alphaforge.data.public_web` for public datasets that should expose a
stable `Query`-driven interface but do not need a separate adapter package.

The refactor goal is modest: centralize fetch/finalize boilerplate while
keeping provider-specific normalization logic explicit inside each source
module.

## Choose the shallowest helper family

| Loader shape | Use when | Preferred helpers | Current examples |
| --- | --- | --- | --- |
| Registry-backed API | entity definitions come from YAML registry metadata | `PublicWebSourceBase`, `RegistryApiSourceBase`, `schema_helpers.py` | `bcb_sgs.py`, `bea.py`, `bls.py`, `destatis_genesis.py`, `ecb_sdmx.py`, `eia.py`, `eurostat.py`, `ibge_sidra.py` |
| Tabular document | provider publishes HTML, XLSX, or CSV documents with recurring table cleanup patterns | `PublicWebSourceBase`, `TabularDocumentSourceBase`, `tabular.py`, `schema_helpers.py` | `cme_productslate_reference.py`, `ec_weekly_oil_bulletin.py`, `eurex_refdata_contracts.py`, `eurex_stats_daily.py`, `ezoic_adrevenue_daily.py`, `frb_term_structure.py`, `lch_cdsclear_daily.py` |
| Archive or batch source | data is distributed as yearly ZIPs or historical archive bundles | `PublicWebSourceBase`, `archive.py`, `schema_helpers.py` | `b3_historical_quotes.py`, `cftc_cot.py`, `cftc_swaps_weekly.py` |
| True outlier | workflow is too source-specific for a family helper | `PublicWebSourceBase` only | `dtcc_ppd.py`, `mof_jgb.py`, `philadelphia_spf.py` |

If a source only shares HTTP setup and finalization, stop at
`PublicWebSourceBase`. Do not force it into a deeper family abstraction.

## Required implementation checklist

1. Read the target source, its matching test file, and any shared helper
   modules it already depends on.
2. Preserve table names, schema contracts, entity-id shapes, sorting, and
   `asof_utc` semantics unless the ticket explicitly changes them.
3. Define schemas with `table_schema()`, `daily_panel_schema()`,
   `single_value_schema()`, or `event_table_schema()` rather than open-coding
   `TableSchema` where the helper fits.
4. Use `_empty_frame()` and `_finalize()` so projection, entity filtering,
   date filtering, and `asof_utc` handling stay consistent.
5. Update `alphaforge/data/public_web/registry.py` and
   `alphaforge/data/public_web/__init__.py` when adding a new default source.
6. If the source is intended to be available from the package root, also update
   `alphaforge/__init__.py`.
7. Add or update the matching tests in `tests/public_web/` and adapter tests in
   `tests/` when higher-level routing changes.
8. Update docs after the implementation is stable:
   `docs/api/`, `docs/getting-started/`, `docs/guides/`, and the mirrored plan
   file in `doc/plan/` when ticket state or scope changed.

## Defensive parsing rules

- Expect provider archives to have header drift, renamed files, and historical
  batch exceptions.
- Prefer `discover_archive_fetches(...)` or `iter_yearly_archive_fetches(...)`
  over open-coded URL lists when a source repeatedly downloads archive files.
  They keep deterministic artifact names and handle query-string download links
  more robustly than ad hoc string filtering.
- Normalize date columns through shared helpers such as `ensure_date_utc()`,
  `ensure_utc()`, `resolved_date_series()`, and `_asof_utc()`.
- Treat empty upstream payloads as a valid case and return schema-correct empty
  frames instead of ad hoc partial frames.
- Keep artifact naming deterministic when caching HTTP responses.
- Do not invent entity ids, table names, or aliases. Verify them from the
  provider payload, registry, or existing tests.

## Validation

Targeted source tests should fail first for the intended reason, then pass after
the implementation lands.

Recommended validation commands:

```bash
/Users/steveyang/miniforge3/bin/python -m pytest tests/public_web/test_<source>.py -q
/Users/steveyang/miniforge3/bin/python -m pytest tests/public_web -k 'not live_sources' -q
/Users/steveyang/miniforge3/bin/python -m ruff check .
/Users/steveyang/miniforge3/bin/python -m mkdocs build --strict
```

Run adapter regressions in `tests/` when the source is reachable through
`alphaforge.data.sources` or a PIT transform.

## Registry and discovery

- `default_public_web_sources()` is the default constructor registry for the
  public-web pack.
- `tests/public_web/test_source_test_mapping.py` enforces that every concrete
  source module has a matching test module.
- `tests/public_web/test_live_sources.py` is opt-in and should stay resilient
  to provider outages and empty live responses.

## Coordination

Follow the implementation workflow in `AGENTS.md` for Linear-driven work:
review upstream tickets first, announce the active ticket on screen, implement
with TDD, update docs, leave a Linear close-out note, mark the ticket done, and
only then sync the mirrored plan table.
