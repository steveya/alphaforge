# Data Sources Guide

Alphaforge data access is mediated by `DataContext`, which wires source names to source objects.

## Source categories

- Local/in-memory sources for tests and prototyping
- Local configurable futures sources such as `alphaforge.futures.FirstRateFuturesLoader`
- Public web source pack under `alphaforge.data.public_web`
- FRED-style macro sources

## Query contract

Most source fetches are driven by `alphaforge.data.query.Query`, including:

- `table`
- `columns`
- `start` / `end`
- `entities`
- `asof`
- `grid`

## Registries

Some public web sources are configured through YAML registries in `alphaforge/data/registries`.

## Practical recommendation

For production pipelines, keep source instantiation and registry/version pins explicit in one bootstrap module so dataset builds remain reproducible over time.

## Local futures artifacts

The First Rate futures integration is configured through explicit paths, YAML config,
or environment variables. The loader ingests a flat raw directory of
`*_5min.txt` contract files, writes canonical parquet artifacts under a separate
artifact root, and exposes those artifacts through a `SourceAdapter`.

See [First Rate futures guide](first-rate-futures.md) for the expected folder
structure, environment variables, and dataset names.
