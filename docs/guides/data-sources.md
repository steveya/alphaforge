# Data Sources Guide

Alphaforge data access is mediated by `DataContext`, which wires source names to source objects.

## Source categories

- Local/in-memory sources for tests and prototyping
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
