![matchers: conflation for Overture Maps data, in PySpark](assets/hero.svg)

[![Test Python](https://github.com/OvertureMaps/matchers/actions/workflows/test_python.yml/badge.svg?branch=main)](https://github.com/OvertureMaps/matchers/actions/workflows/test_python.yml)
[![Lint](https://github.com/OvertureMaps/matchers/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/OvertureMaps/matchers/actions/workflows/lint.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-4051CC?labelColor=001A39)](../.python-version)
[![Spark 3.5](https://img.shields.io/badge/Spark-3.5-4051CC?labelColor=001A39)](../packages/overture_matchers/pyproject.toml)
[![uv](https://img.shields.io/badge/managed%20with-uv-0EC1BD?labelColor=001A39)](https://docs.astral.sh/uv/)
[![Ruff](https://img.shields.io/badge/lint-ruff-0EC1BD?labelColor=001A39)](https://docs.astral.sh/ruff/)
[![License: MIT](https://img.shields.io/badge/license-MIT-05A5AF?labelColor=001A39)](../LICENSE)

Python matchers for conflating Overture Maps data: places, divisions and addresses. A matcher compares incoming records against the Overture baseline and assigns each one a GERS ID: the matched baseline record's ID, or a new ID when nothing matches.

```mermaid
%%{init: {"theme": "dark", "flowchart": {"padding": 14}, "themeVariables": {"fontSize": "14px", "fontFamily": "Montserrat, Arial, sans-serif", "mainBkg": "#001A39", "nodeBorder": "#0EC1BD", "lineColor": "#4051CC", "primaryTextColor": "#FFFFFF", "edgeLabelBackground": "#001A39"}}}%%
flowchart LR
    classDef default rx:8,ry:8,stroke-width:0.75px
    C[Incoming records] --> M[Matcher]
    B[Overture baseline] --> M
    M -->|match| E[Baseline GERS ID]
    M -->|no match| N[New GERS ID]
```

Each matcher runs as a PySpark job on AWS Glue 5 (Spark 3.5, Python 3.11). This repo is a port of Overture's Scala matchers, and the package is pre-alpha while the port is in progress. The plan and progress are tracked in [OvertureMaps/matchers#1](https://github.com/OvertureMaps/matchers/issues/1).

## Packages

| Package | Description | PyPI |
| --- | --- | --- |
| [overture_matchers](../packages/overture_matchers) | The places, divisions and addresses matchers. | [![PyPI](https://img.shields.io/pypi/v/overture-matchers?color=4051CC&labelColor=001A39)](https://pypi.org/project/overture-matchers/) |

Each directory under [packages/](../packages) with a `pyproject.toml` and a `tests/` directory is one package, versioned and published to PyPI on its own.

## Design and parity

| Path | What it holds |
| --- | --- |
| [python-migration.md](python-migration.md) | The design for the port: the Scala system in brief, the target structure, library swaps and sequencing. |
| [parity/](../parity) | A harness that compares two matcher runs on the same input and prints where they differ. |

## Releases

Merging to `main` releases any package whose changes warrant a version bump. The PR title's [Conventional Commits](https://www.conventionalcommits.org/) type sets the bump (`feat!:` major, `feat:` minor, any other type patch), and the release is tagged `<package_dir>-v<version>`. See [PACKAGE_VERSIONING.md](PACKAGE_VERSIONING.md) for the flow and the one-time PyPI setup a new package needs.

## Contributing

See [CONTRIBUTING.md](../.github/CONTRIBUTING.md) for development setup, tests and lint.

## Maintainers

This repository uses `MAINTAINERS.md` files to track ownership for [LFX Insights](https://insights.linuxfoundation.org/docs/introduction/maintainers/) ingestion. LFX scans the full repository tree, so these files can live anywhere.

To add a `MAINTAINERS.md` for a module or package, create it in the relevant directory using this format:

```markdown
# MAINTAINERS

| Name     | GitHub Username | Role            | Affiliation |
| -------- | --------------- | --------------- | ----------- |
| Jane Doe | @janedoe        | Lead Maintainer | Org         |
```

## License

[MIT](../LICENSE)
