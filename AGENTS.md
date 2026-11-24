# AGENTS.md

Instructions for AI agents working in this repository.

## Repository

This repository is the Python port of Overture's Scala matchers, which assign GERS IDs by matching incoming records against the Overture baseline. The port's design, structure and sequencing are in [docs/python-migration.md](docs/python-migration.md). Read the design doc before porting stages or adding themes.

| Path | Holds |
| --- | --- |
| `packages/overture_matchers` | The published package: `core` is the run plumbing every theme shares, and each theme has its own subpackage with a `job` module. |
| `parity` | A harness that compares two matcher runs on the same input. |
| `docs` | Repo-wide docs: the port design and package versioning. |

Each directory under `packages/` is an independent package with its own `uv.lock`. The repo root is not a package: its `pyproject.toml` pins ruff and holds the lint rules every package extends.

## Commands

Tests need Java 17 for the local Spark session. The first test run downloads about 750 MB of jars. Full setup is in [CONTRIBUTING.md](.github/CONTRIBUTING.md).

```sh
uv sync                                        # root: ruff
cd packages/overture_matchers && uv sync       # package: pytest, pyspark, sedona
uv run pytest                                  # from packages/overture_matchers
uv run ruff check . && uv run ruff format --check .   # from the repo root
```

## Conventions

- Name functions verbosely, with a clear verb. Function names are never shorter or vaguer than their Scala counterparts.
- Keep the Scala matchers' names for job params and output folders, so a Python run and a Scala run stay comparable.
- Start every Python file with the SPDX license and copyright header the existing files carry.
- Leave the `pyspark` and `apache-sedona` pins alone unless the production Spark runtime changes. They match it on purpose.
- Add tests in the package's `tests/` directory. They share one local Sedona session through the `spark` fixture.
- Title pull requests with a Conventional Commits type, such as `fix:` or `docs:`. Merging to `main` releases every package whose shipped files changed, and the type sets the version bump. See [PACKAGE_VERSIONING.md](docs/PACKAGE_VERSIONING.md).

## Update the docs in the same change

Behavior changes to `packages/overture_matchers` update the matching docs in the same pull request. Find the doc from the code touched:

| Code changed | Doc to update |
| --- | --- |
| Job params in `core/config.py`: a new, renamed or dropped param, a default, a required flag, path rewriting | The params table in `packages/overture_matchers/docs/run-reference.md` |
| A stage starts or stops reading a param | That param's row in `docs/run-reference.md` |
| Run folder layout or stage markers in `core/io.py` | The layout block in `docs/run-reference.md`, and `docs/reruns.md` |
| `run_stage` or the rerun guard in `core/stages.py` and `core/config.py` | `docs/reruns.md`, including its diagram |
| A job's stages, outputs, required params or extra Spark packages | That job's page, for example `docs/places.md`, including its stage table and diagram |
| A new job or theme | A new reference page in `docs/`, a row in the README's Jobs table, and rows in `docs/README.md` and the README's Documentation table |
| A new module | A row in the README's Modules table |
| Install or run commands, or the pinned Spark and Sedona versions in `pyproject.toml` | The Install and Run sections of `packages/overture_matchers/README.md` |

Skip the doc edit when no reader-observable behavior changes. Do not add doc edits to show the docs were considered.

## Rules for the package README

`packages/overture_matchers/README.md` is the page PyPI displays.

- Keep it an overview: description, install, run, the Modules and Jobs tables, and links. Detail goes in `packages/overture_matchers/docs/`.
- Use absolute `https://` links only. PyPI does not resolve relative links.
- No Mermaid diagrams and no GitHub admonitions (`> [!NOTE]`). PyPI renders neither.
- Modules rows describe each module's scope, not an inventory of functions, so the rows stay true as modules grow.
- Edits to this file release a patch version on merge, because the file is in the package's release path filters. Edits under `docs/` do not.

## Rules for every doc

- Document only what the code does. Read the source for each claim before writing it, and never describe planned behavior as present.
- Write in the present tense about the code as it is. No history, no roadmap, and no Status sections: the `Development Status` classifier in `pyproject.toml` carries the package's maturity.
- Stay platform agnostic. Describe what the package does on every Spark platform, and leave platform-specific deployment detail to the repositories that deploy the package.
- Give each page in `docs/` one purpose: steps for a task (how-to), facts to look up (reference), or reasoning (explanation). Split pages that start to serve two purposes, and list every page under its group in `docs/README.md`.
- Open each Mermaid flowchart with the theme directive the existing diagrams use, and state its takeaway in the sentence above it.

## Check before finishing

Run from `packages/overture_matchers` after editing the package README. Both built files must report `PASSED`:

```sh
uv build --out-dir /tmp/overture-matchers-dist
uvx twine check /tmp/overture-matchers-dist/*
```

After editing pages in `docs/`, confirm that every relative link resolves to a file, and that no doc links to a renamed page.
