# Contributing

See the [Overture contributing guidelines](https://github.com/OvertureMaps/.github/blob/main/CONTRIBUTING.md) for the code of conduct and PR checks.

## Layout

Each directory under [`packages/`](packages) with a `pyproject.toml` and a `tests/` directory is an independent, publishable package with its own `uv.lock` and `.python-version`. CI tests only the packages a change touches. The repo root is not a package: its `pyproject.toml` pins ruff, and `ruff.toml` holds the lint rules every package extends.

## Setup

You need [uv](https://docs.astral.sh/uv/) and Java 17, which the tests' local SparkSession needs. `uv` installs Python 3.11 and the dev dependencies:

```bash
uv sync                                         # root: ruff
(cd packages/overture_matchers && uv sync)      # package: pytest, pyspark
uvx pre-commit install                          # optional: runs ruff on commit
```

If Java 17 is not your default JVM, point `JAVA_HOME` at it, for example `export JAVA_HOME=/opt/homebrew/opt/openjdk@17` on macOS with Homebrew.

## Tests and lint

```bash
(cd packages/overture_matchers && uv run pytest)
uv run ruff check . && uv run ruff format --check .
```

CI lints every pull request and tests each package the pull request changes.

## Releases

Each package releases on its own. Squash-merge PR titles follow [Conventional Commits](https://www.conventionalcommits.org/), and on push to `main` they decide each touched package's next version (`fix:` patch, `feat:` minor, `!` major). There is no version field to edit. See [PACKAGE_VERSIONING.md](PACKAGE_VERSIONING.md).

## Conventions

- Every Python file starts with the SPDX MIT header and a docstring paragraph stating its purpose.
- Public functions have Google-style docstrings; ruff enforces this outside `tests/`.
- Commits are signed off for the [DCO](https://developercertificate.org/): `git commit -s`.
