# Contributing

## Setup

You need [uv](https://docs.astral.sh/uv/) and Java 17, which the tests' local SparkSession needs. `uv` installs Python 3.11 and the dev dependencies:

```bash
uv sync
uvx pre-commit install  # optional: runs ruff on commit
```

If Java 17 is not your default JVM, point `JAVA_HOME` at it, for example `export JAVA_HOME=/opt/homebrew/opt/openjdk@17` on macOS with Homebrew.

## Tests and lint

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

CI runs both on every pull request.

## Conventions

- Every Python file starts with the SPDX MIT header and a docstring paragraph stating its purpose.
- Public functions have Google-style docstrings.
- Commits are signed off for the [DCO](https://developercertificate.org/): `git commit -s`.
