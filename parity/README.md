# Parity harness

Compares matcher runs on the same input, to gate the Scala-to-Python port ([#1](https://github.com/OvertureMaps/matchers/issues/1)). It lives outside the `overture_matchers` package and is decommissioned once parity confirms the migration.

- `compare.py A_RUN B_RUN` prints the gate metrics as JSON: ID agreement (overall and per slice), cluster agreement and edge agreement. `--stage setup|matching` diffs one stage's outputs instead.
- `launch.py` starts N Glue runs cloned from a template run and prints their run folders as JSON.

Both are [uv scripts](https://docs.astral.sh/uv/guides/scripts/) with inline dependencies; `--help` documents every option. Run folders keep Scala's layout, `<run folder>/<stage output>/type=<table>/`, and runs need `outputDebug=true`, which `launch.py` sets. Both sides of a comparison must use the same model version.

CI doesn't run the tests. Run them by hand from the repo root; they use local fixtures and no AWS:

```bash
uv run --with pytest --with duckdb pytest parity/
```

## Workflows

Scala noise floor: two Scala runs, compared with each other.

```bash
AWS_PROFILE=... uv run parity/launch.py --job-name SCALA_JOB --template-run-id jr_... \
    --output-root s3://BUCKET/parity/noise --runs 2 > runs.json
uv run parity/compare.py $(jq -r '.[].run_folder' runs.json) > noise.json
```

Python against Scala: Python runs from a Python template, each compared with a Scala run folder.

```bash
uv run parity/launch.py --job-name PYTHON_JOB --template-run-id jr_... \
    --output-root s3://BUCKET/parity/python --runs 1 > py.json
uv run parity/compare.py SCALA_RUN $(jq -r '.[0].run_folder' py.json)
```

Mixed mode: one Python stage on Scala's earlier outputs. `--seed-from` copies the outputs of the stages before the first one in `stages` into each new run folder.

```bash
uv run parity/launch.py --job-name PYTHON_JOB --template-run-id jr_... \
    --output-root s3://BUCKET/parity/mixed --runs 1 --set stages=matching --seed-from SCALA_RUN > mixed.json
uv run parity/compare.py SCALA_RUN $(jq -r '.[0].run_folder' mixed.json) --stage matching
```

Python's diff against Scala only means something next to the noise floor: report both.
