# Parity harness

Compares two matcher runs on the same input and prints where they differ, for people evaluating the Scala-to-Python port ([#1](https://github.com/OvertureMaps/matchers/issues/1)). It lives outside the `overture_matchers` package and is deleted once the migration is confirmed.

`compare.py A_RUN B_RUN` prints JSON for a person to read, with no pass or fail verdict:

- `ids`: the share of candidate records that get the same ID in both runs, overall and broken down by country and by match type. New IDs are random, so two new IDs count as the same when they cover the same records.
- `groups`: pairs of records grouped together in each run, and the share of each run's pairs the other run also has.

It is a [uv script](https://docs.astral.sh/uv/guides/scripts/) with inline dependencies. Run folders keep Scala's layout, `<run folder>/<stage output>/type=<table>/`, and runs need `outputDebug=true`. Both runs must use the same model version.

CI doesn't run the tests. Run them by hand from the repo root; they use local fixtures and no AWS:

```bash
uv run --with pytest --with duckdb pytest parity/
```

## Example commands

Placeholders are in capitals. Start a Scala run; its run folder is `<outputPath>/<runId>`. A Python run has no `runId`; its `outputPath` is the run folder.

```bash
aws glue start-job-run --job-name SCALA_JOB \
    --arguments '{"--runId":"noise-1","--outputPath":"s3://BUCKET/parity","--outputDebug":"true"}'
```

Scala noise floor: start two Scala runs on the same input (`noise-1`, `noise-2`), then compare them. Python's differences from Scala only mean something next to this.

```bash
uv run parity/compare.py s3://BUCKET/parity/noise-1 s3://BUCKET/parity/noise-2
```

One stage at a time: copy a Scala run's earlier stage outputs into a new run folder, run only the next stage there, and compare.

```bash
for f in candidates baseline embeddings; do
    aws s3 cp --recursive s3://BUCKET/parity/noise-1/$f s3://BUCKET/parity/py-1/$f
done
aws glue start-job-run --job-name PYTHON_JOB \
    --arguments '{"--outputPath":"s3://BUCKET/parity/py-1","--stages":"matching,assignid","--outputDebug":"true"}'
uv run parity/compare.py s3://BUCKET/parity/noise-1 s3://BUCKET/parity/py-1
```
