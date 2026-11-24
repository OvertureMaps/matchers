# overture-matchers

[![PyPI](https://img.shields.io/pypi/v/overture-matchers?color=4051CC&labelColor=001A39)](https://pypi.org/project/overture-matchers/)
[![Python versions](https://img.shields.io/pypi/pyversions/overture-matchers?color=4051CC&labelColor=001A39)](https://pypi.org/project/overture-matchers/)
[![uv](https://img.shields.io/badge/managed%20with-uv-0EC1BD?labelColor=001A39)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/license-MIT-05A5AF?labelColor=001A39)](https://github.com/OvertureMaps/matchers/blob/main/LICENSE)

Python matchers for conflating Overture Maps data: places, divisions, and addresses.

A matcher assigns IDs. It compares incoming records from a data provider against the Overture baseline. A record that matches takes the baseline record's GERS ID, and a record that matches nothing gets a new one.

Each matcher runs as a PySpark job on AWS Glue 5.0 (Spark 3.5, Python 3.11), built on the job classes in [overture-spark](https://pypi.org/project/overture-spark/).

## Status

The package is pre-alpha. It is a port of Overture's Scala matchers, and the port is in progress. The package holds the shared run framework and the places job's wiring. The four places stages raise `NotImplementedError`, and the divisions and addresses matchers are not in the package yet. Progress is tracked in [OvertureMaps/matchers#1](https://github.com/OvertureMaps/matchers/issues/1).

## Install

```sh
uv add overture-matchers
```

PySpark and Sedona are not runtime dependencies, because the Glue runtime provides both. A run outside Glue needs them installed alongside the package, at the versions Glue 5.0 pairs with:

```sh
uv add overture-matchers 'pyspark[sql]==3.5.4' apache-sedona==1.8.1
```

A local Spark session also needs Java 17.

## Run

### On AWS Glue

`overture_matchers.places.job.PlacesMatchingJob` is the job class. It extends `SparkSedonaJob` from overture-spark, and its `run(params)` takes the [job params](#job-params) as a dict.

### Locally

Each job module runs from the command line on a local Sedona session, with the params in a JSON file:

```sh
python -m overture_matchers.places.job params.json
```

```json
{
  "outputPath": "/tmp/matcher-runs/2026-01-01",
  "input": {"place": ["/data/candidates/"]},
  "baseline": {"place": ["/data/baseline/"]},
  "provider": "acme",
  "modelJar": "/data/places-matcher-model.jar"
}
```

The first local run downloads the Sedona, GraphFrames and hadoop-aws jars from Maven into `~/.ivy2` (about 750 MB).

### Job params

Params keep the Scala matchers' argument names, so an argument set from a Scala run works unchanged. Params outside this table are ignored.

| Param | Required | Meaning |
| --- | --- | --- |
| `outputPath` | Yes | The run folder. Every output, the stage markers and `config.json` sit inside it. |
| `input` | Yes | Incoming records, as a map of feature type to one parquet path or a list of them. A dict or a JSON string. |
| `baseline` | Yes | Baseline records, in the same shape as `input`. |
| `provider` | Yes | The data provider of the incoming records. |
| `matchHistory` | No | Whether candidates with history matches skip scoring. Defaults to `true`. |
| `modelJar` | Places only | The `places-matcher-model` jar. |
| `embeddingsCacheTable` | No | Accepted and recorded in `config.json`. No stage reads it yet. |
| `previousMetricsPath` | No | Accepted and recorded in `config.json`. No stage reads it yet. |

An `s3://` path is rewritten to `s3a://`, which works on any Spark platform with hadoop-aws.

### Run folder

A run writes zstd parquet under `outputPath`, one folder per output and feature type:

```text
<outputPath>/
  config.json                               the run's settings and the package version
  <output>/type=<feature_type>/             one stage output
  _completed_stages/<stage>/type=<feature_type>   marker written after a stage's outputs
  checkpoints/                              Spark checkpoint directory
```

A rerun on the same folder skips every stage that has a marker and reads that stage's outputs instead. A marker is written only after all of its stage's outputs, so an interrupted stage runs again.

A rerun with changed params fails once any stage has finished, because the finished stages ran with the old params. Start a new run folder instead. Before any stage finishes, the new `config.json` replaces the old one.

## Modules

<!--
When adding or editing a row: describe the module's scope (what kind of code
belongs here), not an inventory of its current functions. The table should
stay accurate as a module grows.
-->

| Module | Scope |
| --- | --- |
| `core.config` | `RunConfig`, the settings for one run: parsing and validating job params, and guarding a run folder against a rerun with different params. |
| `core.stages` | The typed outputs each stage returns, and `run_stage`, which runs a stage, writes its outputs, marks it done and skips it on a rerun. |
| `core.io` | Reads and writes inside the run folder: output paths, stage markers and lineage-cutting checkpoints. |
| `core.local` | Running a job on a local Sedona session from a JSON params file. |
| `places` | The places matcher. |

`core` is the run plumbing every theme shares. Each theme has its own subpackage with a `job` module.

## Jobs

### `places.job.PlacesMatchingJob`

Matches the `place` feature type. `input` and `baseline` each need a `place` entry, and `modelJar` is required. The job runs four stages in order:

| Stage | Does | Outputs |
| --- | --- | --- |
| `setup` | Normalizes candidates and baseline, and attaches history matches when `matchHistory` is true. | `candidates`, `baseline` |
| `matching` | Generates pairs, scores them and clusters the matches. | `matches` |
| `assign_ids` | Picks cluster winners and assigns GERS IDs. | `final`, `fallout` |
| `metrics` | Counts matches and churn. | `metrics` |

`final` holds the records with their IDs, and `fallout` holds the candidates folded into another record.

## Development

```sh
uv sync
uv run pytest
```

Source, issues and the contributing guide are in [OvertureMaps/matchers](https://github.com/OvertureMaps/matchers). The design for the port is in [python-migration.md](https://github.com/OvertureMaps/matchers/blob/main/docs/python-migration.md).