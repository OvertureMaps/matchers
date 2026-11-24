# overture-matchers

[![PyPI](https://img.shields.io/pypi/v/overture-matchers?color=4051CC&labelColor=001A39)](https://pypi.org/project/overture-matchers/)
[![Python versions](https://img.shields.io/pypi/pyversions/overture-matchers?color=4051CC&labelColor=001A39)](https://pypi.org/project/overture-matchers/)
[![uv](https://img.shields.io/badge/managed%20with-uv-0EC1BD?labelColor=001A39)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/license-MIT-05A5AF?labelColor=001A39)](https://github.com/OvertureMaps/matchers/blob/main/LICENSE)

Python matchers for conflating Overture Maps data: places, divisions, and addresses.

A matcher assigns IDs. It compares incoming records from a data provider against the Overture baseline. A record that matches takes the baseline record's GERS ID, and a record that matches nothing gets a new one.

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

On AWS Glue, `overture_matchers.places.job.PlacesMatchingJob` is the job class. It extends `SparkSedonaJob` from overture-spark, and its `run(params)` takes the job params as a dict.

Locally, each job module runs from the command line on a local Sedona session, with the params in a JSON file:

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

Every job param, the run folder layout and the rerun rules are in [Running a matcher](https://github.com/OvertureMaps/matchers/blob/main/packages/overture_matchers/docs/running.md).

## Modules

<!--
When adding or editing a row: describe the module's scope (what kind of code
belongs here), not an inventory of its current functions. The table should
stay accurate as a module grows. Technical detail goes in docs/, not here.
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

| Job | Feature types | Detail |
| --- | --- | --- |
| `places.job.PlacesMatchingJob` | `place` | [Places job](https://github.com/OvertureMaps/matchers/blob/main/packages/overture_matchers/docs/places.md) |

## Documentation

Technical detail lives in the package's [docs directory](https://github.com/OvertureMaps/matchers/tree/main/packages/overture_matchers/docs), one page per topic. The design for the port is in [python-migration.md](https://github.com/OvertureMaps/matchers/blob/main/docs/python-migration.md).

## Development

```sh
uv sync
uv run pytest
```

Source, issues and the contributing guide are in [OvertureMaps/matchers](https://github.com/OvertureMaps/matchers).