# overture-matchers

[![PyPI](https://img.shields.io/pypi/v/overture-matchers?color=4051CC&labelColor=001A39)](https://pypi.org/project/overture-matchers/)
[![Python versions](https://img.shields.io/pypi/pyversions/overture-matchers?color=4051CC&labelColor=001A39)](https://pypi.org/project/overture-matchers/)
[![uv](https://img.shields.io/badge/managed%20with-uv-0EC1BD?labelColor=001A39)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/license-MIT-05A5AF?labelColor=001A39)](https://github.com/OvertureMaps/matchers/blob/main/LICENSE)

Python matchers for conflating Overture Maps data: places, divisions, and addresses.

Matchers assign IDs. Each matcher compares incoming records from a data provider against the Overture baseline. Matching records take the baseline record's GERS ID, and unmatched records get new IDs.

## Install

```sh
uv add overture-matchers
```

PySpark and Sedona are not runtime dependencies, because Spark platforms provide both. Other environments, such as local runs, need PySpark and Sedona installed alongside the package. The package's tests run against these versions:

```sh
uv add overture-matchers 'pyspark[sql]==3.5.4' apache-sedona==1.8.1
```

Local Spark sessions also need Java 17.

## Run

On managed Spark platforms, `overture_matchers.places.job.PlacesMatchingJob` is the job class. The class extends `SparkSedonaJob` from overture-spark, and the class's `run(params)` method takes the job params as a dict.

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

Every job param and the run folder layout are in the [run reference](https://github.com/OvertureMaps/matchers/blob/main/packages/overture_matchers/docs/run-reference.md).

## Modules

<!--
When adding or editing a row: describe the module's scope (what kind of code
belongs here), not an inventory of current functions. The table should
stay accurate as modules grow. Technical detail goes in docs/, not here.
-->

| Module | Scope |
| --- | --- |
| `core.config` | `RunConfig`, the settings for one run: parsing and validating job params, and guarding run folders against reruns with different params. |
| `core.stages` | The typed outputs each stage returns, and `run_stage`, which runs each stage, writes the stage's outputs, marks the stage done and skips finished stages on reruns. |
| `core.io` | Reads and writes inside the run folder: output paths, stage markers and lineage-cutting checkpoints. |
| `core.local` | Running a job on a local Sedona session from a JSON params file. |
| `ids` | Assigning GERS IDs to the selected matches. |
| `metrics` | Run metrics every theme shares, and turning metrics into output rows. |
| `places` | The places matcher. |

`core` is the run plumbing every theme shares. Each theme has its own subpackage with a `job` module.

## Jobs

| Job | Feature types | Detail |
| --- | --- | --- |
| `places.job.PlacesMatchingJob` | `place` | [Places job](https://github.com/OvertureMaps/matchers/blob/main/packages/overture_matchers/docs/places.md) |

## Documentation

Technical detail lives in the package's [docs directory](https://github.com/OvertureMaps/matchers/tree/main/packages/overture_matchers/docs), grouped by what the reader needs:

| Need | Page |
| --- | --- |
| Look up a param or the run folder layout | [Run reference](https://github.com/OvertureMaps/matchers/blob/main/packages/overture_matchers/docs/run-reference.md) |
| Look up a job's stages and outputs | [Places job](https://github.com/OvertureMaps/matchers/blob/main/packages/overture_matchers/docs/places.md) |
| Understand how runs resume | [How reruns work](https://github.com/OvertureMaps/matchers/blob/main/packages/overture_matchers/docs/reruns.md) |
| Understand the design of the port | [python-migration.md](https://github.com/OvertureMaps/matchers/blob/main/docs/python-migration.md) |

## Development

```sh
uv sync
uv run pytest
```

Source, issues and the contributing guide are in [OvertureMaps/matchers](https://github.com/OvertureMaps/matchers).