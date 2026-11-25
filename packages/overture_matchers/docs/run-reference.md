# Run reference

This page lists the params every job takes and the folder layout every run writes. The commands that start a run are in the [package README](../README.md#run), and the reasoning behind the layout is in [How reruns work](reruns.md).

## Job params

Params keep the Scala matchers' argument names, so Scala argument sets work unchanged. Params outside this table are ignored.

| Param | Required | Meaning |
| --- | --- | --- |
| `outputPath` | Yes | The run folder. Every output, the stage markers and `config.json` sit inside it. |
| `input` | Yes | Incoming records, as a map of feature type to one parquet path or a list of paths. Accepts a dict or a JSON string. |
| `baseline` | Yes | Baseline records, in the same shape as `input`. |
| `provider` | Yes | The data provider of the incoming records. |
| `baselineProvider` | No | Places only. The provider to stamp on a baseline that has no `provider` column, such as a Places release. Required when the baseline has no `provider` column, and an error when it has one. |
| `matchHistory` | No | Whether candidates with history matches skip scoring. Defaults to `true`. Accepts a boolean or the strings `true` and `false`. |
| `modelJar` | Places only | The `places-matcher-model` jar. |
| `blockingResolution` | No | Places only. The quadkey zoom, from 1 to 23, for `blocking_keys` and `signature`. Defaults to 16. |
| `embeddingsCacheTable` | No | Accepted and recorded in `config.json`. No stage reads this param yet. |
| `previousMetricsPath` | No | A previous run's metrics. Places only. The `metrics` stage compares this run's score features against them. When the param is unset, that comparison is skipped. |

`s3://` paths in `outputPath`, `input`, `baseline` and `previousMetricsPath` are rewritten to `s3a://`, the scheme that works on every Spark platform with hadoop-aws. Trailing slashes on `outputPath` are dropped.

## Spark conf

Places jobs also need `spark.sedona.libpostal.dataDir` set to the unpacked libpostal v1.0.0 data. It is Spark conf, not a job param, and it must be set when the job launches. Sedona reads it on the workers, which never see a `spark.conf.set` made at runtime. A remote data dir needs Sedona 1.9.0 or later.

## Run folder

Every run writes zstd parquet under `outputPath`, one folder per output and feature type:

```text
<outputPath>/
  config.json                                     the run's settings, the libpostal data dir and library versions
  <output>/type=<feature_type>/                   one stage output
  _completed_stages/<stage>/type=<feature_type>   marker written after a stage's outputs
  checkpoints/                                    Spark checkpoint directory
  places_matcher_model/                           places only: the model's reference dataset, copied from the model jar
```

The output folder names match the Scala matchers' layout. Each job's page lists the job's stages and outputs.