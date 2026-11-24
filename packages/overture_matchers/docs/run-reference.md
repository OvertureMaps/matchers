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
| `matchHistory` | No | Whether candidates with history matches skip scoring. Defaults to `true`. Accepts a boolean or the strings `true` and `false`. |
| `modelJar` | Places only | The `places-matcher-model` jar. |
| `embeddingsCacheTable` | No | Accepted and recorded in `config.json`. No stage reads this param yet. |
| `previousMetricsPath` | No | Accepted and recorded in `config.json`. No stage reads this param yet. |

`s3://` paths in `outputPath`, `input`, `baseline` and `previousMetricsPath` are rewritten to `s3a://`, the scheme that works on every Spark platform with hadoop-aws. Trailing slashes on `outputPath` are dropped.

## Run folder

Every run writes zstd parquet under `outputPath`, one folder per output and feature type:

```text
<outputPath>/
  config.json                                     the run's settings and the package version
  <output>/type=<feature_type>/                   one stage output
  _completed_stages/<stage>/type=<feature_type>   marker written after a stage's outputs
  checkpoints/                                    Spark checkpoint directory
```

The output folder names match the Scala matchers' layout. Each job's page lists the job's stages and outputs.