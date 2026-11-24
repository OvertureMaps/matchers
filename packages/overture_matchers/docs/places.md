# Places job

`overture_matchers.places.job.PlacesMatchingJob` matches the `place` feature type. The shared params and run folder layout are in [Running a matcher](running.md).

## Params

On top of the required params, `input` and `baseline` each need a `place` entry, and `modelJar` is required.

The job adds the GraphFrames package to the Spark session, and sets the Spark checkpoint directory to `checkpoints/` in the run folder, which GraphFrames connected components needs.

## Stages

The stages run in order, and each reads the outputs of the ones before it.

| Stage | Does | Outputs |
| --- | --- | --- |
| `setup` | Normalizes candidates and baseline, and attaches history matches when `matchHistory` is true. | `candidates`, `baseline` |
| `matching` | Generates pairs, scores them and clusters the matches. | `matches` |
| `assign_ids` | Picks cluster winners and assigns GERS IDs. | `final`, `fallout` |
| `metrics` | Counts matches and churn. | `metrics` |

`final` holds the records with their IDs, and `fallout` holds the candidates folded into another record.
