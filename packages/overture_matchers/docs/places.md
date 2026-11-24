# Places job

`overture_matchers.places.job.PlacesMatchingJob` matches the `place` feature type. The shared params and run folder layout are in [Running a matcher](running.md).

## Params

On top of the required params, `input` and `baseline` each need a `place` entry, and `modelJar` is required.

The job adds the GraphFrames package to the Spark session, and sets the Spark checkpoint directory to `checkpoints/` in the run folder, which GraphFrames connected components needs.

## Stages

The stages run in order. Each stage's outputs feed the next one, and later stages also read the outputs of the earlier ones.

```mermaid
%%{init: {"theme": "dark", "flowchart": {"padding": 14}, "themeVariables": {"fontSize": "14px", "fontFamily": "Montserrat, Arial, sans-serif", "mainBkg": "#001A39", "nodeBorder": "#0EC1BD", "lineColor": "#4051CC", "primaryTextColor": "#FFFFFF", "edgeLabelBackground": "#001A39"}}}%%
flowchart LR
    classDef default rx:8,ry:8,stroke-width:0.75px
    I[input and baseline] --> S[setup]
    S -->|candidates, baseline| M[matching]
    M -->|matches| A[assign_ids]
    A -->|final, fallout| X[metrics]
```

| Stage | Does | Outputs |
| --- | --- | --- |
| `setup` | Normalizes candidates and baseline, and attaches history matches when `matchHistory` is true. | `candidates`, `baseline` |
| `matching` | Generates pairs, scores them and clusters the matches. | `matches` |
| `assign_ids` | Picks cluster winners and assigns GERS IDs. | `final`, `fallout` |
| `metrics` | Counts matches and churn. | `metrics` |

`final` holds the records with their IDs, and `fallout` holds the candidates folded into another record.
