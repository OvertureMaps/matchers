# Places job

`overture_matchers.places.job.PlacesMatchingJob` matches the `place` feature type. The shared params and run folder layout are in the [run reference](run-reference.md).

## Params

On top of the required params, `input` and `baseline` each need a `place` entry, and `modelJar` is required.

The job needs GraphFrames on the Spark session. Local runs add GraphFrames from the Maven coordinates in `job.PACKAGES`. Runs on managed platforms do not, so the deployments have to supply the package.

GraphFrames connected components needs a Spark checkpoint directory, so the job sets the directory to `checkpoints/` in the run folder.

## Stages

The stages run in order. Each stage's outputs feed the next stage, and later stages also read the outputs of earlier stages.

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
| `assign_ids` | Picks cluster winners and assigns GERS IDs. | `final`, `fallout`, `idMap` |
| `metrics` | Counts matches and churn. | `metrics` |

`final` holds the records with their IDs, and `fallout` holds the candidates folded into another record. `idMap` maps each winning candidate's `id` to the ID it was assigned.

## Optional metrics inputs

The `metrics` stage also reads these inputs when a run has them. When one is missing or fails to load, the stage logs a warning and skips the metrics that need it.

| Input | Read from | Feeds |
| --- | --- | --- |
| `blocking` | `blocking/` in the run folder | Blocking metrics, and the relative match rate |
| `scores` | `scores/` in the run folder | Score metrics, and a sample of about 200 MB for the feature stats |
| Feature order | `modelJar` | Feature stats |
| Reference dataset | `modelJar`, copied to `places_matcher_model/` in the run folder | Feature drift against the model's reference data |
| Previous metrics | `previousMetricsPath` | Feature drift against a previous run |
