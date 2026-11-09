# Python migration plan

This document outlines the design for porting the matchers (places, divisions, addresses) from OvertureMaps/overture-matchers (Scala) to Python/PySpark in this repo. The evaluation behind it, with benchmarks, is in OvertureMaps/overture-matchers#455.

This final major section of this document is titled "Details for agents" -- as the name implies, it contains lower-level implementation details that I didn't think were important enough to include in the following design notes.

## Goal and constraints

- To port the pipeline to Python/PySpark -- and take the opportunity to clean up the code a little.
- The matching's final output should stay as close to Scala's as is reasonably possible. This can be verified by running the parity harness (see Parity below). Exact parity is not required (but it would be nice).
- Production runs on AWS Glue 5 (Spark 3.5, Python 3.11), launched by the places and divisions match DAGs in tf-data-platform.

## Current system (Scala), in brief

This is just a summary of how the Scala matching works today:

- `Main` runs four stages chosen by `--stages`: setup, matching, assignid, metrics. Stages hand off only through folders under the run's S3 output path (`candidates/`, `baseline/`, `matches/`, `final/`, `fallout/`, `metrics/`). With `outputDebug` on, matching also writes `blocking/`, `scores/` and `matches_validated/`, and assignid writes `idMap/` (each candidate's `id` and `assigned_id`). The parity harness reads `idMap/`.
- Each stage calls the slots of a `Theme` (Transformer, HistoryFilter, EmbeddingsGenerator, Matcher, IdAssignmentPipeline). The config JSON in `resources/configs/` picks each slot's class.
- Matchers form an inheritance chain: `PlacesUnionedMatcher extends PlacesMatcher extends Matcher`, and each level overrides most of the one above.
  - `PlacesUnionedMatcher` (production places) unions candidates and baseline, spatially self-joins within 400 m (Sedona), prunes with an XGBoost blocking model (p ≥ 0.08), scores with a 28-feature XGBoost model (p ≥ 0.48), and clusters with GraphFrames connected components. Candidates with history matches skip blocking and scoring. Their history pairs join the scored pairs after the model, with `composite_score` and `prediction` set to 1.
  - `PlacesMatcher` is used only by the places-signals config: quadkey tile blocking with a name filter, candidate-to-baseline pairs only, best baseline per candidate. It is stale (OvertureMaps/overture-matchers#431).
  - `DivisionsMatcher`: quadkey blocking per subtype, rule-based scores in `Matcher.scala`, per-subtype thresholds, greedy restricted-duplicates selection.
  - `AddressesMatcher`: tier 1 exact signature match, tier 2 H3 res-14 blocking, street and house-number pre-score with a top-50 cut, weighted composite (max of two weightings), threshold 0.95, best baseline per candidate.
- New IDs are UUIDv4 for places and divisions, and H3-based for addresses and places-signals (`idGenerator` in each config).
- The model artifact `places-matcher-model` (from tf-places-matcher-model) is a Maven jar with the XGBoost models, feature order, thresholds, count-min sketch and Bloom filter binaries, and the Java `PlaceTextNormalizer`. The Scala matcher pins its version in `pom.xml` and shades it into the matcher jar, so every model release needs a matcher release.

Reference volumes from a recent production places run (74 min on 31 Glue workers): 140.2M baseline, 61.5M candidates, 57.5M history matches, about 42B spatial pairs, about 170M pairs surviving blocking, 110.7M match rows, 59.7M final records.

## Target structure

At a high level, I'd like to remove the class structure entirely:

* Matching is still a single `SparkSedonaJob`.
* That job is a chain of *stages*
  * e.g. setup -> matching -> assign_ids -> metrics
  * Each stage writes to its own subdirectory within the run's `MatchBundle` (a new kind of `Bundle`, we were kinda already doing this, but now it's official) using Scala's flat folder layout. I did consider using separate bundles, but matching is typically run all-at-once, and the metadata.json gets needlessly messy if everything is in its own bundle. I think that, in general, 1 DAG run ~ 1 Bundle (lmk if you disagree)
  * The job writes directly to the bundle's output path (`runId` is not an argument).
  * To track stage success, the bundle has a `_completed_stages` subdirectory that contains success files at `_completed_stages/<stage>` that are written when `<stage>` has finished running. This is how reruns can pick up again partway through the job.
* Stages are pure functions, too, mapping a `NamedTuple` of dataframes to another `NamedTuple` of dataframes. The orchestration layer sits above this and cand decide whether the functions are just chained together (for local tests) or written to / read from S3 between stages (for runs at scale).
* Stages are composed of *steps*:
  * Each step is a `DataFrame`->`DataFrame` transformation.
  * Steps chain together with `df.transform` when possible, e.g. `df.transform(spatial_pairs, radius_m=400)`
* `RunConfig` is still serialized into JSON, but it lives in-memory as a frozen dataclass.
  * The `RunConfig` object is built from the job params when the job starts, before any stage runs, then written to JSON at `s3://.../<run>/config.json`.
  * `config.json` is also stamped into the bundle's `metadata.json` before finalization -- this is done by the Airflow DAG.

### Pseudocode stack trace

I think this is the simplest way to explain things.

The DAG has a task group that looks something like:
```python
# MatchBundle does not exist yet, that will be part of the tf-data-platform side.
# Helper names are placeholders for tf-data-platform's own.
bundle = MatchBundle(theme="places", provider=..., run=run_id)

with TaskGroup("places_matching"):
    # Candidate and baseline {table: [paths]} maps, known only at run time (XCom)
    inputs = resolve_inputs()
    init = initialize_bundle(bundle, inputs)

    job = spark_job(
        module_name="overture_matchers.places.job",
        class_name="PlacesMatchingJob",
        packages="overture-matchers[places]==<version> <model package>==<version>",
        jars=[...],  # GraphFrames, PlaceTextNormalizer
        sedona_version="1.8.1",
        worker_cores=980,
        parameters={
            "outputPath": bundle.base_uri,
            "input": ...,  # from inputs
            "baseline": ...,
            ...
        },
    )

    add_config = add_config_to_metadata(bundle)  # config.json -> metadata.json

    chain(inputs, init, job, add_config, finalize_bundle(bundle))
```

Inside the job, each stage is just a function : `DataFrames -> NamedTuple of DataFrames`. Bedause stages don't do any I/O themselves, you can run them ad-hoc in a notebook or a unit test.

Stage names / marker names are just taken straight from the stage function's `__name__`, and the attributes are different DataFrames corresponding to output folders in the run directory. I think it's probably fine to put these classes up in `core/stages.py`, since I anticipate that most themes will reuse them. If a theme needs something else, eventually, we can just add a theme-specific tuple.

```python
# core/stages.py


class RawInputs(NamedTuple):
    candidates: DataFrame
    baseline: DataFrame


class SetupOutput(NamedTuple):
    candidates: DataFrame
    baseline: DataFrame


class MatchingOutput(NamedTuple):
    matches: DataFrame


class AssignIdsOutput(NamedTuple):
    final: DataFrame
    fallout: DataFrame


class MetricsOutput(NamedTuple):
    metrics: DataFrame
```

These classes are the input/return types of the functions for each stage. Due to the complexities of our current pipelines, those functions may accept additional arguments, too:

```python
"""Places matching: setup -> matching -> assign_ids -> metrics."""


def setup(raw: RawInputs) -> SetupOutput: ...  # normalize, ParseAddress, signatures
def matching(
    prepared: SetupOutput,
    model: PlacesModel,
    embeddings: DataFrame | None = None,
    match_history: bool = False,
    output_debug: bool = False,
    checkpoint: Checkpoint = no_checkpoint,
) -> MatchingOutput: ...  # pairs, scoring, clusters
def assign_ids(
    raw: RawInputs, prepared: SetupOutput, matched: MatchingOutput
) -> AssignIdsOutput: ...  # cluster winners, GERS IDs
def metrics(
    raw, prepared, matched, ids, previous
) -> MetricsOutput: ...  # counts, churn
```

Then, after each stage is defined (for a specific theme), that theme's `run()` wires them together. It starts by prepping all the pipeline inputs (input data, model, embeddings, previous metrics), and then chains together a bunch of `run_stage` calls:

```python
def run(spark, config):
    raw = RawInputs(
        read_tables(spark, config.input), read_tables(spark, config.baseline)
    )
    model = PlacesModel.load()

    prepared = run_stage(spark, config, setup, raw)
    matched = run_stage(
        spark,
        config,
        matching,
        prepared,
        model,
        embeddings=read_embeddings(spark, config),  # None when no table is set
        match_history=config.match_history,
        output_debug=config.output_debug,
        checkpoint=io.checkpointer(config.output_path),
    )
    ids = run_stage(spark, config, assign_ids, raw, prepared, matched)
    run_stage(
        spark,
        config,
        metrics,
        raw,
        prepared,
        matched,
        ids,
        read_previous_metrics(spark, config),
    )


class PlacesMatchingJob(SparkSedonaJob):
    def execute_job(self):
        run(self.spark, RunConfig.for_run(self.params))


if __name__ == "__main__":
    run_local(run, sys.argv[1])
```

`run_stage` is shared by all themes in `core/`; it does a few things:
* It skips any stage that has a marker present
* It calls the stage function with all the necessary arguments
* It writes the stage output to S3 along with a success marker in `completed_stages/<stage>`
* It "reads" back the written outputs and returns the `DataFrame`s. Parquet reads are lazily eval'd by Spark, so this shouldn't incur any costs on the final stage.

It also uses `ParamSpec` / `TypeVar` from Python 3.10+ for better type hinting.

```python
Checkpoint = Callable[[DataFrame, str], DataFrame]
P = ParamSpec("P")
T = TypeVar("T")


def no_checkpoint(df, name):
    return df


def run_stage(
    spark, config, stage_fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> T:
    stage = stage_fn.__name__
    output_type = get_type_hints(stage_fn)["return"]
    if not io.stage_is_done(config.output_path, stage):
        outputs = stage_fn(*args, **kwargs)
        for name, df in outputs._asdict().items():
            io.write_output(config.output_path, name, df)
        io.mark_stage_done(
            config.output_path, stage
        )  # writes _completed_stages/<stage>
        spark.catalog.clearCache()  # everything is on S3 between stages
    return output_type(
        *(
            io.read_output(spark, config.output_path, name)
            for name in output_type._fields
        )
    )


def run_local(run, params_file):
    params = json.loads(Path(params_file).read_text())
    run(local_sedona_session(), RunConfig.for_run(params))
```

To explore, you can call a stage function directly, e.g. `matching(SetupOutput(my_candidates, my_baseline), model)`. With the default `no_checkpoint` this doesn't write anything, which is fine for small datasets.

Where the little `io` helper has functions like:

```python
def stage_is_done(output_path, stage_name):
    return exists(f"{output_path}/_completed_stages/{stage_name}")
```

And `RunConfig` is pretty flat; it looks something like:

```python
@dataclass(frozen=True)
class RunConfig:
    provider: str
    input: dict[str, list[str]]  # table -> paths
    baseline: dict[str, list[str]]
    output_path: str  # bundle base
    match_history: bool = False
    output_debug: bool = False
    # ...

    @classmethod
    def for_run(cls, params):
        config = cls.from_params(params)
        path = f"{config.output_path}/config.json"
        record = config.to_json()  # also records installed package versions
        if io.exists(path) and io.read(path) != record:
            raise ValueError(f"{path} was written with different params")
        io.write(path, record)
        return config
```

Meanwhile, the actual transformations look like:

```python
def setup(raw: RawInputs) -> SetupOutput:
    """Normalize candidates and baseline."""
    candidates = (
        raw.candidates.transform(normalize_places)
        .transform(parse_addresses)  # libpostal ParseAddress
        .transform(add_signature)
    )
    baseline = raw.baseline.transform(normalize_places).transform(add_signature)
    return SetupOutput(candidates, baseline)
```

Functions like `normalize_places` and `parse_addresses` make heavy use of shared modules in `overture_matchers`, which hold theme-specific features as well as cross-theme normalization logic, etc.

At some point we should also figure out how this interacts with the model training repository. The Scala matcher didn't really have any "feature version" parity safeguards against the Python training repo (which led to headaches, as we know). This is a great opportunity to fix that, but until we get the MVP going, I'd like to hold off on that sort of workflow refactoring.

### Additional Notes

* **Non-Places Themes** - Themes don't share a fixed sequence of transformations: Addresses keeps only the 50 best pairs per candidate and scores each pair two ways, keeping the higher score. Divisions applies per-subtype thresholds and greedy selection, etc. For this reason, I think a functional approach to matching makes more sense; there's not much motivation to standardize the way matchers work across themes. Anyway, that's pretty much what I suggest here.

* **Config refactoring** - Scala's theme config JSON mostly wires classes by name, which plain functions make unnecessary. The remaining settings fall into two categories, roughly: run settings (`outputDebug`, `matchHistory`), which become fields in `RunConfig`, and theme constants (table name, ID generator) -- which are just constants in the theme package. With no config JSON left, there is nothing for `overrideConfigPath` to override, so the job has no such param. The `{table: [paths]}` maps for `input` and `baseline` also become per-theme params during the cutover (for places, just candidate and baseline paths).

* **Mixed Mode** - During the port, we may want to run the Scala pipeline up to a specific point, then let Python take over, or vice versa. To do this, I think we can just copy the Scala run's outputs into a new run folder, write some stage markers, and let the Python job skip any already-completed stages (see Parity). This should never need to happen in a prod environment, so I think I can hack around it during development.

### Package layout

```
packages/
  overture_matchers/
    overture_matchers/
      core/              # run plumbing only, no matching logic
        config.py        # frozen dataclass RunConfig, built from the job params
        io.py            # read_output/write_output, checkpoint, stage markers, config.json write and check
        stages.py        # stage output types, run_stage, run_local, Checkpoint
      clustering.py      # connected components, greedy, simple
      ids.py             # GERS ID minting
      prematch.py        # overrides and history pre-matches
      places/
        job.py           # stage functions, run(), PlacesMatchingJob
        blocking.py      # pair generation: spatial join, blocking filter
        features.py      # place-pair features
        model.py         # PlacesModel, XGBoost predict
        selection.py     # cluster winner selection
      divisions/
        job.py
        ...
    tests/
```

Other modules are added as code arrives, named for what they do.

The repo follows overture-core's `packages/` layout, starting with one package. Each theme is a subpackage holding its job and its theme-only steps. Modules shared across themes, such as clustering, ID minting and the shared normalization, sit at the top level and never import a theme package. Blocking stays in `places/` for now: the current blocking is theme-specific, and the port doesn't change it. Shared code moves into a second package only when something outside `overture_matchers` needs it (see Open items).

Theme-specific dependencies go in per-theme extras (e.g. `overture-matchers[places]` pulls in `xgboost` and `rapidfuzz`), so each DAG only installs what its job runs. `io.py` builds on overture-core's `hadoop_fs` helpers; anything missing there (`exists`, a write helper) gets added upstream.

Each package is versioned from its own `<package>-v<version>` tags through python-semantic-release and setuptools_scm, and published to public PyPI by trusted publishing, as in overture-core.

## Library swaps

| Scala dependency | Python replacement | Note |
| --- | --- | --- |
| `com.wherobots:jpostal` 1.2.2 (OvertureMaps/overture-matchers#478) | Sedona 1.8.1 `ParseAddress` | Same jpostal build Sedona uses. Set `spark.sedona.libpostal.dataDir` to the v1.0.0 model data the Scala job uses today. Tested 100% identical to the earlier custom build on 449k public addresses. Sedona's default model download changes 18% of non-US addresses; do not use it during the port. |
| Sedona 1.7.2 | Sedona 1.8.1 Python API | tf-data-platform pins one Sedona version for all jobs, and its libpostal jobs already use 1.8.1; regression-check the other Sedona jobs. |
| xgboost4j-spark | `xgboost.Booster().load_model()` on the native booster file, `inplace_predict` in a pandas UDF | Numeric Arrow crossing costs about 0.85 µs/row. |
| GraphFrames (Scala) | `graphframes` Python API | Same jar. |
| libphonenumber | `phonenumbers` |  |
| spark-nlp (Scala) | spark-nlp Python API, pin `spark-nlp==5.5.1` | Assembly jar already attached for divisions. |
| commons-text, stringdistance | Spark built-in `levenshtein()` in blocking; `rapidfuzz` in UDFs |  |
| `PlaceTextNormalizer` (Java) | Keep as a JVM UDF via `registerJavaFunction` during the port. The Java source stays in tf-places-matcher-model, which publishes the jar on its own to CodeArtifact Maven. | Training calls it the same way. Porting it to Python is a later, separately measured step. |
| Spark sketch binaries | Small Python reader for Spark's CountMinSketch and BloomFilter formats | Do not re-export as exact lookup tables: the sketches are approximate and the current features include their errors. |

## Parity

`compare.py` compares two run folders and reports, for people to review:

1. ID agreement: the share of candidates with the same outcome in both runs, overall and broken down by country and by match type. A candidate agrees when both runs reuse the same GERS ID, both mint a new ID for the same group of candidates, or both drop it as a duplicate of the same record (fallout). New IDs are random, so their values are never compared. A candidate missing from either run's `idMap/` and `fallout/` counts as a disagreement.
2. Cluster agreement: of the record pairs one run groups together, the share the other run also groups together, measured in both directions. GraphFrames `cluster_id` values are arbitrary, so only which records share a cluster counts.

Method, set up before any porting:

- A fixed regional test input, large enough to cover several countries and scripts.
- A second pair of runs with `matchHistory` off. In the reference places run, 57.5M of 61.5M candidates are history matched, and those skip blocking and scoring, so a comparison with history matching on mostly checks history matching.
- Pin both sides to the same model. The Scala jar shades in its model version (`places.matcher.model.version` in its `pom.xml`), so a Python-vs-Scala run pins the model package to that version.
- Mixed mode: copy a Scala run's earlier stage outputs into a new run folder at the same paths, write markers for those stages, and run the Python job. It skips the marked stages, so differences in the next stage's output come from that stage alone. A comparison of one stage's outputs gets built when matching is ported, shaped by what that work needs. Mixed mode is a development tool only; production never mixes Scala and Python stages.

The harness lives in a top-level `parity/` directory, outside `packages/`, and is deleted once Scala is archived.

Until cutover, places keeps Scala's four stages and job params keep Scala's argument names. Outputs keep Scala's layout, `<postfix>/type=<table>/`, with Scala's `<postfix>` names and columns, so a Scala run folder and a Python run folder look the same. Python adds only `config.json` and the `_completed_stages/` markers. The layout is a contract with its readers. Changing it means a parallel change: write both layouts, move the readers, then drop the old one.

## Sequencing

1. Parity harness (#2).
2. Shared framework: `RunConfig`, I/O.
3. Spike on a staging run, before porting places: Sedona 1.8.1 `ParseAddress` on the provider data where the custom jpostal build originally fixed an executor hang, and Python worker memory on Glue at full scale with rapidfuzz, a booster and the sketch readers loaded.
4. Places setup stage: Sedona `ParseAddress` with the v1.0.0 data, phones, `PlaceTextNormalizer` JVM UDF, signatures, idf maps.
5. Places ID assignment and metrics.
6. Places matcher (`PlacesUnionedMatcher` behavior only; `PlacesMatcher` is not ported).
7. Divisions.
8. Addresses.
9. Behavior changes, one at a time, each with an expected-delta check and a GERS churn review:
   - remove baseline-baseline pairs from candidate runs (OvertureMaps/overture-matchers#431), which also cuts blocking from about 42B to an estimated 2-3B pairs
   - apply overrides in places
   - deterministic divisions selection, if not already adopted in step 7
   - the baseline-conflation and linkage theme functions that replace `PlacesMatcher`
   - porting `PlaceTextNormalizer`, if wanted

Each theme cuts over to Python as a whole, once all its stages pass parity. OvertureMaps/overture-matchers is frozen except for fixes and archived after the last theme cuts over.

## Effects on other repos

- tf-places-matcher-model: releases the model as a Python package to CodeArtifact (manifest JSON, booster files, sketch binaries, test datasets as package data) and the `PlaceTextNormalizer` jar to CodeArtifact Maven, instead of one Maven jar holding both. Dual-publishes the old jar until places cuts over. Training keeps its own feature code for now (see Open items).
- tf-data-platform, in the places match DAG:
  - the match task switches from the Scala main class and jar to the job in `overture_matchers.places.job`, installed from PyPI at a pinned version
  - the match task stays one task running one job, `PlacesMatchingJob`, on one cluster, as Scala does today. An Airflow retry resumes at the first stage without a marker. Each theme gets its own task group
  - the Scala matcher version setting becomes a matcher package version, plus a model package version
  - the platform Sedona version moves to 1.8.1
  - job params, extra jars and libpostal data change too (see Details for agents)
  - the match run becomes one `MatchBundle`, a new `Bundle` subclass, set up by the bundle initialize task group before the job and sealed by the finalize task group after it. Candidate and baseline paths are resolved at run time, so the initialize task group needs a way to record input paths from XCom; today it takes only `Bundle` objects known when the DAG is parsed. Between the job and finalization, a new task reads `config.json` and adds it to `metadata.json` under `matcher`. This replaces the current metadata step, which writes metadata directly after matching and records only candidates as inputs
  - readers of Scala's run folder (telemetry, match evaluation, the previous-metrics lookup, the helper that derives the run folder from the final output path) keep their paths inside the run folder. Only the run folder itself changes: the bundle base URI instead of `<outputPath>/<runId>`
  - the dev publishing and staging dispatch flow is rebuilt for wheels
- tf-places-3ppp (third-party places pipeline): builds its own matcher image from a pinned overture-matchers commit and runs it with the places-signals config, which uses `PlacesMatcher`. The port does not cover `PlacesMatcher`, so 3ppp stays on its pinned Scala commit until the linkage mode exists.
- Local Docker runtime (OvertureMaps/overture-matchers#390, open): the data scientists' local validation loop (OvertureMaps/tf-places-matcher-model#146) runs the Scala matcher through `just match`. After places cuts over, the loop runs the local runner in `overture_matchers.places.job`, which calls the same `run()` as production.

## Open items

- When shared code moves into its own package, and that package's name.
- Training and features, after the migration: where the single copy of the feature code lives, and how a model and the feature code it was trained on are kept in step. One option is for tf-places-matcher-model to import `places/features.py` from this repo. The other is for this repo to import the training repo's feature code as an optional dependency, which needs CodeArtifact access in CI and faster feature code in the training repo. Today `FeatureParityTest.scala` recomputes the model jar's `test_dataset.parquet` features in the matcher's CI.
- Not yet measured: divisions and addresses prod volumes, the 3PPP/signals pipeline.
- Parity acceptance: what each stage's comparison checks and what difference is acceptable before a theme cuts over.
- A mapping of every Scala, DAG and `just match` param to its `RunConfig` field, theme constant, Spark setting, or removal.
- OvertureMaps/overture-embeddings (Python) owns the embedding model pin, int8 quantization, cosine scoring and text normalization. Check whether the places embedding feature can use it before porting `EmbeddingCacheCosineSimilarityCalculator`.

## Details for agents

Implementation details behind the design above.

### Target structure

- Steps and stages do no I/O, so Spark evaluates them lazily and nothing executes until `run_stage` writes the outputs.
- The Python job's `outputPath` is the `MatchBundle` base URI, the counterpart of Scala's `<outputPath>/<runId>`. Inside it, outputs sit at Scala's paths (`candidates/`, `matches/`, `final/`, ...), next to `config.json` and the stage markers in `_completed_stages/` (`_completed_stages/setup`, `_completed_stages/matching`, ...).
- The DAG is the only writer of `metadata.json`, because `Metadata` keys its sections by the Airflow `dag_id` and rewrites the whole file on every update. Downstream DAGs list the match bundle as an input, which carries its metadata forward.
- The code never checks which theme it's running. The DAG picks the theme by choosing which job to launch, and that theme's `run()` fixes the stage order.
- Places has three matching modes (see OvertureMaps/overture-matchers#431): places conflation, baseline conflation and linkage. Each mode gets its own match function. They share the same building blocks and differ only in which kinds of pairs they compare and how they pick the final match.
- Follow Overture conventions in the code: MIT header and a one-paragraph purpose on every file, real docstrings on public functions, DCO sign-off (`git commit -s`), `python:X.Y-slim` images with uv from a pinned `ghcr.io/astral-sh/uv` stage.
- `RunConfig` exists only in the Python process that drives the job (the Spark driver). Executors get plain values from it, never the object.
- `RunConfig` holds the job params, with defaults filled in and values parsed into real types. Theme constants and the model stay out of it.
- The job starts with `RunConfig.for_run(params)`, which builds `RunConfig` from the params and writes `config.json` before reading any input. If the run folder already has a `config.json` that differs, the job fails. Otherwise a rerun with changed params would skip stages that ran with the old ones. Apart from that check, nothing reads `config.json` back: a retry gets the same params from Airflow and builds the same config. Anything resolved at run time (a "latest" baseline, a timestamp) is resolved in the DAG and passed in as a param, so the job never resolves anything on its own.
- Every stage write uses `mode("overwrite")`, so rerunning a stage without a marker replaces whatever a failed attempt left behind.
- `run_stage` writes a stage's marker after all of the stage's outputs are written, and clears Spark's cache before the next stage. Spark's own `_SUCCESS` files mark single DataFrame writes, so they never stand in for a stage marker.
- `config.json` lists every field, defaults included, plus the installed matcher and model package versions, so the bundle metadata shows exactly what the run used. A changed matcher or model version between attempts fails the same check as changed params. This is on purpose: after a version change, start a new run folder. Example (values are placeholders):

  ```json
  {
    "output_path": "s3://.../<run>/",
    "input": {"place": ["s3://.../candidates/"]},
    "baseline": {"place": ["s3://.../baseline/"]},
    "provider": "<provider>",
    "scenario": "<scenario>",
    "environment": "prod",
    "match_history": true,
    "output_debug": false,
    "embeddings_cache_table": null,
    "previous_metrics_path": "s3://.../<previous run>/metrics/",
    "versions": {"overture-matchers": "<version>", "<model package>": "<version>"}
  }
  ```

### Rules

- Transformation functions are pure. S3 reads and writes happen only in each theme's `run()`, in `run_stage`, and in the checkpoint function from `io.checkpointer(output_path)`. Calling it as `checkpoint(df, name)` writes `df` to `<run>/<name>/` and returns a read of it. That cuts the lineage, so later transformations don't recompute earlier ones and the query plan stays small. Matching checkpoints after blocking, where about 42B spatial pairs shrink to about 170M. Checkpoint outputs stay with the run. With `outputDebug` on, matching also writes `blocking/`, `scores/` and `matches_validated/`, and `assign_ids` writes `idMap/`, through the same checkpoint function, as Scala does.
- A DataFrame that two branches of one stage read gets an explicit `.cache()`. In `match_places`, candidates feed both the history pairs and the spatial union. `run_stage` writes each output separately, so upstream work shared by two outputs of one stage gets cached before the stage returns: in `assign_ids`, `final` and `fallout` both come from the cluster winner selection, which Scala caches too.
- Each stage output's columns are defined by the `select` that produces it, as in Scala (the transformer's required-columns list, the `MatchResult` columns). There is no separate schema validation layer for now; schema tracking across Overture pipelines is an open, org-wide question.
- Setup functions build one `select` from a list of column expressions. A `withColumn` chain adds one projection per column to the query plan.
- Boosters and sketch readers load once per executor process, through module-level `functools.cache` loaders keyed by resource name (the blocking booster, the scoring booster, each sketch). The model package is installed on every worker, so UDF closures capture only the resource name, never the loaded object.
- The model package is a Python package in CodeArtifact, published by tf-places-matcher-model, with the boosters, sketches and manifest as package data. The DAG installs it on every worker at a version pinned by its own Airflow Variable, and `PlacesModel` finds its files with `importlib.resources`. It is not a dependency in this repo's `pyproject.toml`, so a model release does not force a matcher release.
- The embeddings cache is an Iceberg table, so read it with `spark.table()`. The table name is a `RunConfig` field, so a local run can point it at a temp view of local parquet.
- Classes hold data and resources only: `RunConfig` (frozen dataclass), `PlacesModel`, and the stage output `NamedTuple`s. No abstract base classes, no strategy objects, no emulation of typed Datasets.
- Overrides and history matches are pre-matches (Scala's `preMatch`): pairs accepted before blocking, without scoring. Each theme function calls the shared overrides and history functions it uses. During the port, places calls only the history one (see Known parity traps).
- From the matching stage, `assign_ids` gets only `MatchingOutput`, never matching's checkpoints.
- Features live in `places/features.py` and run only in Spark: built-in expressions for blocking features, one UDF for pairwise features (see Performance rules). `PlacesModel.load()` reads the manifest for the thresholds, the spatial radius and the feature order. The manifest also carries the feature names and a feature version, and `load()` asserts both against `features.py`.
- All outputs share one folder namespace. A test checks that output names (the output types' fields plus checkpoint names) are unique and that the output types' fields match Scala's postfixes, since readers depend on them.
- Stage function names are also marker names, so every theme names its stage functions `setup`, `matching`, `assign_ids` and `metrics`. A test checks this too.
- Every transformation function and stage has unit tests on a local SparkSession with a few rows. One shared places fixture covers each path through matching: history hit, history miss across a quadkey tile edge, renamed record, candidate-candidate duplicate, deleted baseline, no match, and two baseline records with the same signature.
- Tests never load the production model package, which is private. A fixture trains a tiny booster with `xgboost` and builds sketches through Spark's JVM sketch classes, both on the fixture rows.
- The `PlaceTextNormalizer` jar is private too. Fixtures carry the normalized columns precomputed. The few tests that run the normalizer itself carry a `normalizer` marker and skip when the jar is absent; they run locally and in tf-places-matcher-model's CI.

### Performance rules

- Moving a row of two strings from the JVM to Python costs 3.4-5.4 µs per core, about the same for pickled, Arrow-optimized and pandas UDFs. Numeric rows cost less: seven doubles cross through Arrow in about 0.85 µs. That decides where Python UDFs are acceptable.
- Blocking features run on all spatial pairs (about 42B per run). Implement them as built-in Spark functions, never Python UDFs.
  - name similarity: `1 - levenshtein(a, b) / greatest(length(a), length(b))`
  - name TF-IDF: a `map<token, idf>` column built once per record in setup, then `array_intersect` plus map lookups
- Pairwise features (about 170M pairs) go in one rapidfuzz-backed row-at-a-time `@udf` returning a struct with all features, so each pair crosses to Python once. Measured on the fused features, the plain pickled UDF is the fastest of the three types (42.5 µs/row, against 44.2 Arrow-optimized and 48.3 pandas). Model scoring is numeric and batched, so it runs `inplace_predict` in a pandas UDF.
- Python UDFs inside `when()` run on every row (no short-circuit). Restructure conditional scoring so the expensive branch only sees the rows that need it.
- Projected cost of this layout, extrapolated from local microbenchmarks: about +3-4% run time. Python UDFs everywhere: about +15%. A literal pure-Python port: about 2.7x. The blocking model's pandas UDF runs on all 42B pairs, so each extra µs per row adds about a minute. A full-scale staging run measures the real cost before cutover.
- Keep Scala's spatial-join settings at first: repartition to `defaultParallelism * 20`, its Sedona hints, AQE off, 1,920 shuffle partitions after blocking (`PlacesUnionedMatcher.scala`).
- Set the Spark checkpoint directory for GraphFrames connected components, as Scala's `JobUtils` does.
- XGBoost runs with `nthread=1`, and the job sets `OMP_NUM_THREADS=1`, so 32 Python workers per Glue worker don't oversubscribe the cores.
- Cache with an explicit serialized storage level (`MEMORY_AND_DISK_SER`, as Scala's loader does), on narrow projections, and `unpersist()` as soon as the branches that read it finish.
- Glue: each worker runs up to 32 Python worker processes outside the JVM heap, each holding rapidfuzz, the booster and sketch readers. Test memory at full scale.

### Parity

There is no Scala-vs-Scala baseline. No places step depends on row order: ties break by score, then ID, and new group IDs are random but the comparison matches groups by their members. Divisions and addresses do have order-dependent steps (`first(...)`, greedy selection, `dropDuplicates`, the top-50 cut); if their comparison shows differences nobody can explain, check run-to-run variation then.

Harness layout: `parity/` sits outside `packages/`, so nothing in it ships to PyPI.

- `compare.py` is a standalone uv script with inline dependencies. Scala and Python run folders share a layout, so it reads both the same way. It reads output parquet with DuckDB, locally or straight from S3, with no Spark, and prints JSON.
- `test_compare.py` tests it on small local parquet fixtures. Nothing touches AWS.
- `README.md` gives the plain `aws` CLI commands that start Glue runs, each in its own run folder (a distinct `runId` for Scala, a distinct `outputPath` for Python), and that copy a Scala run's earlier stage outputs into a new run folder and write their stage markers for mixed mode.
- The regional test input lives at a private path passed as an argument, never committed.

### Known parity traps

1. `PlacesUnionedMatcher` never applies overrides: `getMatches` takes the overrides table and does not read it. The uniform overrides pre-step changes places results, so it stays off for places during the port. Decide whether this is a bug, and fix it as its own change.
2. Pair orientation: the unioned matcher emits a pair only when `s.id < t.id`, so a pair's sides are in ID order, not role order. After clustering, Scala swaps baseline-candidate pairs so the candidate is in `id`. Python skips the swap and names each pair's sides by role (candidate, baseline). The `matches` output keeps Scala's orientation, and a test pins it. Scala scores pairs in ID order, so if the sides are named by role before scoring, first check that every pairwise feature is symmetric.
3. Cluster winner selection (`SimpleBestMatchSelection`): prefer active baseline, then occurrence count, then composite score, then `base_id`. The occurrence count for a `base_id` includes every pair with that `base_id`, baseline-baseline pairs too. Only candidate-baseline pairs can be the selected pair.
4. Divisions greedy selection partitions rows by Java `String.hashCode` (`SparkUtils.getHash`), and selection state is shared within a partition. First try a deterministic window-based version and measure. If the differences from Scala are large, reproduce Java's hash as a native Spark expression (signed 32-bit wraparound over UTF-16 code units) with the same partition count.
5. String semantics: commons-text Levenshtein counts UTF-16 units; Spark's `levenshtein()` and Python's `Levenshtein`/`rapidfuzz` count code points. They differ only on emoji and other non-BMP characters. Accept and document it; training already uses code points.
6. Java and Python `unidecode` tables differ. Keep `PlaceTextNormalizer` on the JVM to avoid this.
7. Float details (division order, `log` in idf, float vs double) can flip pairs sitting exactly at a threshold. These differences are expected and get reviewed.

### tf-data-platform DAG changes

- params drop the leading dashes (`outputPath`, not `--outputPath`): a Python job receives its params as one JSON object, and `SparkSedonaJob.get_param` looks up bare names
- the DAG stops sending `overrideConfigPath`, which it always set to `none`
- the DAG stops sending `runId`. Scala uses it to stamp `snapshot` (`<provider>-<theme>-<runId>`) on every metrics row. Python's metrics rows leave out `snapshot` and `stage`, and `MatcherTelemetryJob` fills both through `TelemetryEmitter`, which already builds the same snapshot from its own `run_id` param
- the job's extra jars are GraphFrames (shaded into the Scala jar today), the `PlaceTextNormalizer` jar, and spark-nlp for divisions. jpostal and libphonenumber drop out: Sedona 1.8.1 ships jpostal, and phones use `phonenumbers`
- libpostal model data: the Scala job receives it as a Glue extra file, which works only because a JVM job's params pass through as Glue arguments, and a Python job's params do not. Instead, the v1.0.0 model data sits unpacked under an S3 prefix, the job's Spark conf sets `spark.sedona.libpostal.dataDir` to it, and setup calls `sc.addFile(<prefix>, recursive=True)`, Sedona's documented mode for remote data. The `.so` files drop out
- the DAG stops reading the job's `outputResultFile`. The final path is fixed by the layout (`<outputPath>/final/type=<table>/`), so the DAG builds the `{"<table>": "<path>"}` map itself. The jobs write no result file and ignore an `outputResultFile` param, so cloned Scala arguments still run

### Local Docker runtime

`just match` passes arguments the DAG does not: `embeddingsCachePath`, `embeddingCacheLiteralInMaxHashes`, `blockingSpatialPartitions`, `blockingShufflePartitions`, `platform`, `storageAccountName`.
