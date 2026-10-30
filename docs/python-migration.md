# Python migration plan

Design for porting the matchers (places, divisions, addresses) from OvertureMaps/overture-matchers (Scala) to Python/PySpark in this repo. Tracked in #1. The evaluation behind it, with benchmarks, is in OvertureMaps/overture-matchers#455.

## Goal and constraints

- Port the pipeline to Python/PySpark and use the port to clean up the structure.
- Final matching outputs stay close to Scala's. The parity harness reports the differences, and people review them (see Parity below). Exact parity is not required (but it would be nice).
- Production runs on AWS Glue 5 (Spark 3.5, Python 3.11), launched by the places match DAG in tf-data-platform.
- Follow Overture conventions in the code: MIT header and a one-paragraph purpose on every file, real docstrings on public functions, DCO sign-off (`git commit -s`), `python:X.Y-slim` images with uv from a pinned `ghcr.io/astral-sh/uv` stage.

## Current system (Scala), in brief

- `Main` runs four stages chosen by `--stages`: setup, matching, assignid, metrics. Stages hand off only through folders under the run's S3 output path (`candidates/`, `baseline/`, `matches/`, `final/`, `fallout/`, `metrics/`).
- Each stage calls the slots of a `Theme` (Transformer, HistoryFilter, EmbeddingsGenerator, Matcher, IdAssignmentPipeline). The config JSON in `resources/configs/` picks each slot's class.
- Matchers form an inheritance chain: `PlacesUnionedMatcher extends PlacesMatcher extends Matcher`, and each level overrides most of the one above.
  - `PlacesUnionedMatcher` (production places) unions candidates and baseline, spatially self-joins within 400 m (Sedona), prunes with an XGBoost blocking model (p ≥ 0.08), scores with a 28-feature XGBoost model (p ≥ 0.48), and clusters with GraphFrames connected components. History matches skip blocking.
  - `PlacesMatcher` is used only by the places-signals config: quadkey tile blocking with a name filter, candidate-to-baseline pairs only, best baseline per candidate. It is stale (OvertureMaps/overture-matchers#431).
  - `DivisionsMatcher`: quadkey blocking per subtype, rule-based scores in `Matcher.scala`, per-subtype thresholds, greedy restricted-duplicates selection.
  - `AddressesMatcher`: tier 1 exact signature match, tier 2 H3 res-14 blocking, street and house-number pre-score with a top-50 cut, weighted composite (max of two weightings), threshold 0.95, best baseline per candidate.
- New IDs are UUIDv4 for places and divisions, and H3-based for addresses and places-signals (`idGenerator` in each config).
- The model artifact `places-matcher-model` (from tf-places-matcher-model) is a Maven jar with the XGBoost models, feature order, thresholds, count-min sketch and Bloom filter binaries, and the Java `PlaceTextNormalizer`. The Scala matcher pins its version in `pom.xml` and shades it into the matcher jar, so every model release needs a matcher release.

Reference volumes from a recent production places run (74 min on 31 Glue workers): 140.2M baseline, 61.5M candidates, 57.5M history matches, about 42B spatial pairs, about 170M pairs surviving blocking, 110.7M match rows, 59.7M final records.

## Target structure

At a high level, I'd like to remove the class structure entirely:

* Matching is a chain of *stages*
  * e.g. setup -> matching -> assign_ids -> metrics
  * Stages are Spark jobs
  * Each stage writes to separate subfolders within a `Bundle`. I did consider using separate bundles, but matching is typically run all-at-once, and the metadata.json gets needlessly messy if everything is in its own bundle. I think that, in general, 1 DAG run ~ 1 Bundle (lmk if you disagree)
* Each stage is a DAG (not an Airflow DAG, just a DAG) of *steps*
  * Each step is a `DataFrame`->`DataFrame` transformation.
  * Steps chain together with `df.transform`, e.g. `df.transform(spatial_pairs, radius_m=400)`
* `RunConfig` is still serialized into JSON, but it lives in-memory as a frozen dataclass.
  * The `RunConfig` object is built from the job params during pre-setup, then written to JSON at `s3://.../<run>/config.json`. Subsequent stages initialize the config object from that JSON.
  * `config.json` is stamped into the bundle's `metadata.json` before finalization.

Steps do no I/O, so Spark evaluates them lazily and nothing executes until the stage writes. When a step takes several DataFrames, it is a plain call instead of `df.transform`.

`RunConfig` lives on each stage's Spark driver. Executors never receive it; steps get plain values from it. Pre-setup is the start of the first stage, before it reads anything. A later stage whose params (other than `stages`) disagree with `config.json` fails. While a run mixes Scala and Python stages, a Scala setup writes no `config.json`, so the first Python stage finds none and writes it.

The job knows nothing about bundles: it takes the bundle's base URI as `outputPath` and writes stage outputs as subdirectories under it. It has no `runId`: Scala's `<outputPath>/<runId>` becomes the Python job's `outputPath`, so both write the same run folder layout. The DAG is the only writer of `metadata.json`, because `Metadata` keys its sections by the Airflow `dag_id` and rewrites the whole file on every update. It calls `bundle.initialize(...)` before the first stage; after the last stage, it reads `config.json`, adds it to the metadata under `matcher` and calls `bundle.finalize()`. Downstream DAGs list the match bundle as an input, which carries its metadata forward.

Some pseudocode outlining some of this:

```python
def spatial_pairs(records: DataFrame, radius_m: float) -> DataFrame: ...
# left join on (provider, signature)
def history_pairs(cands: DataFrame, base: DataFrame) -> DataFrame: ...


def match_places(
    cfg: RunConfig, cands: DataFrame, base: DataFrame, model: ModelBundle
) -> DataFrame:
    cands = cands.cache()
    history = history_pairs(cands, base)
    scored = (
        union_records(cands.where("base_ids IS NULL"), base)
        .transform(spatial_pairs, radius_m=model.spatial_radius_m)
        .transform(model_filter, model.blocking)
        .transform(checkpoint, cfg, "blocking")
        .transform(model_score, model.matcher)
        .where("prediction = 1")
    )
    return connected_components(history.unionByName(scored))


def matching_stage(cfg: RunConfig) -> None:
    cands, base = read_stage(cfg, "candidates"), read_stage(cfg, "baseline")
    matches = THEMES[cfg.theme].match(cfg, cands, base, ModelBundle.load())
    write_stage(cfg, "matches", matches)
```

Each theme is one module (`places.py`, later `divisions.py` and `addresses.py`) with one function per stage: `setup`, `match`, `assign_ids` and `metrics`. These stages are built from shared transformation functions when possible.

`THEMES` in `stages.py` maps the theme name to its module, the only place the code branches on theme. Themes don't share a fixed sequence of transformations. Addresses runs two tiers and a top-50 cut. Divisions applies per-subtype thresholds and greedy selection. Places conflation, baseline conflation and linkage (the three modes in OvertureMaps/overture-matchers#431) are three match functions over the same building blocks that differ in pair kinds and resolver.

Scala's theme config JSON mostly wires classes by name, which plain functions make unnecessary. What remains splits two ways: run settings (`outputDebug`, `matchHistory`) are `RunConfig` fields, and theme constants (table name, ID generator) are constants in the theme module. The DAG always passes `overrideConfigPath=none`; the job accepts that value and rejects any other.

The production job is `MatchJob`, a subclass of overture-core's `SparkSedonaJob`, launched from tf-data-platform like the other Overture Spark jobs. Its params use Scala's argument names. `execute_job` builds or loads `RunConfig` (see above) and runs the stages named in `stages`. Locally, a developer runs the same job with `MatchJob().run(params)` on a local SparkSession, pointing `outputPath` at a local directory and the inputs at the test fixtures.

Package layout:

```
packages/
  overture_matchers/
    overture_matchers/
      job.py       # MatchJob(SparkSedonaJob): get RunConfig, run the requested stages
      config.py    # frozen dataclass RunConfig: build from params, or load from config.json
      io.py        # read_stage/write_stage, checkpoint, config.json write and read
      stages.py    # one function per stage: read, call the theme, write; THEMES
      places.py    # places setup, match, assign_ids, metrics
      model.py     # ModelBundle, XGBoost predict
  <library>/       # the library package, from Sequencing step 4
    <library>/
      features.py  # place-pair features, imported by training and inference
```

Other modules are added as code arrives, named for what they do. A module becomes a subpackage only when it outgrows one file.

The repo follows overture-core's `packages/` layout. When the first transformation functions land (Sequencing step 4), they move into a second package, the library package: pure transformation functions, including `features.py`. `overture_matchers` keeps the job, config, io, stages and theme modules, and depends on the library package. The library package never imports the matcher. Each package is tested in its own environment, so CI catches an import in the wrong direction. A function enters the library package when it is theme-agnostic or a second theme needs it; theme-specific steps stay in their theme module.

Each package is versioned from its own `<package>-v<version>` tags through python-semantic-release and setuptools_scm, and published to public PyPI by trusted publishing, as in overture-core.

Rules:

- Transformation functions are pure. S3 reads and writes happen only in stage functions and in `checkpoint(df, cfg, name)`, which writes to `<run>/<name>/` and returns a read of it. That cuts the lineage, so later transformations don't recompute earlier ones and the query plan stays small. Matching checkpoints after blocking, where about 42B spatial pairs shrink to about 170M. Checkpoint outputs stay with the run. With `outputDebug` on, matching also checkpoints `matches_validated/`, as Scala does.
- A DataFrame that two branches of one stage read gets an explicit `.cache()`. In `match_places`, candidates feed both the history pairs and the spatial union.
- Each stage output's columns are defined by the `select` that produces it, as in Scala (the transformer's required-columns list, the `MatchResult` columns). There is no separate schema validation layer for now; schema tracking across Overture pipelines is an open, org-wide question.
- Setup functions build one `select` from a list of column expressions. A `withColumn` chain adds one projection per column to the query plan.
- Boosters and sketch readers load once per executor process, through module-level `functools.cache` loaders keyed by resource name (the blocking booster, the scoring booster, each sketch). The model package is installed on every worker, so UDF closures capture only the resource name, never the loaded object.
- The model bundle is a Python package in CodeArtifact, published by tf-places-matcher-model, with the boosters, sketches and manifest as package data. The DAG installs it on every worker at a version pinned by its own Airflow Variable, and `ModelBundle` finds its files with `importlib.resources`. It is not a dependency in this repo's `pyproject.toml`, so a model release does not force a matcher release.
- Classes hold data and resources only: `RunConfig` (frozen dataclass) and `ModelBundle`. No abstract base classes, no strategy objects, no emulation of typed Datasets.
- Overrides and history matches are pre-matches (Scala's `preMatch`): pairs accepted before blocking, without scoring. Each theme function calls the shared overrides and history functions it uses. During the port, places calls only the history one (see traps).
- From the matching stage, assignid reads only `matches/`, never matching's checkpoints.
- Features live in `features.py` and run only in Spark: built-in expressions for blocking features, one UDF for pairwise features (see Performance rules). Training installs the library package alone and imports the same module, so training and inference compute features with the same code. The model bundle records the library package's version, not the matcher's, so a library release signals a possible retrain. `ModelBundle.load()` compares that version with the installed library package and fails the stage on a mismatch, so features never change under a model silently.
- Every transformation function has unit tests on a local SparkSession with a few rows. One shared places fixture covers each path through matching: history hit, history miss across a quadkey tile edge, renamed record, candidate-candidate duplicate, deleted baseline, no match, and two baseline records with the same signature.
- Tests never load the production model bundle, which is private. A fixture trains a tiny booster with `xgboost` and builds sketches through Spark's JVM sketch classes, both on the fixture rows.
- The `PlaceTextNormalizer` jar is private too. Fixtures carry the normalized columns precomputed. The few tests that run the normalizer itself carry a `normalizer` marker and skip when the jar is absent; they run locally and in tf-places-matcher-model's CI.

## Library swaps

| Scala dependency | Python replacement | Note |
| --- | --- | --- |
| `com.wherobots:jpostal` 1.2.2 (OvertureMaps/overture-matchers#478) | Sedona 1.8.0 `ParseAddress` | Same jpostal build Sedona uses. Set `spark.sedona.libpostal.dataDir` to the v1.0.0 model data the Scala job uses today. Tested 100% identical to the earlier custom build on 449k public addresses. Sedona's default model download changes 18% of non-US addresses; do not use it during the port. |
| Sedona 1.7.2 | Sedona 1.8.0 Python API | tf-data-platform pins one Sedona version for all jobs; regression-check the other Sedona jobs. |
| xgboost4j-spark | `xgboost.Booster().load_model()` on the native booster file, `inplace_predict` in a pandas UDF | Numeric Arrow crossing costs about 0.85 µs/row. |
| GraphFrames (Scala) | `graphframes` Python API | Same jar. |
| libphonenumber | `phonenumbers` |  |
| spark-nlp (Scala) | spark-nlp Python API, pin `spark-nlp==5.5.1` | Assembly jar already attached for divisions. |
| commons-text, stringdistance | Spark built-in `levenshtein()` in blocking; `rapidfuzz` in UDFs |  |
| `PlaceTextNormalizer` (Java) | Keep as a JVM UDF via `registerJavaFunction` during the port. The Java source stays in tf-places-matcher-model, which publishes the jar on its own to CodeArtifact Maven. | Training calls it the same way. Porting it to Python is a later, separately measured step. |
| Spark sketch binaries | Small Python reader for Spark's CountMinSketch and BloomFilter formats | Do not re-export as exact lookup tables: the sketches are approximate and the current features include their errors. |

## Performance rules

- Moving a row of two strings from the JVM to Python costs 3.4-5.4 µs per core, about the same for pickled, Arrow-optimized and pandas UDFs. Numeric rows cost less: seven doubles cross through Arrow in about 0.85 µs. That decides where Python UDFs are acceptable.
- Blocking features run on all spatial pairs (about 42B per run). Implement them as built-in Spark functions, never Python UDFs.
  - name similarity: `1 - levenshtein(a, b) / greatest(length(a), length(b))`
  - name TF-IDF: a `map<token, idf>` column built once per record in setup, then `array_intersect` plus map lookups
- Pairwise features (about 170M pairs) go in one rapidfuzz-backed row-at-a-time `@udf` returning a struct with all features, so each pair crosses to Python once. Measured on the fused features, the plain pickled UDF is the fastest of the three types (42.5 µs/row, against 44.2 Arrow-optimized and 48.3 pandas). Model scoring is numeric and batched, so it runs `inplace_predict` in a pandas UDF.
- Python UDFs inside `when()` run on every row (no short-circuit). Restructure conditional scoring so the expensive branch only sees the rows that need it.
- Expected cost of this layout: about +3-4% run time. Python UDFs everywhere: about +15%. A literal pure-Python port: about 2.7x.
- Glue: each worker runs up to 32 Python worker processes outside the JVM heap, each holding rapidfuzz, the booster and sketch readers. Test memory at full scale.

## Parity

`compare.py` compares two run folders and reports, for people to review:

1. ID agreement: the share of candidates that get the same ID in both runs, overall and broken down by country and by match type. A candidate agrees when both runs reuse the same GERS ID, or both mint a new ID for the same group of candidates. New IDs are random, so their values are never compared. A candidate missing from either run counts as a disagreement.
2. Cluster agreement: of the record pairs one run groups together, the share the other run also groups together, measured in both directions. GraphFrames `cluster_id` values are arbitrary, so only which records share a cluster counts.

Method, set up before any porting:

- A fixed regional test input, large enough to cover several countries and scripts.
- Pin both sides to the same model. The Scala jar shades in its model version (`places.matcher.model.version` in its `pom.xml`), so a Python-vs-Scala run pins the model package to that version.
- Mixed mode: run one Python stage on a Scala run's earlier stage outputs, so the differences come from that stage alone. A comparison of one stage's outputs gets built when matching is ported, shaped by what that work needs.

There is no Scala-vs-Scala baseline. No places step depends on row order: ties break by score, then ID, and new group IDs are random but the comparison matches groups by their members. Divisions and addresses do have order-dependent steps (`first(...)`, greedy selection, `dropDuplicates`, the top-50 cut); if their comparison shows differences nobody can explain, check run-to-run variation then.

Harness layout: a top-level `parity/` directory in this repo, outside `packages/`, so nothing in it ships to PyPI. It is deleted once Scala is archived.

- `compare.py` is a standalone uv script with inline dependencies. It reads stage parquet with DuckDB, locally or straight from S3, with no Spark, and prints JSON.
- `test_compare.py` tests it on small local parquet fixtures. Nothing touches AWS.
- `README.md` gives the plain `aws` CLI commands that start Glue runs, each in its own run folder (a distinct `runId` for Scala, a distinct `outputPath` for Python), and that copy a Scala run's earlier stage outputs into a new run folder for mixed mode.
- The regional test input lives at a private path passed as an argument, never committed.

Two constraints hold until cutover: job params use Scala's argument names, including `stages`; and stage outputs keep Scala's subdirectory names (`<postfix>/type=<table>/`) and columns, so either implementation can read the other's output.

## Known parity traps

1. `PlacesUnionedMatcher` never applies overrides: `getMatches` takes the overrides table and does not read it. The uniform overrides pre-step changes places results, so it stays off for places during the port. Decide whether this is a bug, and fix it as its own change.
2. Pair orientation: the unioned matcher emits a pair only when `s.id < t.id`, then swaps baseline-candidate pairs into candidate-baseline order before clustering. Encode both as tested rules.
3. Cluster winner selection (`SimpleBestMatchSelection`): prefer active baseline, then occurrence count, then composite score, then `base_id`. Occurrence counting ignores baseline-baseline pairs.
4. Divisions greedy selection partitions rows by Java `String.hashCode` (`SparkUtils.getHash`), and selection state is shared within a partition. First try a deterministic window-based version and measure. If the differences from Scala are large, reproduce Java's hash as a native Spark expression (32-bit wraparound over code points) with the same partition count.
5. String semantics: commons-text Levenshtein counts UTF-16 units; Spark's `levenshtein()` and Python's `Levenshtein`/`rapidfuzz` count code points. They differ only on emoji and other non-BMP characters. Accept and document it; training already uses code points.
6. Java and Python `unidecode` tables differ. Keep `PlaceTextNormalizer` on the JVM to avoid this.
7. Float details (division order, `log` in idf, float vs double) can flip pairs sitting exactly at a threshold. These differences are expected and get reviewed.

## Sequencing

1. Parity harness (#2).
2. Shared framework: `RunConfig`, I/O, `MatchJob`.
3. Spike on a staging run, before porting places: Sedona 1.8.0 `ParseAddress` on the provider data where the custom jpostal build originally fixed an executor hang, and Python worker memory on Glue at full scale with rapidfuzz, a booster and the sketch readers loaded.
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

OvertureMaps/overture-matchers is frozen except for fixes and archived after the last stage cuts over.

## Effects on other repos

- tf-places-matcher-model: computes features in Spark with the library package from PyPI instead of its own copies. Releases the model bundle as a Python package to CodeArtifact (manifest JSON, booster files, sketch binaries, test datasets as package data) and the `PlaceTextNormalizer` jar to CodeArtifact Maven, instead of one Maven jar holding both. Dual-publishes the old jar only while Scala stages remain in production.
- tf-data-platform, in the places match DAG:
  - the match task switches from the Scala main class and jar to `overture_matchers.job.MatchJob`, installed from PyPI at a pinned version
  - the single match task splits into four chained tasks, one per stage (`stages=setup`, then `matching`, `assignid`, `metrics`), so a failed stage retries alone. Each job adds about a minute of Glue startup, which is accepted
  - params drop the leading dashes (`outputPath`, not `--outputPath`): a Python job receives its params as one JSON object, and `SparkSedonaJob.get_param` looks up bare names
  - the Scala matcher version setting becomes a matcher package version, plus a model package version
  - the job's extra jars are GraphFrames (shaded into the Scala jar today), the `PlaceTextNormalizer` jar, and spark-nlp for divisions. jpostal and libphonenumber drop out: Sedona 1.8.0 ships jpostal, and phones use `phonenumbers`
  - libpostal model data: the Scala job receives it as a Glue extra file, which works only because a JVM job's params pass through as Glue arguments, and a Python job's params do not. Instead, the v1.0.0 model data sits unpacked under an S3 prefix, the job's Spark conf sets `spark.sedona.libpostal.dataDir` to it, and setup calls `sc.addFile(<prefix>, recursive=True)`, Sedona's documented mode for remote data. The `.so` files drop out
  - the platform Sedona version moves to 1.8.0
  - the DAG stops reading the job's `outputResultFile`. The final path is fixed by the layout (`<outputPath>/final/type=<table>/`), so the DAG builds the `{"<table>": "<path>"}` map itself. `MatchJob` writes no result file and ignores an `outputResultFile` param, so cloned Scala arguments still run
  - the match run becomes one `Bundle`: `initialize` before the first stage, with both candidates and baseline as inputs; after metrics, the DAG adds `config.json` under `matcher` and calls `finalize()`. This replaces the current metadata step, which writes metadata directly after matching and records only candidates as inputs
  - the dev publishing and staging dispatch flow is rebuilt for wheels
- tf-places-3ppp (third-party places pipeline): builds its own matcher image from a pinned overture-matchers commit and runs it with the places-signals config, which uses `PlacesMatcher`. The port does not cover `PlacesMatcher`, so 3ppp stays on its pinned Scala commit until the linkage mode exists.
- Local Docker runtime (OvertureMaps/overture-matchers#390, open): the data scientists' local validation loop (OvertureMaps/tf-places-matcher-model#146) runs the Scala matcher through `just match`. It passes arguments the DAG does not: `embeddingsCachePath`, `embeddingCacheLiteralInMaxHashes`, `blockingSpatialPartitions`, `blockingShufflePartitions`, `platform`, `storageAccountName`. After places cuts over, the loop has to run the Python job to test what production runs.

## Open items

- Effort estimate: about 8-12 engineer-weeks, mostly tests (7.5k lines of ScalaTest) and parity work.
- The library package's name.
- Not yet measured: divisions and addresses prod volumes, the 3PPP/signals pipeline.
- OvertureMaps/overture-embeddings (Python) owns the embedding model pin, int8 quantization, cosine scoring and text normalization. Check whether the places embedding feature can use it before porting `EmbeddingCacheCosineSimilarityCalculator`.
