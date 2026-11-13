# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Places matching job.

Run locally with ``python -m overture_matchers.places.job <params.json>``.
"""

from __future__ import annotations

import json
import os
import sys
from functools import partial
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from overture_spark.job import SparkSedonaJob
from pyspark import StorageLevel
from pyspark.sql import functions as F

from overture_matchers import ids
from overture_matchers.core import io
from overture_matchers.core.config import RunConfig, rewrite_s3_to_s3a
from overture_matchers.core.libpostal import ship_libpostal_data_to_workers
from overture_matchers.core.local import run_local
from overture_matchers.core.stages import (
    AssignIdsOutput,
    MatchingOutput,
    MetricsOutput,
    RawInputs,
    SetupOutput,
    run_stage,
)
from overture_matchers.metrics import (
    MetricInputs,
    build_metrics_dataframe,
    compute_metrics,
)
from overture_matchers.places import features
from overture_matchers.places.blocking import (
    SPATIAL_JOIN_CONF,
    get_blocking_pairs,
    get_shuffle_partition_count,
    repartition_by_pair_key,
)
from overture_matchers.places.matches import accepted_pairs, match_results
from overture_matchers.places.metric_inputs import load_optional_metric_inputs
from overture_matchers.places.metrics import METRICS
from overture_matchers.places.model import (
    BLOCKING_BOOSTER_RESOURCE,
    SCORING_BOOSTER_RESOURCE,
    UNIQUE_NAME_H3_BIN_GE2_RESOURCE,
    UNIQUE_NAME_H3_BIN_GE10_RESOURCE,
    PlacesModel,
    bind_margin_and_probability_udf,
    bind_match_probability_udf,
    bind_name_idf_udf,
    load_scoring_lookups,
)
from overture_matchers.places.model_jar import CONFIG_RESOURCE, open_model_jar
from overture_matchers.places.prepare import (
    BLOCKING_RESOLUTION,
    add_base_ids,
    get_baseline_provider_column,
    prepare_baseline,
    prepare_candidates,
)
from overture_matchers.places.runtime import (
    bind_jvm_text_functions,
    bind_unique_name_h3_bin_udf,
    register_text_udfs,
    register_unique_name_h3_bin_udf,
)
from overture_matchers.places.scoring import (
    MISSING_EMBEDDINGS_CACHE_MESSAGE,
    PlacesPairScorer,
    score_pairs,
)
from overture_matchers.places.selection import select_cluster_winners_and_fallout
from overture_matchers.prematch import generate_history_pairs

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyspark.sql import DataFrame, SparkSession

FEATURE_TYPE = "place"
THEME = "places"
# GraphFrames 0.8.3 built for Spark 3.5. It has no dependencies beyond Spark.
PACKAGES = ("graphframes:graphframes:0.8.3-spark3.5-s_2.12",)


def setup(
    raw: RawInputs,
    provider: str,
    baseline_provider: str | None = None,
    match_history: bool = True,
    blocking_resolution: int = BLOCKING_RESOLUTION,
) -> SetupOutput:
    """Normalize candidates and baseline, and attach history matches if asked.

    Args:
        raw: Candidates and baseline as read.
        provider: Stamped on every candidate.
        baseline_provider: Provider for a baseline without a ``provider``
            column.
        match_history: Whether to fill ``base_ids`` from the baseline. Only
            setup reads this flag.
        blocking_resolution: Quadkey zoom for ``blocking_keys`` and ``signature``.

    Returns:
        Candidates with ``base_ids``, and the baseline with ``active``.
    """
    spark = raw.candidates.sparkSession
    # This fails fast before shipping if the baseline has a `provider` column
    # and baseline_provider is set, or if it has neither.
    baseline_provider_column = get_baseline_provider_column(
        raw.baseline, baseline_provider
    )
    ship_libpostal_data_to_workers(spark)
    register_text_udfs(spark)
    text_functions = bind_jvm_text_functions()
    baseline = prepare_baseline(
        raw.baseline, baseline_provider_column, text_functions, blocking_resolution
    )
    if match_history:
        # The baseline feeds both the history join and the stage output.
        baseline = baseline.persist(StorageLevel.MEMORY_AND_DISK)
    candidates = add_base_ids(
        prepare_candidates(
            raw.candidates, provider, text_functions, blocking_resolution
        ),
        baseline,
        match_history,
    )
    return SetupOutput(candidates, baseline)


def get_shipped_jar_name(model_jar: str) -> str:
    """The name workers find ``model_jar`` under once it is shipped."""
    return os.path.basename(urlparse(model_jar).path.rstrip("/"))


def build_places_pair_scorer(config: dict, jar_name: str) -> PlacesPairScorer:
    """The scorer of the jar's parsed scoring ``config``, reading the shipped jar."""
    return PlacesPairScorer.from_config(
        config,
        get_lookups=partial(load_scoring_lookups, jar_name),
        compute_unique_name_bin=bind_unique_name_h3_bin_udf(),
        predict_scores=bind_margin_and_probability_udf(
            jar_name, SCORING_BOOSTER_RESOURCE
        ),
    )


def load_embeddings_cache(
    spark: SparkSession, scorer: PlacesPairScorer, table_name: str | None
) -> DataFrame | None:
    """The embeddings cache ``table_name`` if ``scorer`` reads it, else None.

    Raises:
        ValueError: The scorer reads the cache, but ``table_name`` is None.
    """
    if not scorer.needs_embeddings:
        return None
    if table_name is None:
        raise ValueError(MISSING_EMBEDDINGS_CACHE_MESSAGE)
    return spark.table(table_name)


def matching(
    prepared: SetupOutput,
    model_jar: str,
    output_path: str | None = None,
    embeddings_cache_table: str | None = None,
    output_debug: bool = True,
) -> MatchingOutput:
    """Generate pairs, score them and cluster the matches.

    Args:
        prepared: Output of ``setup``.
        model_jar: Path or URI of the ``places-matcher-model`` jar, shipped to
            Python workers for the boosters and sketches.
        output_path: Run directory that receives the ``blocking`` and
            ``scores`` checkpoints.
        embeddings_cache_table: Table of name and taxonomy embeddings. It is
            resolved before blocking, and only if the model has a cosine feature.
        output_debug: Whether ``output_path`` also receives the accepted
            pairs, as ``matches_validated``.
    """
    spark = prepared.candidates.sparkSession
    normalized_jar = rewrite_s3_to_s3a(model_jar)
    spark.sparkContext.addFile(normalized_jar)
    jar_name = get_shipped_jar_name(normalized_jar)
    with open_model_jar(spark, normalized_jar) as jar:
        model = PlacesModel.load_from_jar(jar)
        scorer = build_places_pair_scorer(
            json.loads(jar.read(CONFIG_RESOURCE)), jar_name
        )
        if features.UNIQUE_NAME_H3_CELL_COUNT_BIN in scorer.feature_order:
            register_unique_name_h3_bin_udf(
                spark,
                jar.read(UNIQUE_NAME_H3_BIN_GE2_RESOURCE),
                jar.read(UNIQUE_NAME_H3_BIN_GE10_RESOURCE),
            )
    embeddings_cache = load_embeddings_cache(spark, scorer, embeddings_cache_table)
    candidates_to_block = prepared.candidates.filter(F.col("base_ids").isNull()).drop(
        "base_ids"
    )
    blocked_pairs = io.checkpoint(
        get_blocking_pairs(
            candidates_to_block,
            prepared.baseline,
            model,
            bind_name_idf_udf(jar_name),
            bind_match_probability_udf(jar_name, BLOCKING_BOOSTER_RESOURCE),
        ),
        output_path,
        "blocking",
        FEATURE_TYPE,
    )
    scores = io.checkpoint(
        repartition_by_pair_key(
            score_pairs(blocked_pairs, scorer, embeddings_cache),
            get_shuffle_partition_count(spark),
        ),
        output_path,
        "scores",
        FEATURE_TYPE,
    )
    validated = accepted_pairs(scores, generate_history_pairs(prepared.candidates))
    if output_debug:
        validated = io.checkpoint(
            validated, output_path, "matches_validated", FEATURE_TYPE
        )
    return MatchingOutput(match_results(validated, prepared.candidates))


def assign_ids(
    raw: RawInputs,
    prepared: SetupOutput,
    matched: MatchingOutput,
    provider: str,
    output_path: str | None = None,
) -> AssignIdsOutput:
    """Pick cluster winners and assign GERS IDs."""
    winners, fallout = select_cluster_winners_and_fallout(
        matched.matches, prepared.baseline
    )
    final = ids.assign_ids(
        raw.candidates, winners, provider, output_path, feature_type=FEATURE_TYPE
    )
    return AssignIdsOutput(final, fallout)


def metrics(
    raw: RawInputs,
    prepared: SetupOutput,
    matched: MatchingOutput,
    assigned: AssignIdsOutput,
    provider: str,
    load_optional_inputs: Callable[[], dict[str, object]] = dict,
) -> MetricsOutput:
    """Count matches and churn.

    ``load_optional_inputs`` returns the optional `MetricInputs` fields that this run
    has. It is called only when the stage runs.
    """
    inputs = MetricInputs(
        raw.candidates,
        raw.baseline,
        prepared.candidates,
        prepared.baseline,
        matched.matches,
        assigned.final,
        provider,
        **load_optional_inputs(),
    )
    rows = compute_metrics(inputs, METRICS)
    spark = raw.candidates.sparkSession
    return MetricsOutput(
        build_metrics_dataframe(spark, rows, THEME, FEATURE_TYPE, provider)
    )


def run(spark: SparkSession, config: RunConfig) -> None:
    """Run the places stages, skipping completed ones."""
    if FEATURE_TYPE not in config.input or FEATURE_TYPE not in config.baseline:
        raise ValueError(f"input and baseline need a {FEATURE_TYPE!r} entry")
    if config.model_jar is None:
        raise ValueError("places needs the modelJar param")
    # GraphFrames connected components needs a checkpoint directory.
    spark.sparkContext.setCheckpointDir(io.run_uri(config.output_path, "checkpoints"))
    # Set once for the whole job, so fresh and resumed runs plan stages alike.
    for key, value in SPATIAL_JOIN_CONF.items():
        spark.conf.set(key, value)
    raw = RawInputs(
        spark.read.parquet(*config.input[FEATURE_TYPE]),
        spark.read.parquet(*config.baseline[FEATURE_TYPE]),
    )
    prepared = run_stage(
        spark,
        config,
        FEATURE_TYPE,
        setup,
        raw,
        provider=config.provider,
        baseline_provider=config.baseline_provider,
        match_history=config.match_history,
        blocking_resolution=config.blocking_resolution or BLOCKING_RESOLUTION,
    )
    matched = run_stage(
        spark,
        config,
        FEATURE_TYPE,
        matching,
        prepared,
        model_jar=config.model_jar,
        output_path=config.output_path,
        embeddings_cache_table=config.embeddings_cache_table,
        output_debug=config.output_debug,
    )
    assigned = run_stage(
        spark,
        config,
        FEATURE_TYPE,
        assign_ids,
        raw,
        prepared,
        matched,
        config.provider,
        output_path=config.output_path,
    )
    run_stage(
        spark,
        config,
        FEATURE_TYPE,
        metrics,
        raw,
        prepared,
        matched,
        assigned,
        config.provider,
        load_optional_inputs=lambda: load_optional_metric_inputs(
            spark, config, FEATURE_TYPE
        ),
    )


class PlacesMatchingJob(SparkSedonaJob):
    """The four-stage place matching job."""

    def execute_job(self) -> None:
        """Run with the job params."""
        run(self.spark, RunConfig.for_run(self.spark, self.result.params))


if __name__ == "__main__":
    run_local(run, sys.argv[1], packages=PACKAGES)
