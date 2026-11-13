# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Places matching job.

Run locally with ``python -m overture_matchers.places.job <params.json>``.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from overture_spark.hadoop_fs import list_files
from overture_spark.job import SparkSedonaJob
from pyspark import StorageLevel

from overture_matchers import ids
from overture_matchers.core import io
from overture_matchers.core.config import RunConfig
from overture_matchers.core.local import run_local
from overture_matchers.core.stages import (
    AssignIdsOutput,
    MatchingOutput,
    MetricsOutput,
    RawInputs,
    SetupOutput,
    run_stage,
)
from overture_matchers.metrics import MetricInputs, to_frame
from overture_matchers.metrics import compute as compute_metrics
from overture_matchers.places import model_jar
from overture_matchers.places.metrics import METRICS, sample_score_files
from overture_matchers.places.prepare import (
    add_base_ids,
    baseline_provider_column,
    prepare_baseline,
    prepare_candidates,
)
from overture_matchers.places.runtime import (
    jvm_text_fns,
    register_text_udfs,
    ship_model_and_libpostal_data_to_workers,
)
from overture_matchers.places.selection import select_winners

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)

FEATURE_TYPE = "place"
THEME = "places"
# GraphFrames build. Its jar also carries the Python API.
PACKAGES = ("graphframes:graphframes:0.8.3-spark3.4-s_2.12",)


def setup(
    raw: RawInputs,
    provider: str,
    model_jar: str,
    baseline_provider: str | None = None,
    match_history: bool = True,
) -> SetupOutput:
    """Normalize candidates and baseline, and attach history matches if asked.

    Args:
        raw: Candidates and baseline as read.
        provider: Stamped on every candidate.
        model_jar: Path of the ``places-matcher-model`` jar. Its classes must
            be on the Spark classpath.
        baseline_provider: Provider for a baseline without a ``provider``
            column.
        match_history: Whether to fill ``base_ids`` from the baseline. Only
            setup reads this flag.

    Returns:
        Candidates with ``base_ids``, and the baseline with ``active``.
    """
    spark = raw.candidates.sparkSession
    # This fails fast before shipping if the baseline has a `provider` column
    # and baseline_provider is set, or if it has neither.
    baseline_provider_col = baseline_provider_column(raw.baseline, baseline_provider)
    ship_model_and_libpostal_data_to_workers(spark, model_jar)
    register_text_udfs(spark)
    text = jvm_text_fns(model_jar)
    baseline = prepare_baseline(raw.baseline, baseline_provider_col, text)
    if match_history:
        # The baseline feeds both the history join and the stage output.
        baseline = baseline.persist(StorageLevel.MEMORY_AND_DISK)
    candidates = add_base_ids(
        prepare_candidates(raw.candidates, provider, text),
        baseline,
        match_history,
    )
    return SetupOutput(candidates, baseline)


def matching(prepared: SetupOutput, output_path: str | None = None) -> MatchingOutput:
    """Generate pairs, score them and cluster the matches."""
    raise NotImplementedError("places matching is not ported yet")


def assign_ids(
    raw: RawInputs,
    prepared: SetupOutput,
    matched: MatchingOutput,
    provider: str,
    output_path: str | None = None,
) -> AssignIdsOutput:
    """Pick cluster winners and assign GERS IDs."""
    winners, fallout = select_winners(matched.matches, prepared.baseline)
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
    **optional: object,
) -> MetricsOutput:
    """Count matches and churn.

    ``optional`` holds the `MetricInputs` fields that are not always there: ``blocking``,
    ``scores``, ``score_sample``, ``feature_order``, ``reference`` and ``previous``.
    """
    inputs = MetricInputs(
        raw.candidates,
        raw.baseline,
        prepared.candidates,
        prepared.baseline,
        matched.matches,
        assigned.final,
        provider,
        **optional,
    )
    rows = compute_metrics(inputs, METRICS)
    spark = raw.candidates.sparkSession
    return MetricsOutput(to_frame(spark, rows, THEME, FEATURE_TYPE, provider))


def _read_if_present(
    spark: SparkSession, config: RunConfig, name: str
) -> DataFrame | None:
    uri = io.output_uri(config.output_path, name, FEATURE_TYPE)
    return (
        io.read_output(spark, config.output_path, name, FEATURE_TYPE)
        if list_files(spark, uri)
        else None
    )


def _try(description: str, load):
    """The result of ``load()``, or None, with a warning, if it raises."""
    try:
        return load()
    except Exception:
        logger.exception("Skipping %s", description)
        return None


def _optional_metric_inputs(
    spark: SparkSession, config: RunConfig
) -> dict[str, object]:
    """Load the metric inputs that a run may lack. Anything missing is left out."""
    scores = _read_if_present(spark, config, "scores")
    found = {
        "blocking": _read_if_present(spark, config, "blocking"),
        "scores": scores,
        "score_sample": None if scores is None else sample_score_files(spark, scores),
        "feature_order": _try(
            "feature stats",
            lambda: model_jar.read_feature_order(spark, config.model_jar),
        ),
        "reference": _try(
            "reference z-scores",
            lambda: model_jar.read_reference_dataset(
                spark, config.model_jar, config.output_path
            ),
        ),
        "previous": _try(
            "previous release z-scores",
            lambda: spark.read.parquet(config.previous_metrics_path),
        )
        if config.previous_metrics_path
        else None,
    }
    return {k: v for k, v in found.items() if v is not None}


def run(spark: SparkSession, config: RunConfig) -> None:
    """Run the places stages, skipping completed ones."""
    if FEATURE_TYPE not in config.input or FEATURE_TYPE not in config.baseline:
        raise ValueError(f"input and baseline need a {FEATURE_TYPE!r} entry")
    if config.model_jar is None:
        raise ValueError("places needs the modelJar param")
    # GraphFrames connected components needs a checkpoint directory.
    spark.sparkContext.setCheckpointDir(io.run_uri(config.output_path, "checkpoints"))
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
        model_jar=config.model_jar,
        baseline_provider=config.baseline_provider,
        match_history=config.match_history,
    )
    matched = run_stage(
        spark, config, FEATURE_TYPE, matching, prepared, output_path=config.output_path
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
        **_optional_metric_inputs(spark, config),
    )


class PlacesMatchingJob(SparkSedonaJob):
    """The four-stage place matching job."""

    def execute_job(self) -> None:
        """Run with the job params."""
        run(self.spark, RunConfig.for_run(self.spark, self.result.params))


if __name__ == "__main__":
    run_local(run, sys.argv[1], packages=PACKAGES)
