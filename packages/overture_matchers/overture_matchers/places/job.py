# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Places matching job.

Run locally with ``python -m overture_matchers.places.job <params.json>``.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from overture_spark.job import SparkSedonaJob
from pyspark import StorageLevel

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
from overture_matchers.places.prepare import (
    add_base_ids,
    baseline_provider_column,
    prepare_baseline,
    prepare_candidates,
    register_text_udfs,
    ship_resources,
)

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

FEATURE_TYPE = "place"
# Scala's GraphFrames build. Its jar also carries the Python API.
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
    baseline_provider_column(raw.baseline, baseline_provider)  # fail before shipping
    ship_resources(spark, model_jar)
    register_text_udfs(spark)
    # The baseline feeds both the history join and the stage output.
    baseline = prepare_baseline(raw.baseline, model_jar, baseline_provider).persist(
        StorageLevel.MEMORY_AND_DISK
    )
    candidates = add_base_ids(
        prepare_candidates(raw.candidates, provider, model_jar),
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
    output_path: str | None = None,
) -> AssignIdsOutput:
    """Pick cluster winners and assign GERS IDs."""
    raise NotImplementedError("places assign_ids is not ported yet")


def metrics(
    raw: RawInputs,
    prepared: SetupOutput,
    matched: MatchingOutput,
    ids: AssignIdsOutput,
) -> MetricsOutput:
    """Count matches and churn."""
    raise NotImplementedError("places metrics is not ported yet")


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
    ids = run_stage(
        spark,
        config,
        FEATURE_TYPE,
        assign_ids,
        raw,
        prepared,
        matched,
        output_path=config.output_path,
    )
    run_stage(spark, config, FEATURE_TYPE, metrics, raw, prepared, matched, ids)


class PlacesMatchingJob(SparkSedonaJob):
    """The four-stage place matching job."""

    def execute_job(self) -> None:
        """Run with the job params."""
        run(self.spark, RunConfig.for_run(self.spark, self.result.params))


if __name__ == "__main__":
    run_local(run, sys.argv[1], packages=PACKAGES)
