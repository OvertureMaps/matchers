# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Places matching job: setup, matching, assign_ids and metrics.

``PlacesMatchingJob`` runs on Spark and launches ``run``, which reads the inputs
and chains the four stages through ``run_stage``. The stages are stubs until
each is ported. Run locally with
``python -m overture_matchers.places.job <params.json>``.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from overture_spark.job import SparkSedonaJob

from overture_matchers.core import io
from overture_matchers.core.config import RunConfig
from overture_matchers.core.stages import (
    AssignIdsOutput,
    MatchingOutput,
    MetricsOutput,
    RawInputs,
    SetupOutput,
    no_checkpoint,
    run_local,
    run_stage,
)

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

TABLE = "place"


def setup(raw: RawInputs) -> SetupOutput:
    """Normalize candidates and baseline."""
    raise NotImplementedError("places setup is not ported yet")


def matching(
    prepared: SetupOutput, checkpoint: io.Checkpoint = no_checkpoint
) -> MatchingOutput:
    """Generate pairs, score them and cluster the matches."""
    raise NotImplementedError("places matching is not ported yet")


def assign_ids(
    raw: RawInputs, prepared: SetupOutput, matched: MatchingOutput
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
    """Run the places stages, resuming after the last completed one."""
    raw = RawInputs(
        spark.read.parquet(*config.input[TABLE]),
        spark.read.parquet(*config.baseline[TABLE]),
    )
    prepared = run_stage(spark, config, TABLE, setup, raw)
    matched = run_stage(
        spark,
        config,
        TABLE,
        matching,
        prepared,
        checkpoint=io.checkpointer(spark, config.output_path, TABLE),
    )
    ids = run_stage(spark, config, TABLE, assign_ids, raw, prepared, matched)
    run_stage(spark, config, TABLE, metrics, raw, prepared, matched, ids)


class PlacesMatchingJob(SparkSedonaJob):
    """The places match task: one job, one cluster, four stages."""

    def execute_job(self):
        """Build the config from the job params and run the stages."""
        run(self.spark, RunConfig.for_run(self.spark, self.result.params))


if __name__ == "__main__":
    run_local(run, sys.argv[1])
