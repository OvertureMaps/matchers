# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Places matching job.

Run locally with ``python -m overture_matchers.places.job <params.json>``.
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


def setup(raw: RawInputs, match_history: bool = True) -> SetupOutput:
    """Normalize candidates and baseline, and attach history matches if asked."""
    raise NotImplementedError("places setup is not ported yet")


def matching(
    prepared: SetupOutput, checkpoint: io.Checkpoint = no_checkpoint
) -> MatchingOutput:
    """Generate pairs, score them and cluster the matches."""
    raise NotImplementedError("places matching is not ported yet")


def assign_ids(
    raw: RawInputs,
    prepared: SetupOutput,
    matched: MatchingOutput,
    checkpoint: io.Checkpoint = no_checkpoint,
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
    if TABLE not in config.input or TABLE not in config.baseline:
        raise ValueError(f"input and baseline need a {TABLE!r} entry")
    if config.model_jar is None:
        raise ValueError("places needs the modelJar param")
    # GraphFrames connected components needs a checkpoint directory.
    spark.sparkContext.setCheckpointDir(io.run_uri(config.output_path, "checkpoints"))
    raw = RawInputs(
        spark.read.parquet(*config.input[TABLE]),
        spark.read.parquet(*config.baseline[TABLE]),
    )
    checkpoint = io.checkpointer(spark, config.output_path, TABLE)
    prepared = run_stage(
        spark, config, TABLE, setup, raw, match_history=config.match_history
    )
    matched = run_stage(spark, config, TABLE, matching, prepared, checkpoint=checkpoint)
    ids = run_stage(
        spark, config, TABLE, assign_ids, raw, prepared, matched, checkpoint=checkpoint
    )
    run_stage(spark, config, TABLE, metrics, raw, prepared, matched, ids)


class PlacesMatchingJob(SparkSedonaJob):
    """The four-stage place matching job."""

    def execute_job(self):
        """Run with the job params."""
        run(self.spark, RunConfig.for_run(self.spark, self.result.params))


if __name__ == "__main__":
    run_local(run, sys.argv[1])
