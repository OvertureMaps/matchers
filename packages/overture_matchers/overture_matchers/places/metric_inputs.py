# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Loads the places metric inputs that a run may lack."""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from overture_matchers.core import io
from overture_matchers.places import model_jar

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

    from overture_matchers.core.config import RunConfig

logger = logging.getLogger(__name__)

# Sampled score files stop once they add up to this many bytes.
SAMPLE_BYTES = 200 * 1024 * 1024
SAMPLE_SEED = 42


def sample_score_files(spark: SparkSession, scores: DataFrame) -> DataFrame:
    """Read a random subset of the score files, about ``SAMPLE_BYTES`` in total.

    Scores are hash-partitioned by pair, so any subset of files is representative.
    Reads every file if the whole table is smaller than the target.
    """
    files = [f for f in scores.inputFiles() if f.endswith(".parquet")]
    random.Random(SAMPLE_SEED).shuffle(files)
    jvm, conf = spark._jvm, spark._jsc.hadoopConfiguration()
    chosen, total = [], 0
    for f in files:
        if total >= SAMPLE_BYTES:
            break
        path = jvm.org.apache.hadoop.fs.Path(f)
        total += path.getFileSystem(conf).getFileStatus(path).getLen()
        chosen.append(f)
    return spark.read.parquet(*(chosen or files))


def _try(description: str, load):
    """The result of ``load()``, or None, with a warning, if it raises."""
    try:
        return load()
    except Exception:
        logger.exception("Skipping %s", description)
        return None


def _reference(spark: SparkSession, config: RunConfig) -> DataFrame | None:
    reference = model_jar.read_reference_dataset(
        spark, config.model_jar, config.output_path
    )
    if reference is None:
        logger.warning("The model jar has no reference dataset; skipping z-scores")
    return reference


def load_optional_inputs(
    spark: SparkSession, config: RunConfig, feature_type: str
) -> dict[str, object]:
    """Load the metric inputs that a run may lack. Anything missing is left out."""
    scores = _try(
        "score metrics",
        lambda: io.read_output_if_present(
            spark, config.output_path, "scores", feature_type
        ),
    )
    found = {
        "blocking": _try(
            "blocking metrics",
            lambda: io.read_output_if_present(
                spark, config.output_path, "blocking", feature_type
            ),
        ),
        "scores": scores,
        "score_sample": None
        if scores is None
        else _try("feature stats", lambda: sample_score_files(spark, scores)),
        "feature_order": _try(
            "feature stats",
            lambda: model_jar.read_feature_order(spark, config.model_jar),
        ),
        "reference": _try("reference z-scores", lambda: _reference(spark, config)),
        "previous": _try(
            "previous release z-scores",
            lambda: spark.read.parquet(config.previous_metrics_path),
        )
        if config.previous_metrics_path
        else None,
    }
    return {k: v for k, v in found.items() if v is not None}
