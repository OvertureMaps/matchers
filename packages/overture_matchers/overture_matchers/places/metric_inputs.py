# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Loads optional places metric inputs."""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from overture_matchers.core import io
from overture_matchers.places import model_jar

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeVar

    from pyspark.sql import DataFrame, SparkSession

    from overture_matchers.core.config import RunConfig

    T = TypeVar("T")

logger = logging.getLogger(__name__)

# Sampled score files stop once they add up to this many bytes.
SAMPLE_BYTES = 200 * 1024 * 1024
SAMPLE_SEED = 42


def sample_score_files(spark: SparkSession, scores: DataFrame) -> DataFrame:
    """Read a random subset of the score files, about ``SAMPLE_BYTES`` in total.

    Scores are hash-partitioned by pair, so any subset of files is representative.
    Reads every file if the whole table is smaller than the target.
    """
    parquet_files = [
        file_uri for file_uri in scores.inputFiles() if file_uri.endswith(".parquet")
    ]
    random.Random(SAMPLE_SEED).shuffle(parquet_files)
    jvm, hadoop_conf = spark._jvm, spark._jsc.hadoopConfiguration()
    selected_files, accumulated_bytes = [], 0
    for file_uri in parquet_files:
        if accumulated_bytes >= SAMPLE_BYTES:
            break
        path = jvm.org.apache.hadoop.fs.Path(file_uri)
        accumulated_bytes += (
            path.getFileSystem(hadoop_conf).getFileStatus(path).getLen()
        )
        selected_files.append(file_uri)
    return spark.read.parquet(*(selected_files or parquet_files))


def _try_load_optional_input(description: str, loader: Callable[[], T]) -> T | None:
    """Return ``loader()``, or log a warning and return None if it raises."""
    try:
        return loader()
    except Exception:
        logger.exception("Skipping %s", description)
        return None


def _load_reference_dataset(spark: SparkSession, config: RunConfig) -> DataFrame | None:
    reference_dataset = model_jar.read_reference_dataset(
        spark, config.model_jar, config.output_path
    )
    if reference_dataset is None:
        logger.warning("The model jar has no reference dataset; skipping z-scores")
    return reference_dataset


def load_optional_metric_inputs(
    spark: SparkSession, config: RunConfig, feature_type: str
) -> dict[str, object]:
    """Load the metric inputs that a run may lack. Anything missing is left out."""
    scores = _try_load_optional_input(
        "score metrics",
        lambda: io.read_output_if_present(
            spark, config.output_path, "scores", feature_type
        ),
    )
    found = {
        "blocking_pairs": _try_load_optional_input(
            "blocking metrics",
            lambda: io.read_output_if_present(
                spark, config.output_path, "blocking", feature_type
            ),
        ),
        "scores": scores,
        "score_sample": None
        if scores is None
        else _try_load_optional_input(
            "feature stats", lambda: sample_score_files(spark, scores)
        ),
        "feature_order": _try_load_optional_input(
            "feature stats",
            lambda: model_jar.read_feature_order(spark, config.model_jar),
        ),
        "reference_dataset": _try_load_optional_input(
            "reference z-scores", lambda: _load_reference_dataset(spark, config)
        ),
        "previous_metrics": _try_load_optional_input(
            "previous release z-scores",
            lambda: spark.read.parquet(config.previous_metrics_path),
        )
        if config.previous_metrics_path
        else None,
    }
    return {key: value for key, value in found.items() if value is not None}
