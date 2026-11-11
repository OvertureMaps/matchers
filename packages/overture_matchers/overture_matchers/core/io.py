# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Run-folder I/O in Scala's layout: ``<output_path>/<name>/type=<feature_type>/``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from overture_spark.hadoop_fs import list_files, write_text
from pyspark.sql import DataFrame

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

MARKER_DIR = "_completed_stages"


def run_uri(output_path: str, *parts: str) -> str:
    """URI of ``parts`` inside the run folder."""
    return "/".join([output_path.rstrip("/"), *parts])


def output_uri(output_path: str, name: str, feature_type: str) -> str:
    """URI of an output folder."""
    return run_uri(output_path, name, f"type={feature_type}")


def write_output(output_path: str, name: str, feature_type: str, df: DataFrame) -> None:
    """Write ``df`` as zstd parquet, with overwrite."""
    df.write.mode("overwrite").option("compression", "zstd").parquet(
        output_uri(output_path, name, feature_type)
    )


def read_output(
    spark: SparkSession, output_path: str, name: str, feature_type: str
) -> DataFrame:
    """Read an output folder."""
    return spark.read.parquet(output_uri(output_path, name, feature_type))


def _marker_uri(output_path: str, stage: str) -> str:
    return run_uri(output_path, MARKER_DIR, stage)


def stage_is_done(spark: SparkSession, output_path: str, stage: str) -> bool:
    """Whether ``stage`` has a marker."""
    return bool(list_files(spark, _marker_uri(output_path, stage)))


def mark_stage_done(spark: SparkSession, output_path: str, stage: str) -> None:
    """Write the marker that makes reruns skip ``stage``."""
    write_text(spark, _marker_uri(output_path, stage), "")


def checkpoint(
    df: DataFrame, output_path: str | None, name: str, feature_type: str
) -> DataFrame:
    """Write ``df`` and return a read of it, which has no lineage back to ``df``.

    Returns ``df`` unchanged when ``output_path`` is None, so stages run in memory.
    """
    if output_path is None:
        return df
    write_output(output_path, name, feature_type, df)
    return read_output(df.sparkSession, output_path, name, feature_type)
