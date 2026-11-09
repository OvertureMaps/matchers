# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Run-folder I/O in Scala's layout: ``<output_path>/<name>/type=<table>/``."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from overture_spark.hadoop_fs import list_files

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

Checkpoint = Callable[["DataFrame", str], "DataFrame"]

MARKER_DIR = "_completed_stages"


def _join(output_path: str, *parts: str) -> str:
    return "/".join([output_path.rstrip("/"), *parts])


def output_uri(output_path: str, name: str, table: str) -> str:
    """URI of an output folder."""
    return _join(output_path, name, f"type={table}")


def write_file(spark: SparkSession, uri: str, text: str) -> None:
    """Write a small text file through the driver, replacing any file there."""
    if uri.startswith("s3://"):
        uri = "s3a://" + uri[len("s3://") :]
    path = spark._jvm.org.apache.hadoop.fs.Path(uri)
    fs = path.getFileSystem(spark._jsc.hadoopConfiguration())
    stream = fs.create(path, True)
    try:
        stream.write(bytearray(text.encode("utf-8")))
    finally:
        stream.close()


def write_output(output_path: str, name: str, table: str, df: DataFrame) -> None:
    """Write ``df`` as zstd parquet, with overwrite."""
    df.write.mode("overwrite").option("compression", "zstd").parquet(
        output_uri(output_path, name, table)
    )


def read_output(
    spark: SparkSession, output_path: str, name: str, table: str
) -> DataFrame:
    """Read an output folder."""
    return spark.read.parquet(output_uri(output_path, name, table))


def _marker_uri(output_path: str, stage: str) -> str:
    return _join(output_path, MARKER_DIR, stage)


def stage_is_done(spark: SparkSession, output_path: str, stage: str) -> bool:
    """Whether ``stage`` has a marker."""
    return bool(list_files(spark, _marker_uri(output_path, stage)))


def mark_stage_done(spark: SparkSession, output_path: str, stage: str) -> None:
    """Write the marker that makes reruns skip ``stage``."""
    write_file(spark, _marker_uri(output_path, stage), "")


def checkpointer(spark: SparkSession, output_path: str, table: str) -> Checkpoint:
    """Build ``checkpoint(df, name)``, which writes ``df`` and returns a read of it.

    The read has no lineage back to ``df``, so later steps don't recompute it.
    """

    def checkpoint(df: DataFrame, name: str) -> DataFrame:
        write_output(output_path, name, table, df)
        return read_output(spark, output_path, name, table)

    return checkpoint
