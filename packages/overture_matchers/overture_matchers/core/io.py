# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Run-folder I/O: Scala's output layout, stage markers and text files.

Outputs sit at ``<output_path>/<name>/type=<table>/`` as zstd parquet, the
layout Scala's ``JobUtils.writeOutput`` produces, so Scala and Python run
folders read the same way. Markers sit at ``<output_path>/_completed_stages/``.
Small text files go through Spark's Hadoop FileSystem
(``overture_spark.hadoop_fs``), so one code path serves S3 and local runs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from overture_spark.hadoop_fs import list_files, read_text

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

Checkpoint = Callable[["DataFrame", str], "DataFrame"]

MARKER_DIR = "_completed_stages"


def _join(output_path: str, *parts: str) -> str:
    return "/".join([output_path.rstrip("/"), *parts])


def output_uri(output_path: str, name: str, table: str) -> str:
    """Location of the output ``name`` for ``table`` inside a run folder."""
    return _join(output_path, name, f"type={table}")


def exists(spark: SparkSession, uri: str) -> bool:
    """Whether a file, or a directory with at least one file, exists at ``uri``."""
    return bool(list_files(spark, uri))


def read_file(spark: SparkSession, uri: str) -> str:
    """The UTF-8 text of the file at ``uri``."""
    return read_text(spark, uri)


def write_file(spark: SparkSession, uri: str, text: str) -> None:
    """Write ``text`` to ``uri``, replacing any file there.

    For small files only: the bytes pass through the driver.
    """
    if uri.startswith("s3://"):
        uri = "s3a://" + uri[len("s3://") :]
    path = spark._jvm.org.apache.hadoop.fs.Path(uri)
    fs = path.getFileSystem(spark._jsc.hadoopConfiguration())
    stream = fs.create(path, True)
    try:
        stream.write(bytearray(text.encode("utf-8")))
    finally:
        stream.close()


def write_output(output_path: str, name: str, table: str, df: DataFrame) -> str:
    """Write ``df`` as zstd parquet, replacing earlier output. Returns its URI."""
    uri = output_uri(output_path, name, table)
    df.write.mode("overwrite").option("compression", "zstd").parquet(uri)
    return uri


def read_output(
    spark: SparkSession, output_path: str, name: str, table: str
) -> DataFrame:
    """A lazy read of the output ``name`` for ``table``."""
    return spark.read.parquet(output_uri(output_path, name, table))


def _marker_uri(output_path: str, stage: str) -> str:
    return _join(output_path, MARKER_DIR, stage)


def stage_is_done(spark: SparkSession, output_path: str, stage: str) -> bool:
    """Whether ``stage`` has finished in this run folder."""
    return exists(spark, _marker_uri(output_path, stage))


def mark_stage_done(spark: SparkSession, output_path: str, stage: str) -> None:
    """Write the marker that tells reruns to skip ``stage``."""
    write_file(spark, _marker_uri(output_path, stage), "")


def checkpointer(spark: SparkSession, output_path: str, table: str) -> Checkpoint:
    """Build ``checkpoint(df, name)`` for a run folder.

    The returned function writes ``df`` to ``<output_path>/<name>/type=<table>/``
    and returns a read of it. The read has no lineage back to ``df``, so later
    transformations do not recompute earlier ones.
    """

    def checkpoint(df: DataFrame, name: str) -> DataFrame:
        write_output(output_path, name, table, df)
        return read_output(spark, output_path, name, table)

    return checkpoint
