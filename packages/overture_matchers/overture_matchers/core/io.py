# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Run-folder I/O in Scala's layout: ``<output_path>/<name>/type=<table>/``."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from overture_spark.hadoop_fs import delete_prefix, list_files, read_text

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

Checkpoint = Callable[["DataFrame", str], "DataFrame"]

MARKER_DIR = "_completed_stages"


def s3a(uri: str) -> str:
    """``uri`` with ``s3://`` read as ``s3a://``, the only scheme every platform maps."""
    return "s3a://" + uri[len("s3://") :] if uri.startswith("s3://") else uri


def run_uri(output_path: str, *parts: str) -> str:
    """URI of ``parts`` inside the run folder."""
    return "/".join([s3a(output_path).rstrip("/"), *parts])


def output_uri(output_path: str, name: str, table: str) -> str:
    """URI of an output folder."""
    return run_uri(output_path, name, f"type={table}")


def write_file(spark: SparkSession, uri: str, text: str) -> None:
    """Write a small text file through the driver, replacing any file there."""
    path = spark._jvm.org.apache.hadoop.fs.Path(s3a(uri))
    fs = path.getFileSystem(spark._jsc.hadoopConfiguration())
    stream = fs.create(path, True)
    try:
        stream.write(bytearray(text.encode("utf-8")))
    finally:
        stream.close()


def read_tables(spark: SparkSession, paths: list[str]) -> DataFrame:
    """Read one table's parquet paths as a single DataFrame."""
    return spark.read.parquet(*map(s3a, paths))


def read_baseline(spark: SparkSession, paths: list[str]) -> DataFrame:
    """Read a baseline table plus its soft deletes, as Scala's ``loadInput`` does.

    Each path's deletes sit at ``<path without .parquet>_deletes.parquet``. Deleted
    records whose ``id`` and ``provider`` are still active are dropped.
    """
    active = read_tables(spark, paths)
    deletes_paths = [
        uri
        for uri in (p.removesuffix(".parquet") + "_deletes.parquet" for p in paths)
        if list_files(spark, uri)
    ]
    if not deletes_paths:
        return active
    deletes = read_tables(spark, deletes_paths).join(
        active.select("id", "provider"), ["id", "provider"], "left_anti"
    )
    return active.unionByName(deletes)


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
    return run_uri(output_path, MARKER_DIR, stage)


def stage_is_done(spark: SparkSession, output_path: str, stage: str) -> bool:
    """Whether ``stage`` has a marker."""
    return bool(list_files(spark, _marker_uri(output_path, stage)))


def mark_stage_done(spark: SparkSession, output_path: str, stage: str) -> None:
    """Write the marker that makes reruns skip ``stage``.

    The marker lists the stages already done, which are the ones it was built on.
    """
    done = [uri.rsplit("/", 1)[1] for uri in _marker_uris(spark, output_path)]
    write_file(spark, _marker_uri(output_path, stage), "\n".join(done))


def clear_stages_built_on(spark: SparkSession, output_path: str, stage: str) -> None:
    """Delete the markers of stages built on an earlier run of ``stage``."""
    for uri in _marker_uris(spark, output_path):
        if stage in read_text(spark, uri).splitlines():
            delete_prefix(spark, uri)


def _marker_uris(spark: SparkSession, output_path: str) -> list[str]:
    return list_files(spark, run_uri(output_path, MARKER_DIR))


def checkpointer(spark: SparkSession, output_path: str, table: str) -> Checkpoint:
    """Build ``checkpoint(df, name)``, which writes ``df`` and returns a read of it.

    The read has no lineage back to ``df``, so later steps don't recompute it.
    """

    def checkpoint(df: DataFrame, name: str) -> DataFrame:
        write_output(output_path, name, table, df)
        return read_output(spark, output_path, name, table)

    return checkpoint
