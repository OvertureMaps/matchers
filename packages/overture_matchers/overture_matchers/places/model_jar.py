# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Reads resources from the places model jar."""

from __future__ import annotations

import io
import json
import tempfile
import zipfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

CONFIG_RESOURCE = "places_matcher_model/metadata/config"
REFERENCE_RESOURCE = "places_matcher_model/combined_test_dataset.parquet"


def open_jar(spark: SparkSession, model_jar: str) -> zipfile.ZipFile:
    """Open the jar on the driver."""
    # binaryFile reads any Hadoop-readable URI without changing session state.
    content = spark.read.format("binaryFile").load(model_jar).first()["content"]
    return zipfile.ZipFile(io.BytesIO(content))


def read_feature_order(spark: SparkSession, model_jar: str) -> tuple[str, ...]:
    """The scoring features, in model order."""
    with open_jar(spark, model_jar) as jar:
        config = json.loads(jar.read(CONFIG_RESOURCE))
    return tuple(config["feature_order"])


def read_reference_dataset(
    spark: SparkSession, model_jar: str, output_path: str
) -> DataFrame | None:
    """The model's reference feature table, or None if the jar has none.

    The file is copied into the run folder first, so executors can read it.
    """
    with open_jar(spark, model_jar) as jar, tempfile.TemporaryDirectory() as tmp:
        if REFERENCE_RESOURCE not in jar.namelist():
            return None
        local = jar.extract(REFERENCE_RESOURCE, tmp)
        jvm = spark.sparkContext._jvm
        dest = jvm.org.apache.hadoop.fs.Path(f"{output_path}/{REFERENCE_RESOURCE}")
        fs = dest.getFileSystem(spark.sparkContext._jsc.hadoopConfiguration())
        fs.copyFromLocalFile(False, True, jvm.org.apache.hadoop.fs.Path(local), dest)
    return spark.read.parquet(dest.toString())
