# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Reads resources from the places model jar."""

from __future__ import annotations

import io as _io
import json
import os
import zipfile
from typing import TYPE_CHECKING

from pyspark import SparkFiles

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

CONFIG_RESOURCE = "places_matcher_model/metadata/config"
REFERENCE_RESOURCE = "places_matcher_model/combined_test_dataset.parquet"


def _open(spark: SparkSession, model_jar: str) -> zipfile.ZipFile:
    # addFile fetches any Hadoop-readable URI to the driver's local file area.
    spark.sparkContext.addFile(model_jar)
    return zipfile.ZipFile(SparkFiles.get(os.path.basename(model_jar.rstrip("/"))))


def read_feature_order(spark: SparkSession, model_jar: str) -> tuple[str, ...]:
    """The scoring features, in model order."""
    with _open(spark, model_jar) as jar:
        config = json.loads(jar.read(CONFIG_RESOURCE))
    return tuple(config["feature_order"])


def read_reference_dataset(spark: SparkSession, model_jar: str) -> DataFrame:
    """The model's reference feature table. Raises KeyError if the jar has none."""
    import pyarrow.parquet as pq

    with _open(spark, model_jar) as jar:
        table = pq.read_table(_io.BytesIO(jar.read(REFERENCE_RESOURCE)))
    # Built on the driver: the unpacked file is not visible to executors.
    return spark.createDataFrame(table.to_pandas())
