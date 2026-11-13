# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Spark session setup for the places stages: shipping the model jar and libpostal data to workers, and binding the jar's text functions.

The column logic in ``prepare`` takes its text functions as arguments, so it
runs without the jar. Everything that needs the jar or the libpostal data lives here.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

from overture_matchers.places.background_tokens import strip_background_tokens_udf
from overture_matchers.places.prepare import TextFns

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

LIBPOSTAL_DATA_DIR_CONF = "spark.sedona.libpostal.dataDir"
_NORMALIZE_UDF = "normalize_text"
_TOKENIZE_UDF = "tokenize_text"


def register_text_udfs(spark: SparkSession) -> None:
    """Register ``PlaceTextNormalizer`` as the JVM UDFs ``normalize_text`` and ``tokenize_text``."""
    udfs = "org.overturemaps.places.text"
    spark.udf.registerJavaFunction(
        _NORMALIZE_UDF, f"{udfs}.NormalizeTextUdf", StringType()
    )
    spark.udf.registerJavaFunction(
        _TOKENIZE_UDF, f"{udfs}.TokenizeTextUdf", ArrayType(StringType(), False)
    )


def ship_model_and_libpostal_data_to_workers(
    spark: SparkSession, model_jar: str
) -> None:
    """Send the model jar and the libpostal data to the workers.

    Args:
        spark: The session. ``spark.sedona.libpostal.dataDir`` must name the
            unpacked v1.0.0 libpostal data.
        model_jar: Path of the ``places-matcher-model`` jar.

    Raises:
        ValueError: The libpostal data dir is not set. Sedona's default
            model would change addresses outside the US.
        FileNotFoundError: A local ``model_jar`` does not exist.
    """
    if "://" not in model_jar and not os.path.isfile(model_jar):
        raise FileNotFoundError(model_jar)
    data_dir = spark.conf.get(LIBPOSTAL_DATA_DIR_CONF, None)
    if not data_dir:
        raise ValueError(f"set {LIBPOSTAL_DATA_DIR_CONF} to the libpostal v1.0.0 data")
    spark.sparkContext.addFile(model_jar)
    # Local data dirs work as-is; only remote data needs shipping.
    if "://" in data_dir and not data_dir.startswith("file:"):
        spark.sparkContext.addFile(data_dir, recursive=True)


def jvm_text_fns(model_jar: str) -> TextFns:
    """Text functions backed by the model jar.

    Args:
        model_jar: Path of the ``places-matcher-model`` jar. Its file name is
            where workers find it after ``ship_model_and_libpostal_data_to_workers``.
            ``register_text_udfs`` must have run before the columns are built.
    """
    jar_name = os.path.basename(model_jar)
    return TextFns(
        normalize=lambda text: F.call_udf(_NORMALIZE_UDF, text),
        tokenize=lambda text: F.call_udf(_TOKENIZE_UDF, text),
        strip_background=lambda tokens, cell: strip_background_tokens_udf(
            jar_name, tokens, cell
        ),
    )
