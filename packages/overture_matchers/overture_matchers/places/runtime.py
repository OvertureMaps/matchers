# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Spark session setup for the places stages: registering the model jar's JVM UDFs.

The column logic in ``prepare`` and ``scoring`` takes its JVM-backed functions
as arguments, so it runs without the jar. Everything that needs the jar on the
Spark classpath lives here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

from overture_matchers.places.prepare import TextFunctions

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyspark.sql import Column, SparkSession

_BLOOM_FEATURES_CLASS = "org.overturemaps.places.features.BloomFeatures"
_COREWORD_FILTER_RESOURCE = "/config/places/coreword_bg_filter.bin"
_NORMALIZE_TEXT_UDF = "normalize_text"
_TOKENIZE_TEXT_UDF = "tokenize_text"
_STRIP_BACKGROUND_TOKENS_UDF = "strip_background_tokens"
_UNIQUE_NAME_H3_BIN_UDF = "unique_name_h3_bin"


def register_text_udfs(spark: SparkSession) -> None:
    """Register the model jar's text normalizer, tokenizer and background-token UDFs."""
    udfs = "org.overturemaps.places.text"
    spark.udf.registerJavaFunction(
        _NORMALIZE_TEXT_UDF, f"{udfs}.NormalizeTextUdf", StringType()
    )
    spark.udf.registerJavaFunction(
        _TOKENIZE_TEXT_UDF, f"{udfs}.TokenizeTextUdf", ArrayType(StringType(), False)
    )
    jvm = spark.sparkContext._jvm
    bloom_class = jvm.org.apache.spark.util.Utils.classForName(
        _BLOOM_FEATURES_CLASS, True, False
    )
    stream = bloom_class.getResourceAsStream(_COREWORD_FILTER_RESOURCE)
    if stream is None:
        raise FileNotFoundError(
            f"{_COREWORD_FILTER_RESOURCE} not found on the Spark classpath"
        )
    try:
        filter_bytes = stream.readAllBytes()
    finally:
        stream.close()
    jvm.org.overturemaps.places.features.BloomFeatures.registerCoreword(
        spark._jsparkSession, _STRIP_BACKGROUND_TOKENS_UDF, filter_bytes
    )


def register_unique_name_h3_bin_udf(
    spark: SparkSession, ge2_filter_bytes: bytes, ge10_filter_bytes: bytes
) -> None:
    """Register the model jar's resolution-5 H3 name-repetition bin UDF."""
    jvm = spark.sparkContext._jvm
    jvm.org.overturemaps.places.features.BloomFeatures.registerUniqueName(
        spark._jsparkSession,
        _UNIQUE_NAME_H3_BIN_UDF,
        ge2_filter_bytes,
        ge10_filter_bytes,
    )


def bind_unique_name_h3_bin_udf() -> Callable[[Column, Column, Column], Column]:
    """Bind the JVM UDF registered by ``register_unique_name_h3_bin_udf``."""
    return lambda name, latitude, longitude: F.call_udf(
        _UNIQUE_NAME_H3_BIN_UDF, name, latitude, longitude
    )


def bind_jvm_text_functions() -> TextFunctions:
    """Text functions backed by the JVM UDFs registered by ``register_text_udfs``."""
    return TextFunctions(
        normalize_text=lambda text: F.call_udf(_NORMALIZE_TEXT_UDF, text),
        tokenize_text=lambda text: F.call_udf(_TOKENIZE_TEXT_UDF, text),
        strip_background_tokens=lambda tokens, latitude, longitude: F.call_udf(
            _STRIP_BACKGROUND_TOKENS_UDF, tokens, latitude, longitude
        ),
    )
