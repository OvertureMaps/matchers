# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Tests for the places JVM text UDFs, against a stub session and the model jar."""

from unittest.mock import MagicMock

import pytest
from pyspark.sql import functions as F

from overture_matchers.places.runtime import (
    bind_jvm_text_functions,
    bind_unique_name_h3_bin_udf,
    register_text_udfs,
    register_unique_name_h3_bin_udf,
)


def test_register_text_udfs_registers_normalizer_tokenizer_and_coreword():
    spark = MagicMock()
    jvm = spark.sparkContext._jvm
    stream = MagicMock()
    stream.readAllBytes.return_value = b"bloom-bytes"
    jvm.org.apache.spark.util.Utils.classForName.return_value.getResourceAsStream.return_value = stream
    register_text_udfs(spark)
    names = [c.args[0] for c in spark.udf.registerJavaFunction.call_args_list]
    assert names == ["normalize_text", "tokenize_text"]
    stream.close.assert_called_once_with()
    jvm.org.overturemaps.places.features.BloomFeatures.registerCoreword.assert_called_once_with(
        spark._jsparkSession, "strip_background_tokens", b"bloom-bytes"
    )


def test_register_text_udfs_fails_when_coreword_filter_is_missing():
    spark = MagicMock()
    jvm = spark.sparkContext._jvm
    jvm.org.apache.spark.util.Utils.classForName.return_value.getResourceAsStream.return_value = None
    with pytest.raises(FileNotFoundError, match="coreword_bg_filter.bin"):
        register_text_udfs(spark)


def test_bind_jvm_text_functions_builds_columns(spark):
    text_functions = bind_jvm_text_functions()
    assert text_functions.normalize_text(F.lit("a")) is not None
    assert text_functions.tokenize_text(F.lit("a")) is not None
    assert (
        text_functions.strip_background_tokens(
            F.array(F.lit("a")), F.lit(40.7128), F.lit(-74.006)
        )
        is not None
    )


def test_register_unique_name_h3_bin_udf_registers_bloom_feature():
    spark = MagicMock()
    jvm = spark.sparkContext._jvm
    register_unique_name_h3_bin_udf(spark, b"ge2-bytes", b"ge10-bytes")
    jvm.org.overturemaps.places.features.BloomFeatures.registerUniqueName.assert_called_once_with(
        spark._jsparkSession, "unique_name_h3_bin", b"ge2-bytes", b"ge10-bytes"
    )


def test_bind_unique_name_h3_bin_udf_builds_column(spark):
    compute_bin = bind_unique_name_h3_bin_udf()
    assert compute_bin(F.lit("cafe"), F.lit(40.7128), F.lit(-74.006)) is not None


@pytest.mark.normalizer
def test_jvm_text_functions_against_model_jar(spark):
    register_text_udfs(spark)
    text_functions = bind_jvm_text_functions()
    cleaned = text_functions.normalize_text(F.lit("Joe's New York"))
    tokens = text_functions.tokenize_text(cleaned)
    stripped = text_functions.strip_background_tokens(
        tokens, F.lit(40.758), F.lit(-73.9855)
    )
    row = (
        spark.range(1)
        .select(
            cleaned.alias("clean"),
            tokens.alias("tokens"),
            stripped.alias("stripped"),
        )
        .first()
    )
    assert row.clean == "joe s new york"
    assert row.tokens == ["joe", "new", "york"]
    assert row.stripped == ["joe"]


@pytest.mark.normalizer
def test_jvm_unique_name_h3_bin_against_model_jar(spark):
    cell = spark.sql("SELECT ST_H3CellIDs(ST_Point(0.0, 0.0), 5, false)[0]").first()[0]

    def serialize_filter(names):
        keys = [(f"5:{cell}:{name}",) for name in names]
        bloom = (
            spark.createDataFrame(keys, ["k"])._jdf.stat().bloomFilter("k", 100, 0.001)
        )
        out = spark._jvm.java.io.ByteArrayOutputStream()
        bloom.writeTo(out)
        return bytes(out.toByteArray())

    register_unique_name_h3_bin_udf(
        spark, serialize_filter(["blue cafe", "rare"]), serialize_filter(["blue cafe"])
    )
    compute_bin = bind_unique_name_h3_bin_udf()
    rows = spark.createDataFrame(
        [
            ("  Blue Cafe ", 0.0, 0.0),
            ("rare", 0.0, 0.0),
            ("other", 0.0, 0.0),
            (" \t", 0.0, 0.0),
            ("blue cafe", None, 0.0),
        ],
        "name string, latitude double, longitude double",
    )
    bins = [
        row.bin
        for row in rows.select(
            compute_bin(F.col("name"), F.col("latitude"), F.col("longitude")).alias(
                "bin"
            )
        ).collect()
    ]
    assert bins == [2, 1, 0, 0, 0]
