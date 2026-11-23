# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Tests for the places session setup, against a stub session and the model jar."""

from unittest.mock import MagicMock

import pytest
from pyspark.sql import functions as F

from overture_matchers.places.runtime import (
    LIBPOSTAL_DATA_DIR_CONF,
    bind_jvm_text_functions,
    register_text_udfs,
    ship_libpostal_data_to_workers,
)


def _session(data_dir):
    spark = MagicMock()
    spark.conf.get.side_effect = lambda key, default: (
        data_dir if key == LIBPOSTAL_DATA_DIR_CONF else default
    )
    return spark


def test_libpostal_data_dir_is_required():
    with pytest.raises(ValueError, match=LIBPOSTAL_DATA_DIR_CONF):
        ship_libpostal_data_to_workers(_session(None))


@pytest.mark.parametrize(
    ("data_dir", "expected_data"),
    [
        ("/local/libpostal", None),
        ("file:///local/libpostal", None),
        ("s3://b/lp", "s3a://b/lp"),
        ("s3a://b/lp", "s3a://b/lp"),
    ],
)
def test_only_remote_libpostal_data_is_shipped(data_dir, expected_data):
    spark = _session(data_dir)
    ship_libpostal_data_to_workers(spark)
    if expected_data is None:
        spark.sparkContext.addFile.assert_not_called()
        spark.conf.set.assert_not_called()
    else:
        spark.conf.set.assert_called_once_with(LIBPOSTAL_DATA_DIR_CONF, expected_data)
        spark.sparkContext.addFile.assert_called_once_with(
            expected_data, recursive=True
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
