# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Tests for the places session setup, against a stub session."""

from unittest.mock import MagicMock

import pytest
from pyspark.sql import functions as F

from overture_matchers.places.runtime import (
    LIBPOSTAL_DATA_DIR_CONF,
    jvm_text_fns,
    register_text_udfs,
    ship_model_and_libpostal_data_to_workers,
)


def _session(data_dir):
    spark = MagicMock()
    spark.conf.get.side_effect = lambda key, default: (
        data_dir if key == LIBPOSTAL_DATA_DIR_CONF else default
    )
    return spark


def test_missing_jar_fails_before_touching_the_session():
    spark = _session("/data")
    with pytest.raises(FileNotFoundError):
        ship_model_and_libpostal_data_to_workers(spark, "/no/such.jar")
    spark.sparkContext.addFile.assert_not_called()


def test_libpostal_data_dir_is_required():
    with pytest.raises(ValueError, match=LIBPOSTAL_DATA_DIR_CONF):
        ship_model_and_libpostal_data_to_workers(_session(None), "s3a://b/model.jar")


@pytest.mark.parametrize(
    ("data_dir", "shipped_data"),
    [
        ("/local/libpostal", False),
        ("file:///local/libpostal", False),
        ("s3a://b/lp", True),
    ],
)
def test_only_remote_data_is_shipped(data_dir, shipped_data):
    spark = _session(data_dir)
    ship_model_and_libpostal_data_to_workers(spark, "s3a://b/model.jar")
    calls = spark.sparkContext.addFile.call_args_list
    assert calls[0].args == ("s3a://b/model.jar",)
    assert len(calls) == 1 + shipped_data
    if shipped_data:
        assert calls[1].args == (data_dir,)
        assert calls[1].kwargs == {"recursive": True}


def test_register_text_udfs_names_both_functions():
    spark = MagicMock()
    register_text_udfs(spark)
    names = [c.args[0] for c in spark.udf.registerJavaFunction.call_args_list]
    assert names == ["normalize_text", "tokenize_text"]


def test_jvm_text_fns_build_columns_from_the_jar_basename(spark):
    text = jvm_text_fns("/some/dir/model.jar")
    assert text.normalize(F.lit("a")) is not None
    assert text.tokenize(F.lit("a")) is not None
