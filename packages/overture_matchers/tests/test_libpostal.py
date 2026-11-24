# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Tests for shipping libpostal data to workers, against a stub session."""

from unittest.mock import MagicMock

import pytest

from overture_matchers.core.libpostal import (
    LIBPOSTAL_DATA_DIR_CONF,
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
