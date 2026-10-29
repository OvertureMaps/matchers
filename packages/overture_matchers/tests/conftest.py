# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Shared pytest fixtures: one local SparkSession for the whole test run."""

import os
import sys

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    # UDF workers must run under this venv's interpreter, which is
    # sys.executable here.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    session = (
        SparkSession.builder.master("local[1]")
        .appName("overture-matchers-tests")
        # The driver binds to loopback, because the hostname often fails to
        # resolve on laptops and VPNs.
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        # UTC keeps timestamp results independent of the machine's timezone.
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    yield session
    session.stop()
