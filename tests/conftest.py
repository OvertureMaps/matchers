# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Shared pytest fixtures: one local SparkSession for the whole test run."""

import os
import sys

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    # Python UDF workers must use the venv interpreter, not the system one.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    session = (
        SparkSession.builder.master("local[1]")
        .appName("overture-matchers-tests")
        # Loopback only: the machine's hostname often doesn't resolve on laptops or VPNs.
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()
