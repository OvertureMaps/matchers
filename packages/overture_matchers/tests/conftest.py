# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Shared pytest fixtures: one local Sedona session for the whole test run."""

import os
import sys

import pytest
from overture_spark import SparkPlatform, getSparkSedonaSession

from overture_matchers.places.job import PACKAGES


@pytest.fixture(scope="session")
def spark():
    # UDF workers must run under this venv's interpreter, which is
    # sys.executable here.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    session = getSparkSedonaSession(
        spark_platform=SparkPlatform.LOCAL,
        extra_packages=list(PACKAGES),
        extra_spark_conf={
            "spark.master": "local[1]",
            "spark.app.name": "overture-matchers-tests",
            # The driver binds to loopback, because the hostname often fails to
            # resolve on laptops and VPNs.
            "spark.driver.bindAddress": "127.0.0.1",
            "spark.driver.host": "127.0.0.1",
            "spark.sql.shuffle.partitions": "1",
            "spark.ui.enabled": "false",
            # UTC keeps timestamp results independent of the machine's timezone.
            "spark.sql.session.timeZone": "UTC",
        },
    )
    yield session
    session.stop()
