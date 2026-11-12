# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Shared pytest fixtures: one local Sedona session for the whole test run."""

import os
import sys

import pytest
from overture_spark import SparkPlatform, getSparkSedonaSession

from overture_matchers.places.job import PACKAGES

# Optional private inputs. Tests that need them skip when they are unset.
MODEL_JAR = os.environ.get("PLACES_MATCHER_MODEL_JAR")
LIBPOSTAL_DATA = os.environ.get("LIBPOSTAL_DATA_DIR")


def pytest_collection_modifyitems(items):
    # Set REQUIRE_PRIVATE_INPUTS where the inputs exist, so a missing one fails.
    if os.environ.get("REQUIRE_PRIVATE_INPUTS"):
        return
    skips = {
        "normalizer": (MODEL_JAR, "set PLACES_MATCHER_MODEL_JAR"),
        "libpostal": (LIBPOSTAL_DATA, "set LIBPOSTAL_DATA_DIR"),
    }
    for item in items:
        for marker, (value, reason) in skips.items():
            if marker in item.keywords and not value:
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="session")
def model_jar():
    return MODEL_JAR


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
            **({"spark.jars": MODEL_JAR} if MODEL_JAR else {}),
            **(
                {"spark.sedona.libpostal.dataDir": LIBPOSTAL_DATA}
                if LIBPOSTAL_DATA
                else {}
            ),
        },
    )
    yield session
    session.stop()
