# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Checks that the package imports and a local SparkSession runs a query.

A missing Java 17 or PySpark fails here first.
"""

import overture_matchers


def test_local_spark_session_runs_a_query(spark):
    assert overture_matchers.__doc__
    assert spark.range(3).count() == 3
