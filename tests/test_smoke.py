# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Checks that the package imports and a local SparkSession starts.

Matcher tests run on a local SparkSession, so this fails first when Java 17
or PySpark is missing from the environment.
"""

import overture_matchers


def test_local_spark_session_runs_a_query(spark):
    assert overture_matchers.__doc__
    assert spark.range(3).count() == 3
