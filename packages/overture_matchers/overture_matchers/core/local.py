# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Run a theme's job on a local Sedona session from a JSON params file."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from overture_spark import SparkPlatform, getSparkSedonaSession

from overture_matchers.core.config import RunConfig

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def run_local(
    run: Callable[[SparkSession, RunConfig], None],
    params_file: str,
    spark: SparkSession | None = None,
    packages: Sequence[str] = (),
) -> None:
    """Run a theme's ``run`` from a JSON params file, on a local Sedona session."""
    spark = spark or getSparkSedonaSession(
        spark_platform=SparkPlatform.LOCAL, extra_packages=list(packages)
    )
    params = json.loads(Path(params_file).read_text())
    run(spark, RunConfig.for_run(spark, params))
