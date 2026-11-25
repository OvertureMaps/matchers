# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Run a theme's job on a local Sedona session from a JSON params file."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse
from urllib.request import url2pathname

from overture_spark import SparkPlatform, getSparkSedonaSession
from overture_spark.job import is_unset

from overture_matchers.core.config import RunConfig, rewrite_s3_to_s3a

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def _resolve_spark_jars(model_jar: object) -> dict[str, str]:
    if is_unset(model_jar):
        return {}
    jar = str(model_jar)
    parsed = urlparse(jar)
    if parsed.scheme in ("", "file"):
        local_jar = url2pathname(parsed.path) if parsed.scheme == "file" else jar
        if not os.path.isfile(local_jar):
            raise FileNotFoundError(jar)
    return {"spark.jars": rewrite_s3_to_s3a(jar)}


def run_local(
    run: Callable[[SparkSession, RunConfig], None],
    params_file: str,
    spark: SparkSession | None = None,
    packages: Sequence[str] = (),
) -> None:
    """Run a theme's ``run`` from a JSON params file, on a local Sedona session."""
    params = json.loads(Path(params_file).read_text())
    if spark is None:
        spark = getSparkSedonaSession(
            spark_platform=SparkPlatform.LOCAL,
            extra_spark_conf=_resolve_spark_jars(params.get("modelJar")),
            extra_packages=list(packages),
        )
    run(spark, RunConfig.for_run(spark, params))
