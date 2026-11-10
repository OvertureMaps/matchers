# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Stage output types and ``run_stage``."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, ParamSpec, TypeVar

from overture_matchers.core import io
from overture_matchers.core.config import RunConfig

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

P = ParamSpec("P")
T = TypeVar("T", bound=NamedTuple)


class RawInputs(NamedTuple):
    """Candidates and baseline as read."""

    candidates: DataFrame
    baseline: DataFrame


class SetupOutput(NamedTuple):
    """Candidates and baseline after setup."""

    candidates: DataFrame
    baseline: DataFrame


class MatchingOutput(NamedTuple):
    """Match results."""

    matches: DataFrame


class AssignIdsOutput(NamedTuple):
    """Records with IDs, and candidates folded into another record."""

    final: DataFrame
    fallout: DataFrame


class MetricsOutput(NamedTuple):
    """Run metrics."""

    metrics: DataFrame


def no_checkpoint(df: DataFrame, name: str) -> DataFrame:
    """A checkpoint that writes nothing."""
    return df


def _output_type(stage_fn: Callable[..., T]) -> type[T]:
    # Only the return annotation is resolved: get_type_hints would also resolve
    # parameter types, which are often imported only for type checking.
    hint = stage_fn.__annotations__["return"]
    return eval(hint, stage_fn.__globals__) if isinstance(hint, str) else hint


def run_stage(
    spark: SparkSession,
    config: RunConfig,
    table: str,
    stage_fn: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """Run ``stage_fn`` unless its marker exists, and return reads of its outputs.

    ``stage_fn.__name__`` names the marker, and its return annotation names the
    outputs. The marker is written only after every output.
    """
    stage = stage_fn.__name__
    output_type = _output_type(stage_fn)
    if not io.stage_is_done(spark, config.output_path, stage):
        outputs = stage_fn(*args, **kwargs)
        for name, df in outputs._asdict().items():
            io.write_output(config.output_path, name, table, df)
        io.mark_stage_done(spark, config.output_path, stage)
        spark.catalog.clearCache()
    return output_type(
        *(
            io.read_output(spark, config.output_path, name, table)
            for name in output_type._fields
        )
    )


def run_local(
    run: Callable[[SparkSession, RunConfig], None],
    params_file: str,
    spark: SparkSession | None = None,
) -> None:
    """Run a theme's ``run`` from a JSON params file, on a local Sedona session."""
    if spark is None:
        from overture_spark import SparkPlatform, getSparkSedonaSession

        spark = getSparkSedonaSession(spark_platform=SparkPlatform.LOCAL)
    params = json.loads(Path(params_file).read_text())
    run(spark, RunConfig.for_run(spark, params))
