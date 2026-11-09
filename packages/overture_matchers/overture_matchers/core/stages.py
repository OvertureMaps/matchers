# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Stage output types and ``run_stage``, which runs a stage once per run folder.

A stage is a function from DataFrames to a NamedTuple of DataFrames, named for
its marker (``setup``, ``matching``, ``assign_ids``, ``metrics``). The stage
does no I/O. ``run_stage`` skips it when its marker exists, otherwise runs it,
writes each output field to the run folder, writes the marker, clears Spark's
cache, and returns lazy reads of the written outputs.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, ParamSpec, TypeVar, get_type_hints

from overture_matchers.core import io
from overture_matchers.core.config import RunConfig

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

P = ParamSpec("P")
T = TypeVar("T", bound=NamedTuple)


class RawInputs(NamedTuple):
    """Candidates and baseline as read from the provider's and Overture's data."""

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
    """Candidates with assigned IDs, and candidates folded into another record."""

    final: DataFrame
    fallout: DataFrame


class MetricsOutput(NamedTuple):
    """Run metrics."""

    metrics: DataFrame


def no_checkpoint(df: DataFrame, name: str) -> DataFrame:
    """A checkpoint that returns ``df`` unchanged, for runs that write nothing."""
    return df


def run_stage(
    spark: SparkSession,
    config: RunConfig,
    table: str,
    stage_fn: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """Run ``stage_fn`` unless its marker exists, and return its outputs.

    Every output is written to ``<output_path>/<field>/type=<table>/`` with
    overwrite, so a rerun replaces whatever a failed attempt left behind. The
    marker ``_completed_stages/<stage>`` is written after all outputs.

    Args:
        spark: Active session.
        config: The run's config; ``output_path`` is the run folder.
        table: Table name of the ``type=<table>`` output folders.
        stage_fn: Stage function; its ``__name__`` is the marker name and its
            return annotation is the NamedTuple of outputs.
        *args: Positional arguments for ``stage_fn``.
        **kwargs: Keyword arguments for ``stage_fn``.

    Returns:
        The stage's output type, holding lazy reads of the written folders.
    """
    stage = stage_fn.__name__
    output_type = get_type_hints(stage_fn)["return"]
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
    """Run a theme's ``run`` from a JSON params file, on a local Sedona session.

    Args:
        run: The theme's ``run(spark, config)``.
        params_file: Path to a JSON object of job params.
        spark: Session to use; defaults to a local Sedona session.
    """
    if spark is None:
        from overture_spark import SparkPlatform, getSparkSedonaSession

        spark = getSparkSedonaSession(spark_platform=SparkPlatform.LOCAL)
    params = json.loads(Path(params_file).read_text())
    run(spark, RunConfig.for_run(spark, params))
