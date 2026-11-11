# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Stage output types and ``run_stage``."""

from __future__ import annotations

from collections.abc import Callable
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


def _output_type(stage_fn: Callable[..., T]) -> type[T]:
    # Only the return annotation is resolved: get_type_hints would also resolve
    # parameter types, which are often imported only for type checking.
    hint = stage_fn.__annotations__["return"]
    return eval(hint, stage_fn.__globals__) if isinstance(hint, str) else hint


def run_stage(
    spark: SparkSession,
    config: RunConfig,
    feature_type: str,
    stage_fn: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """Run ``stage_fn`` unless its marker exists, and return reads of its outputs.

    The stage marker gets its name from ``stage_fn.__name__`` and ``feature_type``,
    so each feature type in a run has its own marker. The output folders
    get their names from the fields of ``stage_fn``'s return type. The marker is
    written only after every output.
    """
    stage = stage_fn.__name__
    output_type = _output_type(stage_fn)
    if not io.stage_is_done(spark, config.output_path, stage, feature_type):
        outputs = stage_fn(*args, **kwargs)
        for name, df in outputs._asdict().items():
            io.write_output(config.output_path, name, feature_type, df)
        io.mark_stage_done(spark, config.output_path, stage, feature_type)
        spark.catalog.clearCache()
    return output_type(
        *(
            io.read_output(spark, config.output_path, name, feature_type)
            for name in output_type._fields
        )
    )
