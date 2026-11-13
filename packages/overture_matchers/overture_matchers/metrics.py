# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Run metrics shared by all themes, and the plumbing that turns metrics into rows.

A metric is a function from `MetricInputs` to a list of ``(metric_id, value,
sub_partition)`` tuples. A metric that raises is logged and contributes no rows.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING, NamedTuple

from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)

MetricRow = tuple[str, float, str]
Metric = Callable[["MetricInputs"], list[MetricRow]]

ROW_SCHEMA = StructType(
    [
        StructField("metric_id", StringType(), False),
        StructField("value", DoubleType(), False),
        StructField("timestamp", TimestampType(), False),
        StructField("theme", StringType(), False),
        StructField("type", StringType(), False),
        StructField("location_id", StringType(), False),
        StructField("primary_partition", StringType(), False),
        StructField("sub_partition", StringType(), False),
        StructField("metadata", StringType(), False),
        StructField("baseline_snapshot", StringType(), False),
    ]
)


class MetricInputs(NamedTuple):
    """Everything a metric can read.

    ``candidates`` and ``baseline`` are the raw inputs, ``prepared_*`` are setup's
    outputs, and ``results`` is the assigned output. The last six fields are optional:
    a metric that needs a missing one has no rows.
    """

    candidates: DataFrame
    baseline: DataFrame
    prepared_candidates: DataFrame
    prepared_baseline: DataFrame
    matches: DataFrame
    results: DataFrame
    provider: str
    blocking: DataFrame | None = None
    scores: DataFrame | None = None
    # A sample of ``scores`` that is small enough to profile feature by feature.
    score_sample: DataFrame | None = None
    feature_order: tuple[str, ...] = ()
    reference: DataFrame | None = None
    previous: DataFrame | None = None


def round_half_up(value: float, places: int) -> float:
    """Round the shortest decimal form of ``value``, so 0.125 at 2 places gives 0.13."""
    exp = Decimal(1).scaleb(-places)
    return float(Decimal(repr(value)).quantize(exp, rounding=ROUND_HALF_UP))


def percentage(part: float, whole: float) -> float:
    """``part`` as a percentage of ``whole``, or 0 when ``whole`` is 0."""
    return part / whole * 100 if whole > 0 else 0.0


def _active(baseline: DataFrame) -> DataFrame:
    return baseline.where(F.col("active") == 1)


def _of_provider(df: DataFrame, provider: str) -> DataFrame:
    return df.where(F.col("provider") == provider)


def total_counts(inp: MetricInputs) -> list[MetricRow]:
    """Input, active baseline and result counts."""
    provider_baseline = _of_provider(inp.prepared_baseline, inp.provider)
    active = _active(inp.prepared_baseline)
    provider_active = _active(provider_baseline)
    counts = {
        "MatchCandidateInputCount": inp.candidates.count(),
        "MatchBaselineInputCount": inp.baseline.count(),
        "MatchProviderBaselineInputCount": provider_baseline.count(),
        "MatchActiveBaselineRecordCount": active.count(),
        "MatchActiveBaselineEntityCount": active.select("id").distinct().count(),
        "MatchProviderActiveBaselineEntityCount": provider_active.select("id")
        .distinct()
        .count(),
        "MatchResultsCount": inp.results.count(),
    }
    return [(metric_id, float(n), "") for metric_id, n in counts.items()]


def match_result_breakdown(inp: MetricInputs) -> list[MetricRow]:
    """Count the IDs and provider records the run makes new, retained, resurrected or dropped.

    Each result row is classified by its ID, and by its ``(id, provider)`` record.
    New plus retained plus resurrected equals the result count at both levels.
    Dropped counts are active records of the run's provider that no result carries,
    and the IDs among them that no other provider keeps active.
    """
    baseline = inp.prepared_baseline
    results = inp.results
    total = results.count()

    new_ids = results.join(inp.baseline, "id", "left_anti").count()
    deleted_ids = (
        baseline.groupBy("id")
        .agg(F.max("active").alias("max_active"))
        .where(F.col("max_active") == 0)
        .select("id")
    )
    resurrected_ids = results.select("id").join(deleted_ids, "id", "left_semi").count()

    keys = ["id", "provider"]
    new_records = results.join(baseline.select(*keys), keys, "left_anti").count()
    deleted_records = baseline.where(F.col("active") == 0).select(*keys)
    resurrected_records = results.join(deleted_records, keys, "left_semi").count()

    from_provider = _active(_of_provider(baseline, inp.provider))
    dropped_records = from_provider.join(
        results.select("id"), "id", "left_anti"
    ).select("id")
    from_others = _active(baseline).where(
        ~F.coalesce(F.col("provider") == inp.provider, F.lit(False))
    )
    dropped_ids = dropped_records.distinct().join(
        from_others.select("id"), "id", "left_anti"
    )

    counts = {
        "MatchNewEntityCount": new_ids,
        "MatchRetainedEntityCount": total - new_ids - resurrected_ids,
        "MatchResurrectedEntityCount": resurrected_ids,
        "MatchDroppedEntityCount": dropped_ids.count(),
        "MatchNewProviderRecordCount": new_records,
        "MatchRetainedProviderRecordCount": total - new_records - resurrected_records,
        "MatchResurrectedProviderRecordCount": resurrected_records,
        "MatchDroppedProviderRecordCount": dropped_records.count(),
        "MatchReductionCount": inp.candidates.count() - total,
    }
    return [(metric_id, float(n), "") for metric_id, n in counts.items()]


def history_match(inp: MetricInputs) -> list[MetricRow]:
    """Candidates that setup matched through history, and their share."""
    total = inp.prepared_candidates.count()
    matched = inp.prepared_candidates.where(F.col("base_ids").isNotNull()).count()
    rate = round_half_up(percentage(matched, total), 2)
    return [
        ("MatchHistoryMatchCount", float(matched), ""),
        ("MatchHistoryMatchRate", rate, ""),
    ]


SHARED_METRICS: tuple[tuple[str, Metric], ...] = (
    ("Total Counts", total_counts),
    ("Match Result Breakdown", match_result_breakdown),
    ("History Match Rate", history_match),
)


def compute(
    inputs: MetricInputs, metrics: Iterable[tuple[str, Metric]]
) -> list[MetricRow]:
    """Run each metric in order, skipping those that raise."""
    rows: list[MetricRow] = []
    for name, metric in metrics:
        try:
            rows += metric(inputs)
        except Exception:
            logger.exception("Metric %s failed and has no rows", name)
    return rows


def to_frame(
    spark: SparkSession,
    rows: Iterable[MetricRow],
    theme: str,
    feature_type: str,
    provider: str,
) -> DataFrame:
    """One-file metrics table. The first row of each ``(metric_id, sub_partition)`` wins."""
    now = datetime.now(UTC)
    unique = {}
    for metric_id, value, sub_partition in rows:
        if (metric_id, sub_partition) in unique:
            logger.warning("Dropping duplicate metric %s %r", metric_id, sub_partition)
        unique.setdefault((metric_id, sub_partition), value)
    data = [
        (
            metric_id,
            float(value),
            now,
            theme,
            feature_type,
            "",
            f"{provider}-{theme}",
            sub_partition,
            "",
            "",
        )
        for (metric_id, sub_partition), value in unique.items()
    ]
    return spark.createDataFrame(data, ROW_SCHEMA).coalesce(1)
