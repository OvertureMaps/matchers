# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Tests run_stage: skip, rerun, marker ordering and returned reads."""

from typing import NamedTuple

import pytest
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from overture_matchers.core.config import RunConfig
from overture_matchers.core.stages import run_stage


class Out(NamedTuple):
    first: DataFrame
    second: DataFrame


def make_config(tmp_path):
    return RunConfig(output_path=str(tmp_path), input={}, baseline={}, provider="acme")


def test_run_stage_writes_outputs_marker_and_returns_reads(spark, tmp_path):
    def stage(n: int) -> Out:
        return Out(spark.range(n), spark.range(n + 1))

    out = run_stage(spark, make_config(tmp_path), "place", stage, 2)
    assert (out.first.count(), out.second.count()) == (2, 3)
    assert (tmp_path / "_completed_stages" / "stage" / "type=place").is_file()
    assert (tmp_path / "first" / "type=place").is_dir()


def test_run_stage_skips_when_marker_exists_for_that_feature_type(spark, tmp_path):
    calls = []

    def stage(n: int) -> Out:
        calls.append(n)
        return Out(spark.range(n), spark.range(n))

    config = make_config(tmp_path)
    run_stage(spark, config, "place", stage, 2)
    out = run_stage(spark, config, "place", stage, 99)
    assert calls == [2]
    assert out.first.count() == 2
    run_stage(spark, config, "division", stage, 3)
    assert calls == [2, 3]


def test_run_stage_reruns_without_marker_and_overwrites(spark, tmp_path):
    def stage(n: int) -> Out:
        return Out(spark.range(n), spark.range(n))

    config = make_config(tmp_path)
    run_stage(spark, config, "place", stage, 5)
    (tmp_path / "_completed_stages" / "stage" / "type=place").unlink()
    out = run_stage(spark, config, "place", stage, 2)
    assert out.first.count() == 2


def test_marker_is_written_only_after_all_outputs(spark, tmp_path):
    def stage() -> Out:
        bad = spark.range(1).select(F.raise_error("boom").cast("int"))
        return Out(spark.range(1), bad)

    with pytest.raises(Exception, match="boom"):
        run_stage(spark, make_config(tmp_path), "place", stage)
    assert not (tmp_path / "_completed_stages").exists()


def test_run_stage_ignores_unresolvable_parameter_annotations(spark, tmp_path):
    def stage(n: "OnlyForTypeCheckers") -> Out:  # noqa: F821
        return Out(spark.range(n), spark.range(n))

    out = run_stage(spark, make_config(tmp_path), "place", stage, 1)
    assert out.first.count() == 1
