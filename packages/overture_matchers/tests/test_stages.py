# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Tests run_stage: skip, rerun, marker ordering and returned reads."""

from typing import NamedTuple

import pytest
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from overture_matchers.core.config import RunConfig
from overture_matchers.core.stages import run_stage
from overture_matchers.places import job


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
    assert (tmp_path / "_completed_stages" / "stage").is_file()
    assert (tmp_path / "first" / "type=place").is_dir()


def test_run_stage_skips_when_marker_exists(spark, tmp_path):
    calls = []

    def stage(n: int) -> Out:
        calls.append(n)
        return Out(spark.range(n), spark.range(n))

    config = make_config(tmp_path)
    run_stage(spark, config, "place", stage, 2)
    out = run_stage(spark, config, "place", stage, 99)
    assert calls == [2]
    assert out.first.count() == 2


def test_run_stage_reruns_without_marker_and_overwrites(spark, tmp_path):
    def stage(n: int) -> Out:
        return Out(spark.range(n), spark.range(n))

    config = make_config(tmp_path)
    run_stage(spark, config, "place", stage, 5)
    (tmp_path / "_completed_stages" / "stage").unlink()
    out = run_stage(spark, config, "place", stage, 2)
    assert out.first.count() == 2


def test_marker_is_written_only_after_all_outputs(spark, tmp_path):
    def stage() -> Out:
        bad = spark.range(1).select(F.raise_error("boom").cast("int"))
        return Out(spark.range(1), bad)

    with pytest.raises(Exception, match="boom"):
        run_stage(spark, make_config(tmp_path), "place", stage)
    assert not (tmp_path / "_completed_stages").exists()


def test_rerunning_a_stage_reruns_the_stages_built_on_it(spark, tmp_path):
    calls = []

    def first() -> Out:
        calls.append("first")
        return Out(spark.range(1), spark.range(1))

    def second() -> Out:
        calls.append("second")
        return Out(spark.range(1), spark.range(1))

    config = make_config(tmp_path)
    run_stage(spark, config, "place", first)
    run_stage(spark, config, "place", second)
    (tmp_path / "_completed_stages" / "first").unlink()
    run_stage(spark, config, "place", first)
    run_stage(spark, config, "place", second)
    assert calls == ["first", "second", "first", "second"]


def test_run_stage_resolves_stages_with_type_checking_only_annotations(spark, tmp_path):
    with pytest.raises(NotImplementedError, match="matching"):
        run_stage(spark, make_config(tmp_path), "place", job.matching, None)
