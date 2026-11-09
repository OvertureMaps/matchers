# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Tests the places job wiring: stage names, output names, local runner and job."""

import json

import pytest

from overture_matchers.core.stages import (
    AssignIdsOutput,
    MatchingOutput,
    MetricsOutput,
    SetupOutput,
    run_local,
)
from overture_matchers.places import job


def test_stage_function_names_are_marker_names():
    names = [f.__name__ for f in (job.setup, job.matching, job.assign_ids, job.metrics)]
    assert names == ["setup", "matching", "assign_ids", "metrics"]


def test_output_names_match_scala_postfixes_and_are_unique():
    fields = [
        f
        for t in (SetupOutput, MatchingOutput, AssignIdsOutput, MetricsOutput)
        for f in t._fields
    ]
    assert sorted(fields) == sorted(
        ["candidates", "baseline", "matches", "final", "fallout", "metrics"]
    )


@pytest.fixture
def params(spark, tmp_path):
    for side in ("candidates", "baseline"):
        spark.range(2).write.parquet(str(tmp_path / side))
    return {
        "outputPath": str(tmp_path / "run"),
        "input": {"place": [str(tmp_path / "candidates")]},
        "baseline": {"place": [str(tmp_path / "baseline")]},
        "provider": "acme",
    }


def test_job_writes_config_then_stops_at_first_stub(spark, tmp_path, params):
    runner = job.PlacesMatchingJob()
    runner.spark = spark
    result = runner.run(params)
    assert isinstance(result.exception, NotImplementedError)
    assert (tmp_path / "run" / "config.json").is_file()
    assert not (tmp_path / "run" / "_completed_stages").exists()


def test_run_local_reads_params_file(spark, tmp_path, params):
    params_file = tmp_path / "params.json"
    params_file.write_text(json.dumps(params))
    with pytest.raises(NotImplementedError, match="setup"):
        run_local(job.run, str(params_file), spark=spark)
    assert (tmp_path / "run" / "config.json").is_file()
