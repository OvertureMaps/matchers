# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Tests the places job wiring: stage names, output names, local runner and job."""

import json

import pytest

from overture_matchers.core.local import run_local
from overture_matchers.core.stages import (
    AssignIdsOutput,
    MatchingOutput,
    MetricsOutput,
    SetupOutput,
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
        "modelJar": "/tmp/model.jar",
    }


def test_job_writes_config_then_stops_at_first_stub(spark, tmp_path, params):
    runner = job.PlacesMatchingJob()
    runner.spark = spark
    result = runner.run(params)
    assert isinstance(result.exception, NotImplementedError)
    assert (tmp_path / "run" / "config.json").is_file()
    assert "/run/checkpoints/" in spark.sparkContext.getCheckpointDir()
    assert not (tmp_path / "run" / "_completed_stages").exists()


def test_run_rejects_inputs_without_a_place_entry(spark, tmp_path, params):
    params["input"] = {"building": params["input"]["place"]}
    with pytest.raises(ValueError, match="'place' entry"):
        job.run(spark, job.RunConfig.from_params(params))


def test_run_rejects_a_missing_model_jar(spark, params):
    del params["modelJar"]
    with pytest.raises(ValueError, match="modelJar"):
        job.run(spark, job.RunConfig.from_params(params))


def test_run_passes_match_history_to_setup(spark, params, monkeypatch):
    seen = []

    def setup(raw, match_history=True) -> SetupOutput:
        seen.append(match_history)
        raise NotImplementedError

    monkeypatch.setattr(job, "setup", setup)
    with pytest.raises(NotImplementedError):
        job.run(spark, job.RunConfig.from_params(params | {"matchHistory": "false"}))
    assert seen == [False]


def test_run_local_reads_params_file(spark, tmp_path, params):
    params_file = tmp_path / "params.json"
    params_file.write_text(json.dumps(params))
    with pytest.raises(NotImplementedError, match="setup"):
        run_local(job.run, str(params_file), spark=spark)
    assert (tmp_path / "run" / "config.json").is_file()
