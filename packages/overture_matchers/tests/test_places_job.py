# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Tests the places job wiring: stage chaining, output names, local runner and job."""

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


@pytest.fixture
def fake_setup(spark, monkeypatch):
    def setup(raw, provider, model_jar, match_history=True) -> SetupOutput:
        return SetupOutput(spark.range(1), spark.range(1))

    monkeypatch.setattr(job, "setup", setup)


def test_job_writes_config_then_stops_at_first_stub(
    spark, tmp_path, params, fake_setup
):
    runner = job.PlacesMatchingJob()
    runner.spark = spark
    result = runner.run(params)
    assert isinstance(result.exception, NotImplementedError)
    assert (tmp_path / "run" / "config.json").is_file()
    assert "/run/checkpoints/" in spark.sparkContext.getCheckpointDir()
    assert "matching" in str(result.exception)
    assert not (tmp_path / "run" / "_completed_stages" / "matching").exists()


def test_run_rejects_inputs_without_a_place_entry(spark, tmp_path, params):
    params["input"] = {"building": params["input"]["place"]}
    with pytest.raises(ValueError, match="'place' entry"):
        job.run(spark, job.RunConfig.from_params(params))


def test_run_rejects_a_missing_model_jar(spark, params):
    del params["modelJar"]
    with pytest.raises(ValueError, match="modelJar"):
        job.run(spark, job.RunConfig.from_params(params))


def test_run_chains_stage_outputs_and_skips_on_rerun(spark, params, monkeypatch):
    calls = {}

    def setup(raw, provider, model_jar, match_history=True) -> SetupOutput:
        calls["setup"] = (raw.candidates.count(), match_history)
        return SetupOutput(spark.range(3), spark.range(4))

    def matching(prepared, output_path=None) -> MatchingOutput:
        calls["matching"] = (
            prepared.candidates.count(),
            prepared.baseline.count(),
            output_path,
        )
        return MatchingOutput(spark.range(5))

    def assign_ids(raw, prepared, matched, output_path=None) -> AssignIdsOutput:
        calls["assign_ids"] = (matched.matches.count(), output_path)
        return AssignIdsOutput(spark.range(6), spark.range(7))

    def metrics(raw, prepared, matched, ids) -> MetricsOutput:
        calls["metrics"] = (ids.final.count(), ids.fallout.count())
        return MetricsOutput(spark.range(1))

    for fake in (setup, matching, assign_ids, metrics):
        monkeypatch.setattr(job, fake.__name__, fake)
    config = job.RunConfig.from_params(params | {"matchHistory": "false"})
    job.run(spark, config)
    run = params["outputPath"]
    assert calls == {
        "setup": (2, False),
        "matching": (3, 4, run),
        "assign_ids": (5, run),
        "metrics": (6, 7),
    }
    calls.clear()
    job.run(spark, config)
    assert calls == {}


def test_run_local_reads_params_file(spark, tmp_path, params, fake_setup):
    params_file = tmp_path / "params.json"
    params_file.write_text(json.dumps(params))
    with pytest.raises(NotImplementedError, match="matching"):
        run_local(job.run, str(params_file), spark=spark)
    assert (tmp_path / "run" / "config.json").is_file()
