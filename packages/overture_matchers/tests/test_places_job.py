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
    RawInputs,
    SetupOutput,
)
from overture_matchers.places import job
from overture_matchers.places.blocking import SPATIAL_JOIN_CONF


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
def stub_stages(spark, monkeypatch):
    def setup(
        raw, provider, model_jar, baseline_provider=None, match_history=True
    ) -> SetupOutput:
        return SetupOutput(spark.range(1), spark.range(1))

    def matching(prepared, model_jar, **kwargs) -> MatchingOutput:
        raise NotImplementedError("matching is not ported")

    monkeypatch.setattr(job, "setup", setup)
    monkeypatch.setattr(job, "matching", matching)
    saved = {key: spark.conf.get(key, None) for key in SPATIAL_JOIN_CONF}
    yield
    for key, value in saved.items():
        if value is None:
            spark.conf.unset(key)
        else:
            spark.conf.set(key, value)


def test_job_writes_config_then_stops_at_first_stub(
    spark, tmp_path, params, stub_stages
):
    runner = job.PlacesMatchingJob()
    runner.spark = spark
    result = runner.run(params)
    assert isinstance(result.exception, NotImplementedError)
    assert (tmp_path / "run" / "config.json").is_file()
    assert "/run/checkpoints/" in spark.sparkContext.getCheckpointDir()
    assert spark.conf.get("spark.sql.adaptive.enabled") == "false"
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

    def setup(
        raw, provider, model_jar, baseline_provider=None, match_history=True
    ) -> SetupOutput:
        calls["setup"] = (raw.candidates.count(), match_history)
        return SetupOutput(spark.range(3), spark.range(4))

    def matching(
        prepared,
        model_jar,
        output_path=None,
        embeddings_cache_table=None,
        output_debug=True,
    ) -> MatchingOutput:
        calls["matching"] = (
            prepared.candidates.count(),
            prepared.baseline.count(),
            output_path,
            output_debug,
        )
        return MatchingOutput(spark.range(5))

    def assign_ids(
        raw, prepared, matched, provider, output_path=None
    ) -> AssignIdsOutput:
        calls["assign_ids"] = (matched.matches.count(), provider, output_path)
        return AssignIdsOutput(spark.range(6), spark.range(7))

    def metrics(
        raw, prepared, matched, ids, provider, load_optional=dict
    ) -> MetricsOutput:
        calls["metrics"] = (ids.final.count(), ids.fallout.count(), load_optional())
        return MetricsOutput(spark.range(1))

    def load_optional_inputs(spark, config, feature_type):
        return {"loaded": True}

    for fake in (setup, matching, assign_ids, metrics, load_optional_inputs):
        monkeypatch.setattr(job, fake.__name__, fake)
    config = job.RunConfig.from_params(
        params | {"matchHistory": "false", "outputDebug": "false"}
    )
    job.run(spark, config)
    run = params["outputPath"]
    assert calls == {
        "setup": (2, False),
        "matching": (3, 4, run, False),
        "assign_ids": (5, config.provider, run),
        "metrics": (6, 7, {"loaded": True}),
    }
    calls.clear()
    job.run(spark, config)
    assert calls == {}


def test_run_local_reads_params_file(spark, tmp_path, params, stub_stages):
    params_file = tmp_path / "params.json"
    params_file.write_text(json.dumps(params))
    with pytest.raises(NotImplementedError, match="matching"):
        run_local(job.run, str(params_file), spark=spark)
    assert (tmp_path / "run" / "config.json").is_file()


def test_assign_ids_and_metrics_stages(spark):
    raw = RawInputs(
        spark.createDataFrame([("a", "x"), ("b", "y")], "id string, name string"),
        spark.createDataFrame([("B1", "acme")], "id string, provider string"),
    )
    prepared = SetupOutput(
        spark.createDataFrame(
            [("a", None), ("b", None)], "id string, base_ids array<string>"
        ),
        spark.createDataFrame(
            [("B1", "acme", 1)], "id string, provider string, active int"
        ),
    )
    matched = MatchingOutput(
        spark.createDataFrame(
            [
                (
                    "1",
                    "a",
                    "candidate",
                    "B1",
                    "baseline",
                    "match",
                    "candidate_baseline",
                    0.9,
                ),
                ("2", "b", "candidate", None, None, "nomatch", "nomatch", -1.0),
            ],
            "cluster_id string, id string, record_type string, base_id string, "
            "base_record_type string, match_type string, match_sub_type string, "
            "composite_score double",
        )
    )
    assigned = job.assign_ids(raw, prepared, matched, "acme")
    final = {r["name"]: r["id"] for r in assigned.final.collect()}
    assert final["x"] == "B1"
    assert final["y"] not in ("b", "B1")
    assert assigned.fallout.count() == 0

    scores = spark.createDataFrame([(0.9,)], "composite_score double")
    out = job.metrics(
        raw,
        prepared,
        matched,
        assigned,
        "acme",
        load_optional=lambda: {"scores": scores},
    )
    rows = {
        (r["metric_id"], r["sub_partition"]): r["value"] for r in out.metrics.collect()
    }
    assert rows[("MatchGERSIdUniquenessCheck", "")] == 1.0
    assert rows[("MatchScoreDistribution", "clear_match")] == 1
