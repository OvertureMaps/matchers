# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Tests RunConfig param parsing and the config.json write and mismatch check."""

import json

import pytest

from overture_matchers.core import io
from overture_matchers.core.config import RunConfig

PARAMS = {
    "outputPath": "/tmp/run",
    "input": {"place": ["/tmp/in"]},
    "baseline": '{"place": "/tmp/base"}',
    "provider": "acme",
}


def test_from_params_fills_defaults_and_parses_types():
    config = RunConfig.from_params(
        PARAMS | {"matchHistory": "true", "outputResultFile": "ignored.json"}
    )
    assert config.baseline == {"place": ["/tmp/base"]}
    assert config.match_history is True
    assert config.output_debug is True
    assert config.embeddings_cache_table is None
    assert config.model_jar is None
    assert (
        RunConfig.from_params(PARAMS | {"outputDebug": "false"}).output_debug is False
    )
    assert RunConfig.from_params(PARAMS | {"modelJar": "/m.jar"}).model_jar == "/m.jar"
    assert RunConfig.from_params(PARAMS | {"outputPath": "/r/"}).output_path == "/r"
    s3 = RunConfig.from_params(
        PARAMS
        | {
            "outputPath": "s3://b/run/",
            "input": {"place": "s3://b/in"},
            "previousMetricsPath": "s3://b/m",
        }
    )
    assert (s3.output_path, s3.input, s3.previous_metrics_path) == (
        "s3a://b/run",
        {"place": ["s3a://b/in"]},
        "s3a://b/m",
    )


def test_from_params_treats_empty_and_none_strings_as_unset():
    config = RunConfig.from_params(
        PARAMS
        | {
            "previousMetricsPath": "",
            "embeddingsCacheTable": " None ",
            "matchHistory": "none",
        }
    )
    assert config.previous_metrics_path is None
    assert config.embeddings_cache_table is None
    assert config.match_history is True


def test_from_params_rejects_missing_required_bad_bool_and_bad_map():
    with pytest.raises(ValueError, match="provider"):
        RunConfig.from_params({k: v for k, v in PARAMS.items() if k != "provider"})
    with pytest.raises(ValueError, match="true or false"):
        RunConfig.from_params(PARAMS | {"matchHistory": "maybe"})
    with pytest.raises(TypeError, match="input must be a"):
        RunConfig.from_params(PARAMS | {"input": '["s3://bucket/a/"]'})
    with pytest.raises(TypeError, match="input must be a"):
        RunConfig.from_params(PARAMS | {"input": {"place": {"path": "/tmp/in"}}})
    with pytest.raises(TypeError, match="baseline must be a"):
        RunConfig.from_params(PARAMS | {"baseline": '{"place": '})


def test_for_run_writes_config_json_with_versions(spark, tmp_path):
    RunConfig.for_run(spark, PARAMS | {"outputPath": str(tmp_path)})
    record = json.loads((tmp_path / "config.json").read_text())
    assert record["provider"] == "acme"
    assert record["match_history"] is True
    assert "overture-matchers" in record["versions"]


def test_for_run_rejects_changed_params_only_once_a_stage_is_done(spark, tmp_path):
    params = PARAMS | {"outputPath": str(tmp_path)}
    RunConfig.for_run(spark, params | {"matchHistory": False})
    RunConfig.for_run(spark, params)
    io.mark_stage_done(spark, str(tmp_path), "setup", "place")
    RunConfig.for_run(spark, params)
    with pytest.raises(
        ValueError, match=r"differs in \['match_history'\]; start a new run folder"
    ):
        RunConfig.for_run(spark, params | {"matchHistory": False})


def test_for_run_rejects_a_new_matcher_version_once_a_stage_is_done(
    spark, tmp_path, monkeypatch
):
    params = PARAMS | {"outputPath": str(tmp_path)}
    RunConfig.for_run(spark, params)
    io.mark_stage_done(spark, str(tmp_path), "setup", "place")
    monkeypatch.setattr("overture_matchers.core.config.version", lambda _: "99")
    with pytest.raises(ValueError, match=r"differs in \['versions'\]"):
        RunConfig.for_run(spark, params)
