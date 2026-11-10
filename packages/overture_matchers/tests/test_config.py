# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Tests RunConfig param parsing and the config.json write and mismatch check."""

import json

import pytest

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
    assert RunConfig.from_params(PARAMS | {"modelJar": "/m.jar"}).model_jar == "/m.jar"


def test_from_params_treats_empty_and_none_strings_as_unset():
    config = RunConfig.from_params(
        PARAMS
        | {
            "previousMetricsPath": "",
            "embeddingsCacheTable": " None ",
            "outputDebug": "none",
        }
    )
    assert config.previous_metrics_path is None
    assert config.embeddings_cache_table is None
    assert config.output_debug is True


def test_from_params_rejects_missing_required_and_bad_bool():
    with pytest.raises(ValueError, match="provider"):
        RunConfig.from_params({k: v for k, v in PARAMS.items() if k != "provider"})
    with pytest.raises(ValueError, match="true or false"):
        RunConfig.from_params(PARAMS | {"outputDebug": "maybe"})


def test_for_run_writes_config_json_with_versions(spark, tmp_path):
    RunConfig.for_run(spark, PARAMS | {"outputPath": str(tmp_path)})
    record = json.loads((tmp_path / "config.json").read_text())
    assert record["provider"] == "acme"
    assert record["output_debug"] is True
    assert "overture-matchers" in record["versions"]


def test_for_run_accepts_same_params_and_rejects_changed_ones(spark, tmp_path):
    params = PARAMS | {"outputPath": str(tmp_path)}
    RunConfig.for_run(spark, params)
    RunConfig.for_run(spark, params)
    with pytest.raises(ValueError, match="different params"):
        RunConfig.for_run(spark, params | {"outputDebug": False})
