"""Tests loading the optional places metric inputs from a run folder."""

import json
import zipfile
from types import SimpleNamespace

from overture_matchers.core import io
from overture_matchers.places import metric_inputs, model_jar


def _jar(path, features=("f",)):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(model_jar.CONFIG_RESOURCE, json.dumps({"feature_order": features}))
    return str(path)


def test_loads_what_the_run_folder_has(spark, tmp_path, monkeypatch):
    run = str(tmp_path / "run")
    scores = spark.createDataFrame([(float(i),) for i in range(4)], "f double")
    io.write_output(run, "scores", "place", scores.repartition(4))
    previous = tmp_path / "previous"
    spark.createDataFrame(
        [("M", "", 1.0)], "metric_id string, sub_partition string, value double"
    ).write.parquet(str(previous))
    config = SimpleNamespace(
        output_path=run,
        model_jar=_jar(tmp_path / "model.jar"),
        previous_metrics_path=str(previous),
    )
    # One file is enough to reach the target.
    monkeypatch.setattr(metric_inputs, "SAMPLE_BYTES", 1)
    found = metric_inputs.load_optional_inputs(spark, config, "place")
    assert set(found) == {"scores", "score_sample", "feature_order", "previous"}
    assert found["scores"].count() == 4
    assert len(found["score_sample"].inputFiles()) == 1
    assert found["feature_order"] == ("f",)


def test_leaves_out_what_is_missing_or_unreadable(spark, tmp_path):
    config = SimpleNamespace(
        output_path=str(tmp_path / "run"),
        model_jar=str(tmp_path / "missing.jar"),
        previous_metrics_path=None,
    )
    assert metric_inputs.load_optional_inputs(spark, config, "place") == {}
