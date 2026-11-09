# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Tests the run-folder layout and that a checkpoint cuts lineage."""

from overture_matchers.core import io


def test_write_output_uses_scala_layout_with_zstd(spark, tmp_path):
    out = str(tmp_path)
    io.write_output(out, "candidates", "place", spark.range(3))
    files = list((tmp_path / "candidates" / "type=place").glob("*.parquet"))
    assert files and all(f.name.endswith(".zstd.parquet") for f in files)
    assert io.read_output(spark, out, "candidates", "place").count() == 3


def test_write_output_overwrites(spark, tmp_path):
    out = str(tmp_path)
    io.write_output(out, "final", "place", spark.range(5))
    io.write_output(out, "final", "place", spark.range(2))
    assert io.read_output(spark, out, "final", "place").count() == 2


def test_stage_marker_roundtrip(spark, tmp_path):
    out = str(tmp_path)
    assert not io.stage_is_done(spark, out, "setup")
    io.mark_stage_done(spark, out, "setup")
    assert io.stage_is_done(spark, out, "setup")
    assert (tmp_path / "_completed_stages" / "setup").is_file()


def test_checkpoint_writes_and_cuts_lineage(spark, tmp_path):
    source = str(tmp_path / "source")
    spark.range(4).write.parquet(source)
    df = spark.read.parquet(source).filter("id > 0")
    checkpoint = io.checkpointer(spark, str(tmp_path / "run"), "place")
    result = checkpoint(df, "blocking")
    assert result.count() == 3
    assert all("/run/blocking/type=place/" in f for f in result.inputFiles())
    assert not any("/source/" in f for f in result.inputFiles())


def test_run_uri_reads_s3_as_s3a():
    assert io.output_uri("s3://b/run/", "final", "place") == (
        "s3a://b/run/final/type=place"
    )


def test_read_baseline_adds_soft_deletes_not_still_active(spark, tmp_path):
    base = str(tmp_path / "base.parquet")
    spark.createDataFrame(
        [("a", "p"), ("b", "p")], "id string, provider string"
    ).write.parquet(base)
    spark.createDataFrame(
        [("b", "p"), ("c", "p")], "id string, provider string"
    ).write.parquet(str(tmp_path / "base_deletes.parquet"))
    ids = sorted(r.id for r in io.read_baseline(spark, [base]).collect())
    assert ids == ["a", "b", "c"]


def test_read_baseline_without_deletes_reads_active(spark, tmp_path):
    base = str(tmp_path / "base")
    spark.range(2).write.parquet(base)
    assert io.read_baseline(spark, [base]).count() == 2


def test_clear_stages_built_on_keeps_upstream_markers(spark, tmp_path):
    out = str(tmp_path)
    for stage in ("setup", "matching", "assign_ids"):
        io.mark_stage_done(spark, out, stage)
    io.clear_stages_built_on(spark, out, "matching")
    done = [
        s
        for s in ("setup", "matching", "assign_ids")
        if io.stage_is_done(spark, out, s)
    ]
    assert done == ["setup", "matching"]
