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


def test_stage_marker_roundtrip(spark, tmp_path):
    out = str(tmp_path)
    assert not io.any_stage_done(spark, out)
    io.mark_stage_done(spark, out, "setup", "place")
    assert io.stage_is_done(spark, out, "setup", "place")
    assert not io.stage_is_done(spark, out, "setup", "division")
    assert io.any_stage_done(spark, out)
    assert (tmp_path / "_completed_stages" / "setup" / "type=place").is_file()


def test_checkpoint_writes_and_cuts_lineage(spark, tmp_path):
    source = str(tmp_path / "source")
    spark.range(4).write.parquet(source)
    df = spark.read.parquet(source).filter("id > 0")
    result = io.checkpoint(df, str(tmp_path / "run"), "blocking", "place")
    assert result.count() == 3
    assert all("/run/blocking/type=place/" in f for f in result.inputFiles())
    assert not any("/source/" in f for f in result.inputFiles())


def test_checkpoint_without_output_path_returns_df(spark):
    df = spark.range(2)
    assert io.checkpoint(df, None, "blocking", "place") is df
