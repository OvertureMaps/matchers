# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
# ruff: noqa: D103
"""Checks compare.py and launch.py on tiny synthetic run folders; no AWS involved.

The reference run has candidates c1 and c2 matched to baseline b1, and c3 and
c4 matched to each other under a minted ID. Each test changes one thing.
"""

from pathlib import Path

import duckdb
import pytest
from compare import compare
from launch import build_arguments, parse_overrides, seed, seed_folders

BASELINE = ["b1", "b2"]
CANDIDATES = {"c1": "US", "c2": "US", "c3": "FR", "c4": "FR"}
REFERENCE_MATCHES = [
    ("c1", "b1", "candidate_baseline", "k1"),
    ("c2", "b1", "candidate_baseline", "k1"),
    ("c3", "c4", "candidate_candidate", "k2"),
]
REFERENCE_IDS = {"c1": "b1", "c2": "b1", "c3": "new-x", "c4": "new-x"}


def sql_value(v) -> str:
    return "NULL" if v is None else repr(v)


def write(run: Path, folder: str, columns: list[str], rows: list[tuple]) -> None:
    """Write rows as one parquet file in `<run>/<folder>/type=place/`."""
    out = run / folder / "type=place"
    out.mkdir(parents=True)
    values = ", ".join("(" + ", ".join(sql_value(v) for v in r) + ")" for r in rows)
    duckdb.sql(
        f"COPY (SELECT * FROM (VALUES {values}) t({', '.join(columns)})) TO '{out}/part-0.parquet'"
    )


def make_run(
    root: Path, name: str, matches=REFERENCE_MATCHES, ids=REFERENCE_IDS
) -> str:
    """A run folder with the outputs compare.py reads. `matches` rows are (id, base_id, match_sub_type, cluster_id)."""
    run = root / name
    write(run, "baseline", ["id"], [(b,) for b in BASELINE])
    write(
        run,
        "candidates",
        ["id", "country"],
        [(c, k) for c, k in CANDIDATES.items()],
    )
    write(run, "idMap", ["id", "assigned_id"], list(ids.items()))
    rows = [(i, b, "nomatch" if b is None else "match", s, k) for i, b, s, k in matches]
    write(
        run,
        "matches",
        ["id", "base_id", "match_type", "match_sub_type", "cluster_id"],
        rows,
    )
    return str(run)


def by_value(rows: list[dict]) -> dict:
    return {r["value"]: r["agreement"] for r in rows}


def test_identical_runs_agree(tmp_path):
    a = make_run(tmp_path, "a")
    r = compare(a, make_run(tmp_path, "b"))
    assert r["id"]["agreement"] == 1.0 and r["id"]["n"] == 4
    assert r["cluster"]["precision"] == r["cluster"]["recall"] == 1.0
    assert r["edge"]["precision"] == r["edge"]["recall"] == 1.0


def test_relabeled_cluster_and_minted_id_still_agree(tmp_path):
    a = make_run(tmp_path, "a")
    relabeled = [
        (i, b, s, {"k1": "z9", "k2": "z8"}[k]) for i, b, s, k in REFERENCE_MATCHES
    ]
    b = make_run(
        tmp_path, "b", relabeled, REFERENCE_IDS | {"c3": "new-y", "c4": "new-y"}
    )
    r = compare(a, b)
    assert r["id"]["agreement"] == 1.0
    assert r["cluster"]["precision"] == r["cluster"]["recall"] == 1.0


def test_split_cluster(tmp_path):
    a = make_run(tmp_path, "a")
    split = REFERENCE_MATCHES[:2] + [
        ("c3", None, "nomatch", "u1"),
        ("c4", None, "nomatch", "u2"),
    ]
    b = make_run(tmp_path, "b", split, REFERENCE_IDS | {"c3": "new-y", "c4": "new-z"})
    r = compare(a, b)
    assert r["id"]["agreement"] == 0.5
    assert by_value(r["id"]["by"]["country"]) == {"US": 1.0, "FR": 0.0}
    assert by_value(r["id"]["by"]["match_sub_type"]) == {
        "candidate_baseline": 1.0,
        "candidate_candidate": 0.0,
    }
    # A's pairs: (c1,c2), (c1,b1), (c2,b1), (c3,c4). B keeps all but (c3,c4).
    assert r["cluster"]["precision"] == 1.0 and r["cluster"]["recall"] == 0.75
    assert r["edge"]["precision"] == 1.0 and r["edge"]["recall"] == pytest.approx(2 / 3)


def test_changed_reused_id(tmp_path):
    a = make_run(tmp_path, "a")
    moved = [
        REFERENCE_MATCHES[0],
        ("c2", "b2", "candidate_baseline", "k3"),
        REFERENCE_MATCHES[2],
    ]
    b = make_run(tmp_path, "b", moved, REFERENCE_IDS | {"c2": "b2"})
    r = compare(a, b)
    assert r["id"]["agreement"] == 0.75
    assert r["edge"]["precision"] == r["edge"]["recall"] == pytest.approx(2 / 3)


def test_candidate_missing_from_one_run_disagrees(tmp_path):
    a = make_run(tmp_path, "a")
    b = make_run(
        tmp_path, "b", ids={k: v for k, v in REFERENCE_IDS.items() if k != "c1"}
    )
    assert compare(a, b, group_by=[])["id"]["agreement"] == 0.75


def test_stage_setup_diffs_columns_within_tolerance(tmp_path):
    def run(name, rows, extra=()):
        write(tmp_path / name, "candidates", ["id", "score", "name", *extra], rows)
        write(tmp_path / name, "baseline", ["id"], [("b1",)])
        return str(tmp_path / name)

    a = run("a", [("c1", 0.5, "x"), ("c2", 0.5, "y"), ("c3", 0.5, "z")])
    b = run(
        "b",
        [("c1", 0.5 + 1e-9, "x", 1), ("c2", 0.6, "Y", 1), ("c4", 0.5, "z", 1)],
        ["extra"],
    )
    t = compare(a, b, stage="setup")["tables"]["candidates"]
    assert (t["keys_only_a"], t["keys_only_b"], t["keys_both"]) == (1, 1, 2)
    assert t["columns_only_b"] == ["extra"]
    assert t["mismatches"]["score"]["n"] == 1
    assert t["mismatches"]["score"]["max_abs_diff"] == pytest.approx(0.1)
    assert t["mismatches"]["name"] == {"n": 1}


def test_stage_matching_reports_pairs_and_flips(tmp_path):
    pair = ["id", "record_type", "base_id", "base_record_type"]

    def run(name, scores):
        # b3 is also a candidate, so c3 has a candidate-candidate row too.
        rows = [(i, "candidate", b, "baseline", s) for i, b, s in scores]
        rows += [("c3", "candidate", "b3", "candidate", 0.1)]
        write(tmp_path / name, "blocking", pair, [r[:4] for r in rows])
        write(tmp_path / name, "scores", [*pair, "composite_score"], rows)
        accepted = [
            (i, b, "candidate_baseline", "k") for i, b, s in scores if s >= 0.48
        ]
        write(
            tmp_path / name,
            "matches_validated",
            pair,
            [(i, "candidate", b, "baseline") for i, b, *_ in accepted],
        )
        write(
            tmp_path / name,
            "matches",
            ["id", "base_id", "match_type", "match_sub_type", "cluster_id"],
            [(i, b, "match", s, k) for i, b, s, k in accepted],
        )
        return str(tmp_path / name)

    a = run(
        "a",
        [("c1", "b1", 0.47), ("c2", "b2", 0.1), ("c3", "b3", 0.9), ("c4", "b4", 0.2)],
    )
    b = run(
        "b",
        [("c1", "b1", 0.49), ("c2", "b2", 0.9), ("c3", "b3", 0.9), ("c5", "b5", 0.2)],
    )
    r = compare(a, b, stage="matching")
    assert (
        r["tables"]["blocking"]["keys_only_a"],
        r["tables"]["blocking"]["keys_only_b"],
    ) == (1, 1)
    scores = r["tables"]["scores"]
    assert (scores["duplicate_keys_a"], scores["keys_both"]) == (0, 4)
    assert r["tables"]["blocking"]["flips"] is None
    flips = r["tables"]["scores"]["flips"]
    assert flips["n"] == 2
    assert [f["id"] for f in flips["examples"]] == ["c2", "c1"]
    assert flips["max_distance"] == pytest.approx(0.42)
    assert r["edge"]["edges_a"] == 1 and r["edge"]["edges_b"] == 3


def test_build_arguments_scala_template_gets_a_run_id():
    template = {
        "--runId": "old",
        "--outputPath": "s3a://bucket/old",
        "--input": "in",
        "--outputDebug": "false",
    }
    args, folder = build_arguments(
        template, "s3://bucket/root/", "r0", parse_overrides(["stages=setup"])
    )
    assert args == {
        "--runId": "r0",
        "--outputPath": "s3a://bucket/root",
        "--input": "in",
        "--outputDebug": "true",
        "--stages": "setup",
    }
    assert folder == "s3://bucket/root/r0"


def test_build_arguments_python_template_gets_an_output_path():
    args, folder = build_arguments(
        {"--outputPath": "s3://bucket/old", "x": "y"}, "s3://bucket/root", "r0", {}
    )
    assert args == {"--outputPath": "s3://bucket/root/r0", "--outputDebug": "true"}
    assert folder == "s3://bucket/root/r0"


def test_seed_folders_cover_the_stages_before_the_first_requested():
    assert seed_folders("matching") == ["candidates", "baseline", "embeddings"]
    assert seed_folders(" AssignId, metrics") == seed_folders("matching") + [
        "blocking",
        "scores",
        "matches_validated",
        "matches",
    ]
    with pytest.raises(SystemExit):
        seed_folders("setup,matching")
    with pytest.raises(SystemExit):
        seed_folders("bogus")


def test_seed_copies_folders_under_the_new_run_folder():
    class FakeS3:
        def __init__(self):
            self.copied = []

        def get_paginator(self, _):
            return self

        def paginate(self, Bucket, Prefix):
            return [{"Contents": [{"Key": f"{Prefix}type=place/part-0.parquet"}]}]

        def copy(self, source, bucket, key):
            self.copied.append((source["Bucket"], source["Key"], bucket, key))

    s3 = FakeS3()
    seed(s3, "s3a://src/runs/scala/", "s3://dst/parity/r0", ["candidates"])
    assert s3.copied == [
        (
            "src",
            "runs/scala/candidates/type=place/part-0.parquet",
            "dst",
            "parity/r0/candidates/type=place/part-0.parquet",
        )
    ]
