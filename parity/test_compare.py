# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
# ruff: noqa: D103
"""Checks compare.py on tiny synthetic run folders; no AWS involved.

The reference run has candidates c1 and c2 matched to baseline b1, and c3 and
c4 matched to each other under a new ID. Each test changes one thing.
"""

from pathlib import Path

import duckdb
from compare import compare

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


def by_value(rows: list[dict], col: str) -> dict:
    return {r[col]: r["same_id_share"] for r in rows}


def shares_of_pairs_kept(report: dict) -> tuple[float, float]:
    g = report["groups"]
    return g["share_of_a_pairs_also_in_b"], g["share_of_b_pairs_also_in_a"]


def test_identical_runs_agree(tmp_path):
    a = make_run(tmp_path, "a")
    r = compare(a, make_run(tmp_path, "b"))
    assert r["ids"]["same_id_share"] == 1.0 and r["ids"]["records"] == 4
    assert shares_of_pairs_kept(r) == (1.0, 1.0)


def test_relabeled_group_and_new_id_still_agree(tmp_path):
    a = make_run(tmp_path, "a")
    relabeled = [
        (i, b, s, {"k1": "z9", "k2": "z8"}[k]) for i, b, s, k in REFERENCE_MATCHES
    ]
    b = make_run(
        tmp_path, "b", relabeled, REFERENCE_IDS | {"c3": "new-y", "c4": "new-y"}
    )
    r = compare(a, b)
    assert r["ids"]["same_id_share"] == 1.0
    assert shares_of_pairs_kept(r) == (1.0, 1.0)


def test_split_group(tmp_path):
    a = make_run(tmp_path, "a")
    split = REFERENCE_MATCHES[:2] + [
        ("c3", None, "nomatch", "u1"),
        ("c4", None, "nomatch", "u2"),
    ]
    b = make_run(tmp_path, "b", split, REFERENCE_IDS | {"c3": "new-y", "c4": "new-z"})
    r = compare(a, b)
    assert r["ids"]["same_id_share"] == 0.5
    assert by_value(r["ids"]["by_country"], "country") == {"US": 1.0, "FR": 0.0}
    assert by_value(r["ids"]["by_match_type"], "match_sub_type") == {
        "candidate_baseline": 1.0,
        "candidate_candidate": 0.0,
    }
    # A's pairs: (c1,c2), (c1,b1), (c2,b1), (c3,c4). B keeps all but (c3,c4).
    assert shares_of_pairs_kept(r) == (0.75, 1.0)
    assert r["groups"]["pairs_grouped_together_in_both"] == 3


def test_changed_reused_id(tmp_path):
    a = make_run(tmp_path, "a")
    moved = [
        REFERENCE_MATCHES[0],
        ("c2", "b2", "candidate_baseline", "k3"),
        REFERENCE_MATCHES[2],
    ]
    b = make_run(tmp_path, "b", moved, REFERENCE_IDS | {"c2": "b2"})
    r = compare(a, b)
    assert r["ids"]["same_id_share"] == 0.75
    # b2 is grouped only in B, so it is left out; B keeps (c1,b1) and (c3,c4).
    assert shares_of_pairs_kept(r) == (0.5, 1.0)


def test_candidate_missing_from_one_run_disagrees(tmp_path):
    a = make_run(tmp_path, "a")
    b = make_run(
        tmp_path, "b", ids={k: v for k, v in REFERENCE_IDS.items() if k != "c1"}
    )
    assert compare(a, b)["ids"]["same_id_share"] == 0.75
