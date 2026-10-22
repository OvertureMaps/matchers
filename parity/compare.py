# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.1"]
# ///
"""Compare two matcher runs on the same input and print how far apart they are, as JSON.

A run is a run folder in Scala's stage layout, `<run>/<folder>/type=<table>/`.
Run A is the reference, run B the run under test. Without `--stage`, the report
holds the parity gate metrics, which need the assignid stage's debug outputs
(`outputDebug=true`):

  id       share of candidates with the same outcome: the same reused ID, or both
           minted a new ID for the same set of candidates. Overall and per slice.
  cluster  pairwise precision and recall of "same cluster". Cluster IDs are
           arbitrary, so only membership is compared.
  edge     precision and recall of accepted pairs, a diagnostic.

`--stage setup|matching` instead diffs one stage's outputs: row keys, columns
(numbers within `--tolerance`), and score flips across the blocking and match
thresholds with their distance to the threshold. `--stage matching` adds the
cluster and edge metrics. Use it in mixed mode, where run B ran one stage on a
copy of run A's earlier outputs.

    uv run parity/compare.py A_RUN B_RUN [--table place] [--group-by country] [--stage matching]

Runs are local folders or s3:// URIs; S3 credentials come from the AWS config
chain, so AWS_PROFILE applies. Both runs must use the same model version.
"""

import argparse
import json

import duckdb

# Output folders each stage writes, in stage order. launch.py seeds mixed-mode
# run folders from this.
STAGE_OUTPUTS = {
    "setup": ["candidates", "baseline", "embeddings"],
    "matching": ["blocking", "scores", "matches_validated", "matches"],
    "assignid": ["idMap", "final", "fallout"],
    "metrics": ["metrics"],
}
PAIR_KEY = ["id", "base_id"]
# Row keys of the outputs `--stage` diffs column by column.
DIFF_KEYS = {
    "candidates": ["id"],
    "baseline": ["id"],
    "blocking": PAIR_KEY,
    "scores": PAIR_KEY,
    "matches_validated": PAIR_KEY,
}
# Places model thresholds per output folder. Scala persists no blocking score,
# so blocking flips appear only when both runs write `composite_score` there.
THRESHOLDS = {"blocking": 0.08, "scores": 0.48}
SLICES = ["country", "provider", "match_sub_type"]
NUMERIC_TYPES = {
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "FLOAT",
    "DOUBLE",
}
MAX_EXAMPLES = 10


def scan(run: str, folder: str, table: str) -> str:
    """SQL table expression reading one output folder of a run.

    Args:
        run: Run folder, local or s3:// (s3a:// is accepted).
        folder: Stage output folder, e.g. `matches`.
        table: Value of the `type=` partition, e.g. `place`.

    Returns:
        A DuckDB `read_parquet(...)` expression.
    """
    run = run.rstrip("/").replace("s3a://", "s3://", 1)
    path = f"{run}/{folder}/type={table}/**/*.parquet"
    return f"read_parquet('{path}', union_by_name=true, hive_partitioning=false)"


def ratio(num: int, den: int) -> float:
    """num / den, or 1.0 when there is nothing to compare."""
    return num / den if den else 1.0


def load_ids(con: duckdb.DuckDBPyConnection, tag: str, run: str, table: str) -> None:
    """Create the `<tag>_ids` view: each candidate's assigned ID and its co-members."""
    con.execute(f"""
        CREATE VIEW {tag}_ids AS
        WITH m AS (SELECT DISTINCT id, assigned_id FROM {scan(run, "idMap", table)})
        SELECT id, assigned_id,
               assigned_id IN (SELECT id FROM {scan(run, "baseline", table)}) AS reused,
               -- simplification: an XOR of member hashes stands for the member set.
               bit_xor(hash(id)) OVER (PARTITION BY assigned_id) AS members
        FROM m
    """)


def load_clusters(
    con: duckdb.DuckDBPyConnection, tag: str, run: str, table: str
) -> None:
    """Create the `<tag>_members` (entity, cluster) and `<tag>_edges` views from `matches`."""
    matches = scan(run, "matches", table)
    con.execute(f"""
        CREATE VIEW {tag}_members AS
        SELECT id, cluster_id FROM {matches}
        UNION
        SELECT base_id, cluster_id FROM {matches} WHERE base_id IS NOT NULL;
        CREATE VIEW {tag}_edges AS
        SELECT DISTINCT least(id, base_id) AS u, greatest(id, base_id) AS v
        FROM {matches} WHERE match_type = 'match' AND base_id IS NOT NULL;
    """)


def load_slices(con: duckdb.DuckDBPyConnection, run: str, table: str) -> None:
    """Create the `slices` view: the reference run's candidates plus `match_sub_type`.

    A candidate's `match_sub_type` is the best across its match rows, which
    alphabetical order gives: candidate_baseline, candidate_candidate, nomatch.
    """
    matches = scan(run, "matches", table)
    con.execute(f"""
        CREATE VIEW slices AS
        SELECT c.*, t.match_sub_type
        FROM {scan(run, "candidates", table)} c
        LEFT JOIN (
            SELECT id, min(match_sub_type) AS match_sub_type FROM (
                SELECT id, match_sub_type FROM {matches}
                UNION ALL
                SELECT base_id, match_sub_type FROM {matches}
            ) GROUP BY id
        ) t USING (id)
    """)


def id_agreement(con: duckdb.DuckDBPyConnection, by: str | None = None) -> list[dict]:
    """ID agreement over the union of both runs' candidates, largest group first.

    A candidate missing from either run's idMap counts as a disagreement.

    Args:
        con: Connection holding the `a_ids`, `b_ids` and `slices` views.
        by: `slices` column to group by, or None for the overall figure.

    Returns:
        One `{"value", "n", "agreement"}` dict per group.
    """
    grp, join = "NULL", ""
    if by:
        grp = "s.grp"
        join = f'LEFT JOIN (SELECT id, any_value("{by}") AS grp FROM slices GROUP BY id) s USING (id)'
    rows = con.execute(f"""
        SELECT {grp} AS value, count(*) AS n,
               avg(CAST(coalesce(CASE
                   WHEN a.reused AND b.reused THEN a.assigned_id = b.assigned_id
                   WHEN NOT a.reused AND NOT b.reused THEN a.members = b.members
                   ELSE false END, false) AS INT)) AS agreement
        FROM a_ids a FULL JOIN b_ids b USING (id) {join}
        GROUP BY ALL ORDER BY n DESC, value
    """).fetchall()
    return [{"value": v, "n": n, "agreement": ag} for v, n, ag in rows]


def cluster_agreement(con: duckdb.DuckDBPyConnection) -> dict:
    """Pairwise precision and recall of "same cluster" over entities in both runs.

    Precision is the share of B's same-cluster pairs that A confirms; recall is
    the share of A's pairs that B keeps.
    """
    pairs_a, pairs_b, shared, n = con.execute("""
        WITH j AS (
            SELECT a.cluster_id AS ca, b.cluster_id AS cb
            FROM a_members a JOIN b_members b USING (id)
        ),
        p AS (SELECT count(*) AS k FROM j GROUP BY ca, cb),
        pa AS (SELECT count(*) AS k FROM j GROUP BY ca),
        pb AS (SELECT count(*) AS k FROM j GROUP BY cb)
        SELECT (SELECT coalesce(sum(k * (k - 1) // 2), 0) FROM pa),
               (SELECT coalesce(sum(k * (k - 1) // 2), 0) FROM pb),
               (SELECT coalesce(sum(k * (k - 1) // 2), 0) FROM p),
               (SELECT count(*) FROM j)
    """).fetchone()
    return {
        "entities_compared": n,
        "pairs_a": pairs_a,
        "pairs_b": pairs_b,
        "precision": ratio(shared, pairs_b),
        "recall": ratio(shared, pairs_a),
    }


def edge_agreement(con: duckdb.DuckDBPyConnection) -> dict:
    """Precision and recall of accepted pairs, regardless of pair orientation."""
    na, nb, both = con.execute("""
        SELECT (SELECT count(*) FROM a_edges), (SELECT count(*) FROM b_edges),
               (SELECT count(*) FROM a_edges JOIN b_edges USING (u, v))
    """).fetchone()
    return {
        "edges_a": na,
        "edges_b": nb,
        "precision": ratio(both, nb),
        "recall": ratio(both, na),
    }


def column_types(con: duckdb.DuckDBPyConnection, view: str) -> dict[str, str]:
    """Column name to DuckDB type, for a view."""
    return dict(
        con.execute(
            f"SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM {view})"
        ).fetchall()
    )


def is_numeric(column_type: str) -> bool:
    """Whether a DuckDB type is compared within the tolerance."""
    return column_type in NUMERIC_TYPES or column_type.startswith(("DECIMAL", "U"))


def mismatch_sql(
    col: str, type_a: str, type_b: str, tolerance: float
) -> tuple[str, str | None]:
    """SQL for "the two sides differ" and, for numbers, for their largest absolute gap."""
    a, b = f'a."{col}"', f'b."{col}"'
    if is_numeric(type_a) and is_numeric(type_b):
        gap = f"abs({a}::DOUBLE - {b}::DOUBLE)"
        return (
            f"{a} IS DISTINCT FROM {b} AND NOT coalesce({gap} <= {tolerance}, false)",
            f"max({gap})",
        )
    if type_a != type_b:
        a, b = f"{a}::VARCHAR", f"{b}::VARCHAR"
    return f"{a} IS DISTINCT FROM {b}", None


def diff_table(
    con: duckdb.DuckDBPyConnection, folder: str, key: list[str], tolerance: float
) -> dict:
    """Diff one output folder of runs A and B, joined on `key`.

    Args:
        con: Connection holding the `a_<folder>` and `b_<folder>` views.
        folder: Stage output folder.
        key: Row key columns.
        tolerance: Largest absolute difference at which numbers still agree.

    Returns:
        Row and key counts, columns present on one side only, columns whose type
        differs, and for each shared column with mismatches the number of
        mismatched rows (plus the largest gap for numbers).
    """
    a, b = f"a_{folder}", f"b_{folder}"
    types_a, types_b = column_types(con, a), column_types(con, b)
    keys = ", ".join(f'"{k}"' for k in key)
    rows_a, rows_b, dup_a, dup_b, only_a, only_b, both = con.execute(f"""
        SELECT (SELECT count(*) FROM {a}), (SELECT count(*) FROM {b}),
               (SELECT count(*) - count(DISTINCT ({keys})) FROM {a}),
               (SELECT count(*) - count(DISTINCT ({keys})) FROM {b}),
               (SELECT count(*) FROM {a} ANTI JOIN {b} USING ({keys})),
               (SELECT count(*) FROM {b} ANTI JOIN {a} USING ({keys})),
               (SELECT count(*) FROM {a} JOIN {b} USING ({keys}))
    """).fetchone()
    shared = [c for c in types_a if c in types_b and c not in key]
    mismatches = {}
    for c in shared:
        differs, gap = mismatch_sql(c, types_a[c], types_b[c], tolerance)
        n, max_gap = con.execute(f"""
            SELECT count(*) FILTER (WHERE {differs}), {gap or "NULL"}
            FROM {a} a JOIN {b} b USING ({keys})
        """).fetchone()
        if n:
            mismatches[c] = {"n": n} | ({"max_abs_diff": max_gap} if gap else {})
    return {
        "rows_a": rows_a,
        "rows_b": rows_b,
        "duplicate_keys_a": dup_a,
        "duplicate_keys_b": dup_b,
        "keys_only_a": only_a,
        "keys_only_b": only_b,
        "keys_both": both,
        "columns_only_a": sorted(set(types_a) - set(types_b)),
        "columns_only_b": sorted(set(types_b) - set(types_a)),
        "column_types_differ": {
            c: [types_a[c], types_b[c]] for c in shared if types_a[c] != types_b[c]
        },
        "mismatches": mismatches,
    }


def score_flips(
    con: duckdb.DuckDBPyConnection, folder: str, threshold: float
) -> dict | None:
    """Pairs whose `composite_score` lands on different sides of `threshold`.

    A flip's distance is how far the farther of its two scores sits from the
    threshold: near zero is float noise, far from it is a real difference.

    Returns:
        The flip count, the largest distance and the farthest flips, or None
        when either run lacks `composite_score`.
    """
    a, b = f"a_{folder}", f"b_{folder}"
    if not all("composite_score" in column_types(con, v) for v in (a, b)):
        return None
    keys = ", ".join(PAIR_KEY)
    rows = con.execute(f"""
        WITH s AS (
            SELECT {keys}, a.composite_score::DOUBLE AS sa, b.composite_score::DOUBLE AS sb
            FROM {a} a JOIN {b} b USING ({keys})
        )
        SELECT {keys}, sa, sb, greatest(abs(sa - {threshold}), abs(sb - {threshold})) AS distance,
               count(*) OVER () AS n
        FROM s
        WHERE (sa >= {threshold}) IS DISTINCT FROM (sb >= {threshold})
        ORDER BY distance DESC, {keys}
        LIMIT {MAX_EXAMPLES}
    """).fetchall()
    return {
        "threshold": threshold,
        "n": rows[0][-1] if rows else 0,
        "max_distance": rows[0][-2] if rows else None,
        "examples": [
            {"id": i, "base_id": bi, "score_a": sa, "score_b": sb, "distance": d}
            for i, bi, sa, sb, d, _ in rows
        ],
    }


def diff_stage(
    con: duckdb.DuckDBPyConnection,
    runs: dict[str, str],
    stage: str,
    table: str,
    tolerance: float,
    thresholds: dict[str, float],
) -> dict:
    """Diff every keyed output folder of one stage; see `diff_table` and `score_flips`.

    `runs` maps the view prefixes `a` and `b` to run folders.
    """
    report = {}
    for folder in STAGE_OUTPUTS[stage]:
        if folder not in DIFF_KEYS:
            continue
        for tag, run in runs.items():
            con.execute(
                f"CREATE VIEW {tag}_{folder} AS SELECT * FROM {scan(run, folder, table)}"
            )
        report[folder] = diff_table(con, folder, DIFF_KEYS[folder], tolerance)
        if folder in thresholds:
            report[folder]["flips"] = score_flips(con, folder, thresholds[folder])
    return report


def connect(*runs: str) -> duckdb.DuckDBPyConnection:
    """DuckDB connection, with S3 access when any run is on S3."""
    con = duckdb.connect()
    if any(r.startswith("s3") for r in runs):
        con.execute(
            "INSTALL httpfs; LOAD httpfs; CREATE SECRET (TYPE s3, PROVIDER credential_chain)"
        )
    return con


def compare(
    a_run: str,
    b_run: str,
    table: str = "place",
    group_by: list[str] = SLICES,
    stage: str | None = None,
    tolerance: float = 1e-6,
    thresholds: dict[str, float] = THRESHOLDS,
) -> dict:
    """Compare run B against reference run A.

    Args:
        a_run: Reference run folder.
        b_run: Run folder under test.
        table: Value of the `type=` partition to compare.
        group_by: Slices for the ID agreement breakdown: columns of A's
            candidates, or `match_sub_type`.
        stage: Diff this stage's outputs instead of computing the gate metrics.
        tolerance: Largest absolute difference at which numbers agree (`stage` only).
        thresholds: Score threshold per output folder (`stage` only).

    Returns:
        The report the command line prints.
    """
    runs = {"a": a_run, "b": b_run}
    con = connect(*runs.values())
    if stage:
        tables = diff_stage(con, runs, stage, table, tolerance, thresholds)
        report = {"stage": stage, "tables": tables}
        if stage != "matching":
            return report
    else:
        for tag, run in runs.items():
            load_ids(con, tag, run, table)
        load_slices(con, a_run, table)
        overall = id_agreement(con)[0]
        by = {c: id_agreement(con, c) for c in group_by}
        report = {
            "id": {"n": overall["n"], "agreement": overall["agreement"], "by": by}
        }
    for tag, run in runs.items():
        load_clusters(con, tag, run, table)
    return report | {"cluster": cluster_agreement(con), "edge": edge_agreement(con)}


def main() -> None:
    """Command line entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("a_run", help="reference run folder (local, s3:// or s3a://)")
    ap.add_argument("b_run", help="run folder under test")
    ap.add_argument(
        "--table", default="place", help="value of the type= partition (default: place)"
    )
    ap.add_argument(
        "--group-by",
        action="append",
        metavar="COL",
        help=f"ID agreement slice, repeatable (default: {' '.join(SLICES)}); '' for none",
    )
    ap.add_argument(
        "--stage",
        choices=["setup", "matching"],
        help="diff this stage's outputs instead",
    )
    ap.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="numeric tolerance for --stage (default: 1e-6)",
    )
    ap.add_argument("--blocking-threshold", type=float, default=THRESHOLDS["blocking"])
    ap.add_argument("--match-threshold", type=float, default=THRESHOLDS["scores"])
    args = ap.parse_args()
    group_by = SLICES if args.group_by is None else [c for c in args.group_by if c]
    thresholds = {"blocking": args.blocking_threshold, "scores": args.match_threshold}
    report = compare(
        args.a_run,
        args.b_run,
        args.table,
        group_by,
        args.stage,
        args.tolerance,
        thresholds,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
