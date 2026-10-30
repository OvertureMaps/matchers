# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.1"]
# ///
"""Compare two matcher runs on the same input and print where they differ, as JSON.

A_RUN (usually Scala) and B_RUN (usually Python) are the folders a run writes
its stage outputs to, laid out as `<stage output>/type=<table>/`. Both runs
need `outputDebug=true` for the ID map. The report has two parts:

  ids     share of candidate records that get the same ID in both runs, overall
          and broken down by country and by match type. New IDs are random, so
          two new IDs count as the same when they cover the same records.
  groups  how far the runs agree on which records belong together. Group IDs
          are arbitrary, so the report counts pairs of records grouped together
          in each run, and the share of each run's pairs the other run also has.

    uv run parity/compare.py A_RUN B_RUN [--table place]

Runs are local folders or s3:// URIs; S3 credentials come from the AWS config
chain, so AWS_PROFILE applies. Both runs must use the same model version.
"""

import argparse
import json

import duckdb

# Report key to the column it breaks ID agreement down by.
BREAKDOWNS = {"by_country": "country", "by_match_type": "match_sub_type"}


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


def share(num: int, den: int) -> float:
    """Return num / den, or 1.0 when there is nothing to compare."""
    return num / den if den else 1.0


def load_ids(con: duckdb.DuckDBPyConnection, tag: str, run: str, table: str) -> None:
    """Create the `<tag>_ids` view: each candidate's ID and the records sharing it."""
    con.execute(f"""
        CREATE VIEW {tag}_ids AS
        WITH m AS (SELECT DISTINCT id, assigned_id FROM {scan(run, "idMap", table)})
        SELECT id, assigned_id,
               assigned_id IN (SELECT id FROM {scan(run, "baseline", table)}) AS reused,
               -- simplification: an XOR of record hashes stands for the record set.
               bit_xor(hash(id)) OVER (PARTITION BY assigned_id) AS sharing
        FROM m
    """)


def load_groups(con: duckdb.DuckDBPyConnection, tag: str, run: str, table: str) -> None:
    """Create the `<tag>_groups` view: (record, group) rows from `matches`."""
    matches = scan(run, "matches", table)
    con.execute(f"""
        CREATE VIEW {tag}_groups AS
        SELECT id, cluster_id FROM {matches}
        UNION
        SELECT base_id, cluster_id FROM {matches} WHERE base_id IS NOT NULL
    """)


def load_breakdowns(con: duckdb.DuckDBPyConnection, run: str, table: str) -> None:
    """Create the `breakdowns` view: the reference run's candidates plus `match_sub_type`.

    A candidate's `match_sub_type` is the best across its match rows, which
    alphabetical order gives: candidate_baseline, candidate_candidate, nomatch.
    """
    matches = scan(run, "matches", table)
    con.execute(f"""
        CREATE VIEW breakdowns AS
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


def same_ids(con: duckdb.DuckDBPyConnection, by: str | None = None) -> list[tuple]:
    """Share of candidates with the same ID in both runs, largest group first.

    A candidate missing from either run's idMap counts as different.

    Args:
        con: Connection holding the `a_ids`, `b_ids` and `breakdowns` views.
        by: `breakdowns` column to group by, or None for all candidates.

    Returns:
        One `(value, records, same_id_share)` row per group.
    """
    grp, join = "NULL", ""
    if by:
        grp = "s.grp"
        join = f'LEFT JOIN (SELECT id, any_value("{by}") AS grp FROM breakdowns GROUP BY id) s USING (id)'
    return con.execute(f"""
        SELECT {grp}, count(*),
               avg(CAST(coalesce(CASE
                   WHEN a.reused AND b.reused THEN a.assigned_id = b.assigned_id
                   WHEN NOT a.reused AND NOT b.reused THEN a.sharing = b.sharing
                   ELSE false END, false) AS INT))
        FROM a_ids a FULL JOIN b_ids b USING (id) {join}
        GROUP BY ALL ORDER BY 2 DESC, 1
    """).fetchall()


def same_groups(con: duckdb.DuckDBPyConnection) -> dict:
    """Pairs of records grouped together in each run, over records in both runs."""
    in_a, in_b, in_both, n = con.execute("""
        WITH j AS (
            SELECT a.cluster_id AS ga, b.cluster_id AS gb
            FROM a_groups a JOIN b_groups b USING (id)
        ),
        both_ AS (SELECT count(*) AS k FROM j GROUP BY ga, gb),
        a AS (SELECT count(*) AS k FROM j GROUP BY ga),
        b AS (SELECT count(*) AS k FROM j GROUP BY gb)
        SELECT (SELECT coalesce(sum(k * (k - 1) // 2), 0) FROM a),
               (SELECT coalesce(sum(k * (k - 1) // 2), 0) FROM b),
               (SELECT coalesce(sum(k * (k - 1) // 2), 0) FROM both_),
               (SELECT count(*) FROM j)
    """).fetchone()
    return {
        "records_in_both_runs": n,
        "pairs_grouped_together_in_a": in_a,
        "pairs_grouped_together_in_b": in_b,
        "pairs_grouped_together_in_both": in_both,
        "share_of_a_pairs_also_in_b": share(in_both, in_a),
        "share_of_b_pairs_also_in_a": share(in_both, in_b),
    }


def connect(*runs: str) -> duckdb.DuckDBPyConnection:
    """DuckDB connection, with S3 access when any run is on S3."""
    con = duckdb.connect()
    if any(r.startswith("s3") for r in runs):
        con.execute(
            "INSTALL httpfs; LOAD httpfs; CREATE SECRET (TYPE s3, PROVIDER credential_chain)"
        )
    return con


def compare(a_run: str, b_run: str, table: str = "place") -> dict:
    """Compare run B against reference run A.

    Args:
        a_run: Reference run folder.
        b_run: Run folder being checked.
        table: Value of the `type=` partition to compare.

    Returns:
        The report the command line prints.
    """
    runs = {"a": a_run, "b": b_run}
    con = connect(*runs.values())
    for tag, run in runs.items():
        load_ids(con, tag, run, table)
        load_groups(con, tag, run, table)
    load_breakdowns(con, a_run, table)
    _, records, same = same_ids(con)[0]
    ids = {"records": records, "same_id_share": same}
    for key, col in BREAKDOWNS.items():
        ids[key] = [
            {col: v, "records": n, "same_id_share": s} for v, n, s in same_ids(con, col)
        ]
    return {"ids": ids, "groups": same_groups(con)}


def main() -> None:
    """Command line entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("a_run", help="reference run folder (local, s3:// or s3a://)")
    ap.add_argument("b_run", help="run folder being checked")
    ap.add_argument(
        "--table", default="place", help="value of the type= partition (default: place)"
    )
    args = ap.parse_args()
    print(json.dumps(compare(args.a_run, args.b_run, args.table), indent=2))


if __name__ == "__main__":
    main()
