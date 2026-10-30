# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.1"]
# ///
"""Compare two matcher runs on the same input and print where they differ, as JSON.

A_RUN (usually Scala) and B_RUN (usually Python) are the folders a run writes
its stage outputs to, laid out as `<stage output>/type=<feature_type>/`. Both runs
need `outputDebug=true` for the ID map. The report has two parts:

  ids     share of candidate records that get the same ID in both runs, overall
          and broken down by country and by match type. New IDs are random, so
          two new IDs count as the same when they cover the same records.
  groups  how far the runs agree on which records belong together. Group IDs
          are arbitrary, so the report counts pairs of records grouped together
          in each run, and the share of each run's pairs the other run also has.

    uv run parity/compare.py A_RUN B_RUN [--feature-type place]

Runs are local folders or s3:// URIs; S3 credentials come from the AWS config
chain, so AWS_PROFILE applies. Both runs must use the same model version.
"""

import argparse
import json

import duckdb

# Report key to the column it breaks ID agreement down by.
BREAKDOWNS = {"by_country": "country", "by_match_type": "match_sub_type"}


def scan(run: str, folder: str, feature_type: str) -> str:
    """SQL table expression reading one output folder of a run.

    Args:
        run: Run folder, local or s3:// (s3a:// is accepted).
        folder: Stage output folder, e.g. `matches`.
        feature_type: Value of the `type=` partition, e.g. `place`.

    Returns:
        A DuckDB `read_parquet(...)` expression.
    """
    run = run.rstrip("/").replace("s3a://", "s3://", 1)
    path = f"{run}/{folder}/type={feature_type}/**/*.parquet"
    return f"read_parquet('{path}', union_by_name=true, hive_partitioning=false)"


def share(num: int, den: int) -> float:
    """Return num / den, or 1.0 when there is nothing to compare."""
    return num / den if den else 1.0


def create_ids_view(
    con: duckdb.DuckDBPyConnection, run_label: str, run: str, feature_type: str
) -> None:
    """Create the `a_ids` or `b_ids` view: each candidate's assigned ID, whether it came from the baseline, and a hash of the candidates sharing it.

    The view looks like:
        id    assigned_id    got_baseline_id    new_id_group_hash
        ----  -------------  -----------------  -----------------
        1     100            true               1234567890
        2     100            true               1234567890
        3     101            false              9876543210

    Args:
        con: DuckDB connection.
        run_label: Either 'a' or 'b', used to name the view.
        run: Run folder, local or s3://.
        feature_type: Value of the `type=` partition, e.g. `place`.
    """
    con.execute(f"""
        CREATE VIEW {run_label}_ids AS
        WITH m AS (SELECT DISTINCT id, assigned_id FROM {scan(run, "idMap", feature_type)})
        SELECT id, assigned_id,
               assigned_id IN (SELECT id FROM {scan(run, "baseline", feature_type)}) AS got_baseline_id,
               -- XOR all of the ids in a group to get a deterministic order-independent fingerprint of the group
               -- I guess this risks hash collisions, but it's a 64-bit hash, so the probability is very low.
               bit_xor(hash(id)) OVER (PARTITION BY assigned_id) AS new_id_group_hash
        FROM m
    """)


def create_groups_view(
    con: duckdb.DuckDBPyConnection, run_label: str, run: str, feature_type: str
) -> None:
    """Create the `<run_label>_groups` view: (record, group) rows from `matches`."""
    matches = scan(run, "matches", feature_type)
    con.execute(f"""
        CREATE VIEW {run_label}_groups AS
        SELECT id, cluster_id FROM {matches}
        UNION
        SELECT base_id, cluster_id FROM {matches} WHERE base_id IS NOT NULL
    """)


def create_candidates_with_match_type_view(
    con: duckdb.DuckDBPyConnection, run: str, feature_type: str
) -> None:
    """Create the `candidates_with_match_type` view: the reference run's candidates plus `match_sub_type`.

    A candidate's `match_sub_type` is the best across its match rows, which
    alphabetical order gives: candidate_baseline, candidate_candidate, nomatch.
    """
    matches = scan(run, "matches", feature_type)
    con.execute(f"""
        CREATE VIEW candidates_with_match_type AS
        SELECT c.*, t.match_sub_type
        FROM {scan(run, "candidates", feature_type)} c
        LEFT JOIN (
            SELECT id, min(match_sub_type) AS match_sub_type FROM (
                SELECT id, match_sub_type FROM {matches}
                UNION ALL
                SELECT base_id, match_sub_type FROM {matches}
            ) GROUP BY id
        ) t USING (id)
    """)


def id_agreement(
    con: duckdb.DuckDBPyConnection, breakdown_column: str | None = None
) -> list[tuple]:
    """Share of candidates with the same ID in both runs, largest group first.

    A candidate missing from either run's idMap counts as different.

    Args:
        con: Connection holding the `a_ids`, `b_ids` and `candidates_with_match_type` views.
        breakdown_column: `candidates_with_match_type` column to group by, or None for all candidates.

    Returns:
        One `(breakdown_value, records, same_id_share)` row per group.
    """
    value_expr, join = "NULL", ""
    if breakdown_column:
        value_expr = "c.breakdown_value"
        join = f'LEFT JOIN (SELECT id, any_value("{breakdown_column}") AS breakdown_value FROM candidates_with_match_type GROUP BY id) c USING (id)'
    return con.execute(f"""
        SELECT {value_expr}, count(*),
               avg(CAST(coalesce(CASE
                   WHEN a.got_baseline_id AND b.got_baseline_id THEN a.assigned_id = b.assigned_id
                   WHEN NOT a.got_baseline_id AND NOT b.got_baseline_id THEN a.new_id_group_hash = b.new_id_group_hash
                   ELSE false END, false) AS INT))
        FROM a_ids a FULL JOIN b_ids b USING (id) {join}
        GROUP BY ALL ORDER BY 2 DESC, 1
    """).fetchall()


def group_agreement(con: duckdb.DuckDBPyConnection) -> dict:
    """Pairs of records grouped together in each run, over records in both runs."""
    pairs_a, pairs_b, pairs_both, n_records = con.execute("""
        WITH joined AS (
            SELECT a.cluster_id AS cluster_a, b.cluster_id AS cluster_b
            FROM a_groups a JOIN b_groups b USING (id)
        ),
        sizes_both AS (SELECT count(*) AS size FROM joined GROUP BY cluster_a, cluster_b),
        sizes_a AS (SELECT count(*) AS size FROM joined GROUP BY cluster_a),
        sizes_b AS (SELECT count(*) AS size FROM joined GROUP BY cluster_b)
        SELECT (SELECT coalesce(sum(size * (size - 1) // 2), 0) FROM sizes_a),
               (SELECT coalesce(sum(size * (size - 1) // 2), 0) FROM sizes_b),
               (SELECT coalesce(sum(size * (size - 1) // 2), 0) FROM sizes_both),
               (SELECT count(*) FROM joined)
    """).fetchone()
    return {
        "records_in_both_runs": n_records,
        "pairs_grouped_together_in_a": pairs_a,
        "pairs_grouped_together_in_b": pairs_b,
        "pairs_grouped_together_in_both": pairs_both,
        "share_of_a_pairs_also_in_b": share(pairs_both, pairs_a),
        "share_of_b_pairs_also_in_a": share(pairs_both, pairs_b),
    }


def connect(*runs: str) -> duckdb.DuckDBPyConnection:
    """DuckDB connection, with S3 access when any run is on S3."""
    con = duckdb.connect()
    if any(r.startswith("s3") for r in runs):
        con.execute(
            "INSTALL httpfs; LOAD httpfs; CREATE SECRET (TYPE s3, PROVIDER credential_chain)"
        )
    return con


def compare(a_run: str, b_run: str, feature_type: str = "place") -> dict:
    """Compare run B against reference run A.

    Args:
        a_run: Reference run folder.
        b_run: Run folder being checked.
        feature_type: Value of the `type=` partition to compare.

    Returns:
        The report the command line prints.
    """
    runs = {"a": a_run, "b": b_run}
    con = connect(*runs.values())
    for run_label, run in runs.items():
        create_ids_view(con, run_label, run, feature_type)
        create_groups_view(con, run_label, run, feature_type)
    create_candidates_with_match_type_view(con, a_run, feature_type)
    _, records, same_id_share = id_agreement(con)[0]
    ids = {"records": records, "same_id_share": same_id_share}
    for key, column in BREAKDOWNS.items():
        ids[key] = [
            {column: v, "records": n, "same_id_share": s}
            for v, n, s in id_agreement(con, column)
        ]
    return {"ids": ids, "groups": group_agreement(con)}


def main() -> None:
    """Command line entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("a_run", help="reference run folder (local, s3:// or s3a://)")
    ap.add_argument("b_run", help="run folder being checked")
    ap.add_argument(
        "--feature-type",
        default="place",
        help="value of the type= partition (default: place)",
    )
    args = ap.parse_args()
    print(json.dumps(compare(args.a_run, args.b_run, args.feature_type), indent=2))


if __name__ == "__main__":
    main()
