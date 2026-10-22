# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
# /// script
# requires-python = ">=3.11"
# dependencies = ["boto3", "duckdb>=1.1"]
# ///
"""Start N Glue matcher runs cloned from a template run, wait, and print their run folders as JSON.

The template is an existing Glue job run of either matcher. Its arguments
supply the input, baseline, model, jars and Spark conf, so the runs match it;
its worker type, worker count and timeout carry over too. Each run gets its
own run folder under `--output-root` and `outputDebug=true`, which compare.py
needs. A Scala template (it has `--runId`) gets a distinct `runId` under
`outputPath`; a Python template gets a distinct `outputPath`.

  noise floor     Scala template, `--runs 2`, then compare.py on the two folders.
  Python vs Scala Python template, then compare.py against a Scala run folder.
  mixed mode      Python template, `--set stages=matching --seed-from SCALA_RUN`:
                  each run folder first gets a copy of the Scala run's outputs of
                  the stages before the first one in `stages`, then compare.py
                  `--stage matching` against SCALA_RUN.

    AWS_PROFILE=x uv run parity/launch.py --job-name JOB --template-run-id jr_... \\
        --output-root s3://BUCKET/parity/NAME [--runs 2] [--set key=value] [--seed-from RUN]

Both sides of a comparison must use the same model version. Template arguments
that expire, such as jar URLs carrying a token, need fresh values via `--set`.
The job's maximum concurrent runs must allow `--runs`. Progress goes to stderr;
the exit code is non-zero unless every run succeeded.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone

from compare import STAGE_OUTPUTS

TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED", "ERROR", "TIMEOUT"}
POLL_SECONDS = 60
CLONED_RUN_FIELDS = ("WorkerType", "NumberOfWorkers", "Timeout")


def build_arguments(
    template: dict[str, str], output_root: str, name: str, overrides: dict[str, str]
) -> tuple[dict, str]:
    """Job arguments for one run, cloned from the template's.

    Args:
        template: The template run's `Arguments`.
        output_root: Prefix under which each run gets its folder, s3:// or s3a://.
        name: This run's name, unique under `output_root`.
        overrides: Job arguments to set, keyed with their `--` prefix.

    Returns:
        The job arguments and the run folder as an s3:// URI.
    """
    args = {k: v for k, v in template.items() if k.startswith("--")}
    root = output_root.rstrip("/").split("://", 1)[-1]
    scheme = template.get("--outputPath", "s3://").split("://", 1)[0]
    if "--runId" in template:
        args |= {"--outputPath": f"{scheme}://{root}", "--runId": name}
    else:
        args["--outputPath"] = f"{scheme}://{root}/{name}"
    args |= {"--outputDebug": "true"} | overrides
    return args, f"s3://{root}/{name}"


def parse_overrides(pairs: list[str]) -> dict[str, str]:
    """`key=value` strings to job arguments keyed `--key`."""
    bad = [p for p in pairs if "=" not in p]
    if bad:
        raise SystemExit(f"--set takes key=value, got: {bad}")
    return {"--" + k: v for k, v in (p.split("=", 1) for p in pairs)}


def seed_folders(stages: str | None) -> list[str]:
    """Output folders of the stages before the first stage in a `stages` argument.

    Raises:
        SystemExit: When `stages` is unset or starts at the first stage, so
            there is nothing to seed.
    """
    order = list(STAGE_OUTPUTS)
    requested = [s.strip().lower() for s in (stages or "").split(",") if s.strip()]
    unknown = set(requested) - set(order)
    if unknown:
        raise SystemExit(f"unknown stages {sorted(unknown)}; valid: {order}")
    first = min((order.index(s) for s in requested), default=0)
    if first == 0:
        raise SystemExit(
            "--seed-from needs --set stages=... to skip at least the first stage"
        )
    return [f for s in order[:first] for f in STAGE_OUTPUTS[s]]


def split_s3(uri: str) -> tuple[str, str]:
    """`s3://bucket/key` to `(bucket, key)`."""
    bucket, _, key = uri.split("://", 1)[-1].partition("/")
    return bucket, key


def seed(s3, source_run: str, run_folder: str, folders: list[str]) -> None:
    """Copy `folders` of `source_run` into `run_folder`."""
    src_bucket, src_root = split_s3(source_run.rstrip("/"))
    dst_bucket, dst_root = split_s3(run_folder)
    for folder in folders:
        prefix = f"{src_root}/{folder}/"
        for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=src_bucket, Prefix=prefix
        ):
            for obj in page.get("Contents", []):
                key = f"{dst_root}/{obj['Key'][len(src_root) + 1 :]}"
                s3.copy({"Bucket": src_bucket, "Key": obj["Key"]}, dst_bucket, key)
    print(
        f"seeded {run_folder} from {source_run}: {folders}", file=sys.stderr, flush=True
    )


def wait(glue, job_name: str, runs: list[dict]) -> None:
    """Poll until every run is terminal, filling in `state` and `error`."""
    while pending := [r for r in runs if r["state"] not in TERMINAL]:
        time.sleep(POLL_SECONDS)
        for r in pending:
            run = glue.get_job_run(JobName=job_name, RunId=r["job_run_id"])["JobRun"]
            r["state"], r["error"] = run["JobRunState"], run.get("ErrorMessage")
            if r["state"] in TERMINAL:
                print(
                    f"{r['job_run_id']}: {r['state']} {r['error'] or ''}",
                    file=sys.stderr,
                    flush=True,
                )


def main() -> None:
    """Command line entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--job-name", required=True, help="Glue job to run")
    ap.add_argument(
        "--template-run-id",
        required=True,
        help="Glue run of that job whose arguments to clone",
    )
    ap.add_argument(
        "--output-root",
        required=True,
        help="S3 prefix; each run gets a folder under it",
    )
    ap.add_argument("--runs", type=int, default=2, help="number of runs (default: 2)")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="set job argument --KEY",
    )
    ap.add_argument(
        "--seed-from",
        metavar="RUN",
        help="run folder whose earlier-stage outputs each run starts from",
    )
    ap.add_argument("--region", help="AWS region (default: from the AWS config)")
    args = ap.parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    overrides = parse_overrides(args.set)

    import boto3  # imported here so the tests need no boto3

    glue = boto3.client("glue", region_name=args.region)
    template = glue.get_job_run(JobName=args.job_name, RunId=args.template_run_id)[
        "JobRun"
    ]
    run_fields = {k: template[k] for k in CLONED_RUN_FIELDS if k in template}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    builds = [
        build_arguments(
            template.get("Arguments", {}),
            args.output_root,
            f"parity_{stamp}_{i}",
            overrides,
        )
        for i in range(args.runs)
    ]
    if args.seed_from:
        folders = seed_folders(builds[0][0].get("--stages"))
        s3 = boto3.client("s3", region_name=args.region)
        for _, run_folder in builds:
            seed(s3, args.seed_from, run_folder, folders)

    runs = []
    for job_args, run_folder in builds:
        job_run_id = glue.start_job_run(
            JobName=args.job_name, Arguments=job_args, **run_fields
        )["JobRunId"]
        print(f"started {job_run_id}: {run_folder}", file=sys.stderr, flush=True)
        runs.append(
            {
                "run_folder": run_folder,
                "job_run_id": job_run_id,
                "state": "STARTING",
                "error": None,
            }
        )
    wait(glue, args.job_name, runs)
    print(json.dumps(runs, indent=2))
    if any(r["state"] != "SUCCEEDED" for r in runs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
