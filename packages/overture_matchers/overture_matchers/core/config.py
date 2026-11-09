# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Run settings, built from the job params."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

from overture_spark.hadoop_fs import list_files, read_text

from overture_matchers.core import io

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

# Job param (Scala's argument name) -> field. Params outside this table are ignored,
# so argument sets cloned from Scala runs still work.
_PARAM_FIELDS = {
    "outputPath": "output_path",
    "input": "input",
    "baseline": "baseline",
    "provider": "provider",
    "matchHistory": "match_history",
    "outputDebug": "output_debug",
    "embeddingsCacheTable": "embeddings_cache_table",
    "previousMetricsPath": "previous_metrics_path",
}
_REQUIRED = ("output_path", "input", "baseline", "provider")
_BOOLS = ("match_history", "output_debug")
_TABLE_MAPS = ("input", "baseline")


def _is_unset(value: Any) -> bool:
    # The DAG sends "none" (or a rendered Python None) for "use the default".
    return value is None or str(value).strip().lower() in ("", "none")


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "false"):
        return text == "true"
    raise ValueError(f"expected true or false, got {value!r}")


def _parse_table_map(value: Any) -> dict[str, list[str]]:
    """Accept ``{table: path | [paths]}`` as a dict or a JSON string."""
    if isinstance(value, str):
        value = json.loads(value)
    return {
        table: [paths] if isinstance(paths, str) else list(paths)
        for table, paths in value.items()
    }


@dataclass(frozen=True)
class RunConfig:
    """Settings for one matching run.

    Attributes:
        match_history: Whether candidates with history matches skip scoring.
        output_debug: Whether intermediate outputs are also written.
    """

    output_path: str
    input: dict[str, list[str]]
    baseline: dict[str, list[str]]
    provider: str
    # Scala's theme configs default both to true.
    match_history: bool = True
    output_debug: bool = True
    embeddings_cache_table: str | None = None
    previous_metrics_path: str | None = None

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> RunConfig:
        """Build a config from params keyed by Scala's argument names.

        Unknown params are ignored.
        """
        values = {
            field: params[param]
            for param, field in _PARAM_FIELDS.items()
            if not _is_unset(params.get(param))
        }
        missing = [f for f in _REQUIRED if f not in values]
        if missing:
            raise ValueError(f"missing required params: {missing}")
        for field in _BOOLS:
            if field in values:
                values[field] = _parse_bool(values[field])
        # One spelling, so config.json matches across s3:// and s3a:// attempts.
        values["output_path"] = io.s3a(values["output_path"])
        for field in _TABLE_MAPS:
            values[field] = _parse_table_map(values[field])
        return cls(**values)

    def to_json(self) -> str:
        """Every field plus the matcher version."""
        record = asdict(self) | {
            "versions": {"overture-matchers": version("overture-matchers")}
        }
        return json.dumps(record, indent=2, sort_keys=True)

    @classmethod
    def for_run(cls, spark: SparkSession, params: dict[str, Any]) -> RunConfig:
        """Build the config and write ``config.json`` to the run folder.

        Raises:
            ValueError: The folder holds a different ``config.json``. Without
                this check, a rerun with changed params would skip stages that
                ran with the old ones.
        """
        config = cls.from_params(params)
        path = io.run_uri(config.output_path, "config.json")
        record = config.to_json()
        if list_files(spark, path) and json.loads(read_text(spark, path)) != (
            json.loads(record)
        ):
            raise ValueError(f"{path} was written with different params")
        io.write_file(spark, path, record)
        return config
