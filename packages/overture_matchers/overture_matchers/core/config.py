# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""The run's settings: a frozen dataclass built from the job params.

``RunConfig.for_run`` parses the params, writes ``config.json`` into the run
folder with the installed package versions, and fails if the folder already
holds a different ``config.json``. That check stops a rerun with changed
params (or a changed matcher version) from skipping stages that ran with the
old ones. ``RunConfig`` lives only on the Spark driver; executors get plain
values from it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

from overture_matchers.core import io

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

PACKAGES_RECORDED = ("overture-matchers",)

# Job param (Scala's argument name) -> field. Params outside this table are ignored,
# so argument sets cloned from Scala runs still work.
_PARAM_FIELDS = {
    "outputPath": "output_path",
    "input": "input",
    "baseline": "baseline",
    "provider": "provider",
    "scenario": "scenario",
    "environment": "environment",
    "matchHistory": "match_history",
    "outputDebug": "output_debug",
    "embeddingsCacheTable": "embeddings_cache_table",
    "previousMetricsPath": "previous_metrics_path",
}
_REQUIRED = ("output_path", "input", "baseline", "provider")
_BOOLS = ("match_history", "output_debug")
_TABLE_MAPS = ("input", "baseline")


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
        output_path: Run folder, the base URI of the run's bundle.
        input: Candidate paths, ``{table: [paths]}``.
        baseline: Baseline paths, ``{table: [paths]}``.
        provider: Name of the provider whose records are matched.
        scenario: Matching scenario label.
        environment: Deployment environment label.
        match_history: Whether candidates with history matches skip scoring.
        output_debug: Whether intermediate outputs are also written.
        embeddings_cache_table: Table name of the embeddings cache, if any.
        previous_metrics_path: Metrics of the previous run, if any.
    """

    output_path: str
    input: dict[str, list[str]]
    baseline: dict[str, list[str]]
    provider: str
    scenario: str = "overture"
    environment: str = "dev"
    match_history: bool = False
    output_debug: bool = False
    embeddings_cache_table: str | None = None
    previous_metrics_path: str | None = None

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> RunConfig:
        """Build a config from job params, filling defaults and parsing types.

        Args:
            params: Job params keyed by Scala's argument names (``outputPath``,
                ``matchHistory``, ...). Unknown params are ignored.

        Raises:
            ValueError: A required param is missing or a value cannot be parsed.
        """
        values = {
            field: params[param]
            for param, field in _PARAM_FIELDS.items()
            if params.get(param) not in (None, "", "none")
        }
        missing = [f for f in _REQUIRED if f not in values]
        if missing:
            raise ValueError(f"missing required params: {missing}")
        for field in _BOOLS:
            if field in values:
                values[field] = _parse_bool(values[field])
        for field in _TABLE_MAPS:
            values[field] = _parse_table_map(values[field])
        return cls(**values)

    def to_json(self) -> str:
        """Every field, defaults included, plus the installed package versions."""
        record = asdict(self) | {
            "versions": {name: version(name) for name in PACKAGES_RECORDED}
        }
        return json.dumps(record, indent=2, sort_keys=True)

    @classmethod
    def for_run(cls, spark: SparkSession, params: dict[str, Any]) -> RunConfig:
        """Build the config and write ``config.json`` to the run folder.

        Args:
            spark: Session used to reach the run folder.
            params: Job params, see ``from_params``.

        Raises:
            ValueError: The run folder holds a ``config.json`` that differs.
                Start a new run folder after changing params or versions.
        """
        config = cls.from_params(params)
        path = f"{config.output_path.rstrip('/')}/config.json"
        record = config.to_json()
        if io.exists(spark, path) and json.loads(io.read_file(spark, path)) != (
            json.loads(record)
        ):
            raise ValueError(f"{path} was written with different params")
        io.write_file(spark, path, record)
        return config
