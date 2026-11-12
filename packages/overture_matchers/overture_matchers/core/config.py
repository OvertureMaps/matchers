# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Run settings, built from the job params."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

from overture_spark.hadoop_fs import list_files, read_json, write_text
from overture_spark.job import is_unset

from overture_matchers.core import io

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

# Job param (Scala's argument name) -> field. Params outside this table are ignored,
# so argument sets cloned from Scala runs still work. Job params keep Scala's names
# until cutover. After that, the DAG can send field names and this table goes.
_PARAM_FIELDS = {
    "outputPath": "output_path",
    "input": "input",
    "baseline": "baseline",
    "provider": "provider",
    "matchHistory": "match_history",
    "embeddingsCacheTable": "embeddings_cache_table",
    "previousMetricsPath": "previous_metrics_path",
    "modelJar": "model_jar",
}
_REQUIRED = ("output_path", "input", "baseline", "provider")
_BOOLS = ("match_history",)
_PATH_MAPS = ("input", "baseline")


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "false"):
        return text == "true"
    raise ValueError(f"expected true or false, got {value!r}")


def _parse_path_map(param: str, value: Any) -> dict[str, list[str]]:
    """Accept ``{feature_type: path | [paths]}`` as a dict or a JSON string."""
    error = TypeError(
        f"{param} must be a {{feature_type: path | [paths]}} map, got {value!r}"
    )
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            raise error from None
    if not isinstance(value, dict):
        raise error
    path_map = {}
    for feature_type, paths in value.items():
        paths = [paths] if isinstance(paths, str) else paths
        if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
            raise error
        path_map[feature_type] = paths
    return path_map


@dataclass(frozen=True)
class RunConfig:
    """Settings for one matching run.

    Attributes:
        match_history: Whether candidates with history matches skip scoring.
        model_jar: The ``places-matcher-model`` jar. Only places reads it.
    """

    output_path: str
    input: dict[str, list[str]]
    baseline: dict[str, list[str]]
    provider: str
    # Scala's theme configs default this to true.
    match_history: bool = True
    embeddings_cache_table: str | None = None
    previous_metrics_path: str | None = None
    model_jar: str | None = None

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> RunConfig:
        """Build a config from params keyed by Scala's argument names.

        Unknown params are ignored.
        """
        values = {
            field: params[param]
            for param, field in _PARAM_FIELDS.items()
            if not is_unset(params.get(param))
        }
        missing = [f for f in _REQUIRED if f not in values]
        if missing:
            raise ValueError(f"missing required params: {missing}")
        values["output_path"] = values["output_path"].rstrip("/")
        for field in _BOOLS:
            if field in values:
                values[field] = _parse_bool(values[field])
        for field in _PATH_MAPS:
            values[field] = _parse_path_map(field, values[field])
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
            ValueError: A stage has finished and the folder holds a different
                ``config.json``. A rerun with changed params would skip stages
                that ran with the old ones. Before any stage finishes, a new
                ``config.json`` replaces the old one, so a fixed retry works.
        """
        config = cls.from_params(params)
        path = io.run_uri(config.output_path, "config.json")
        record = config.to_json()
        if io.any_stage_done(spark, config.output_path) and list_files(spark, path):
            old, new = read_json(spark, path), json.loads(record)
            changed = sorted(
                k for k in old.keys() | new.keys() if old.get(k) != new.get(k)
            )
            if changed:
                raise ValueError(f"{path} differs in {changed}; start a new run folder")
        write_text(spark, path, record)
        return config
