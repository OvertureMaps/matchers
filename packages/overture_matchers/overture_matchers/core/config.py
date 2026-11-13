# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Run settings, built from the job params."""

from __future__ import annotations

import contextlib
import json
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from overture_spark.hadoop_fs import list_files, read_json, write_text
from overture_spark.job import is_unset
from py4j.protocol import Py4JJavaError

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
    "baselineProvider": "baseline_provider",
    "matchHistory": "match_history",
    "outputDebug": "output_debug",
    "embeddingsCacheTable": "embeddings_cache_table",
    "previousMetricsPath": "previous_metrics_path",
    "modelJar": "model_jar",
    "blockingResolution": "blocking_resolution",
}
_REQUIRED = ("output_path", "input", "baseline", "provider")
_BOOLS = ("match_history", "output_debug")
_PATH_MAPS = ("input", "baseline")

LIBPOSTAL_DATA_DIR_CONF = "spark.sedona.libpostal.dataDir"
_SEDONA_PARSE_ADDRESS_CLASS = "org.apache.spark.sql.sedona_sql.expressions.ParseAddress"


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "false"):
        return text == "true"
    raise ValueError(f"expected true or false, got {value!r}")


def _parse_blocking_resolution(value: Any) -> int:
    if not isinstance(value, bool) and isinstance(value, (int, str)):
        with contextlib.suppress(ValueError):
            resolution = int(value)
            if 1 <= resolution <= 23:
                return resolution
    raise ValueError(
        f"blockingResolution must be an integer from 1 to 23, got {value!r}"
    )


def _sedona_version(spark: SparkSession) -> str | None:
    """The Sedona jar's artifact and version, else the Python package's version.

    The jar is read first because address parsing runs there, and Glue loads
    the jar and the Python package separately.
    """
    jvm = spark.sparkContext._jvm
    loader = jvm.java.lang.Thread.currentThread().getContextClassLoader()
    with contextlib.suppress(Py4JJavaError):
        jar = jvm.java.lang.Class.forName(
            _SEDONA_PARSE_ADDRESS_CLASS, False, loader
        ).getPackage()
        if jar is not None and jar.getImplementationVersion():
            return f"{jar.getImplementationTitle()}:{jar.getImplementationVersion()}"
    with contextlib.suppress(PackageNotFoundError):
        return version("apache-sedona")
    return None


def rewrite_s3_to_s3a(path: str) -> str:
    """Rewrite an ``s3://`` URI to ``s3a://`` so Hadoop's S3A connector handles it."""
    return "s3a://" + path[len("s3://") :] if path.startswith("s3://") else path


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
        path_map[feature_type] = [rewrite_s3_to_s3a(p) for p in paths]
    return path_map


@dataclass(frozen=True)
class RunConfig:
    """Settings for one matching run.

    Attributes:
        baseline_provider: Provider to stamp on a baseline that has no
            ``provider`` column, such as a Places release. Places only.
        match_history: Whether candidates with history matches skip scoring.
        output_debug: Whether matching also writes ``matches_validated``.
        model_jar: The ``places-matcher-model`` jar. Only places reads it.
        blocking_resolution: Quadkey zoom (1 to 23) for ``blocking_keys`` and
            ``signature`` in places setup. Places defaults to 16 when unset.
    """

    output_path: str
    input: dict[str, list[str]]
    baseline: dict[str, list[str]]
    provider: str
    baseline_provider: str | None = None
    # Scala's theme configs default this to true.
    match_history: bool = True
    output_debug: bool = True
    embeddings_cache_table: str | None = None
    previous_metrics_path: str | None = None
    model_jar: str | None = None
    blocking_resolution: int | None = None

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
        values["output_path"] = rewrite_s3_to_s3a(values["output_path"].rstrip("/"))
        for field in ("previous_metrics_path", "model_jar"):
            if field in values:
                values[field] = rewrite_s3_to_s3a(values[field])
        for field in _BOOLS:
            if field in values:
                values[field] = _parse_bool(values[field])
        if "blocking_resolution" in values:
            values["blocking_resolution"] = _parse_blocking_resolution(
                values["blocking_resolution"]
            )
        for field in _PATH_MAPS:
            values[field] = _parse_path_map(field, values[field])
        return cls(**values)

    def to_json(self, spark: SparkSession) -> str:
        """Every field, the libpostal data dir and the library versions."""
        record = asdict(self) | {
            "libpostal_data_dir": spark.conf.get(LIBPOSTAL_DATA_DIR_CONF, None),
            "versions": {
                "overture-matchers": version("overture-matchers"),
                "phonenumbers": version("phonenumbers"),
                "sedona": _sedona_version(spark),
            },
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
        record = config.to_json(spark)
        if io.any_stage_done(spark, config.output_path) and list_files(spark, path):
            old, new = read_json(spark, path), json.loads(record)
            changed = sorted(
                k for k in old.keys() | new.keys() if old.get(k) != new.get(k)
            )
            if changed:
                raise ValueError(f"{path} differs in {changed}; start a new run folder")
        write_text(spark, path, record)
        return config
