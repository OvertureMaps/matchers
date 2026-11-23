# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""The places model: settings read from the model jar, and the engines workers run.

The driver reads the blocking settings once into ``PlacesModel``. The engine
functions return ``Column`` callables that read the booster and the name
sketch out of the jar ``sc.addFile`` shipped to workers, once per process.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow.parquet as pq
from pyspark import SparkFiles
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, MapType, StringType

from overture_matchers.places import features
from overture_matchers.places.model_jar import open_model_jar
from overture_matchers.sketches import CountMinSketch

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pyspark.sql import Column, SparkSession

BLOCKING_CONFIG_RESOURCE = "places_matcher_model/metadata/blocking_config"
BLOCKING_BOOSTER_RESOURCE = "places_matcher_model/data/blocking_model"
NAME_SKETCH_RESOURCE = "config/places/cms-names.parquet"


@dataclass(frozen=True)
class PlacesModel:
    """Settings of the model jar.

    Attributes:
        blocking_features: Blocking features in booster order.
        blocking_threshold: Smallest blocking score that keeps a pair.
        blocking_radius_meters: Largest distance in meters between the places of a pair.
    """

    blocking_features: tuple[str, ...]
    blocking_threshold: float
    blocking_radius_meters: float

    @classmethod
    def load(cls, spark: SparkSession, model_jar: str) -> PlacesModel:
        """Read the settings from ``model_jar`` and check its booster's XGBoost version."""
        import xgboost

        with open_model_jar(spark, model_jar) as jar:
            check_xgboost_version(
                jar.read(BLOCKING_BOOSTER_RESOURCE), xgboost.__version__
            )
            return cls.from_config(json.loads(jar.read(BLOCKING_CONFIG_RESOURCE)))

    @classmethod
    def from_config(cls, config: dict) -> PlacesModel:
        """Settings from the jar's parsed blocking config.

        Raises:
            ValueError: The blocking feature order names a feature that
                ``features.py`` does not produce.
        """
        feature_order = tuple(config["feature_order"])
        unknown_features = [
            feature_name
            for feature_name in feature_order
            if feature_name not in features.BLOCKING_FEATURES
        ]
        if unknown_features:
            raise ValueError(
                f"blocking feature_order needs {unknown_features}, but features.py "
                f"produces only {sorted(features.BLOCKING_FEATURES)}"
            )
        return cls(
            blocking_features=feature_order,
            blocking_threshold=float(config["prediction_threshold"]),
            blocking_radius_meters=float(config["spatial_radius_m"]),
        )


def _read_jar_resource(resource_path: str, jar_name: str) -> bytes:
    with zipfile.ZipFile(SparkFiles.get(jar_name)) as jar:
        return jar.read(resource_path)


@cache
def load_xgboost_booster(jar_name: str, resource_path: str):
    """The XGBoost booster stored at ``resource_path`` in the jar. It predicts on one thread."""
    import xgboost

    model_bytes = _read_jar_resource(resource_path, jar_name)
    booster = xgboost.Booster({"nthread": 1})
    booster.load_model(bytearray(model_bytes))
    return booster


def check_xgboost_version(model_bytes: bytes, installed_version: str) -> None:
    """Raise unless ``installed_version`` XGBoost reads the JSON booster ``model_bytes`` as written.

    XGBoost reads models from older versions only. An older reader can misread
    newer fields and silently shift every score.
    """
    try:
        writer_version = tuple(json.loads(model_bytes)["version"][:2])
    except ValueError as error:
        raise RuntimeError("The booster must be saved as XGBoost JSON.") from error
    reader_version = tuple(int(part) for part in installed_version.split(".")[:2])
    if reader_version < writer_version or reader_version[0] != writer_version[0]:
        raise RuntimeError(
            f"The model was saved by XGBoost {writer_version[0]}.{writer_version[1]}, "
            f"but XGBoost {installed_version} is installed. "
            f"Install XGBoost {writer_version[0]}.x at {writer_version[0]}.{writer_version[1]} or newer."
        )


@cache
def load_name_idf_sketch(jar_name: str) -> tuple[CountMinSketch, int]:
    """The name token sketch and the number of documents it counted."""
    table = pq.read_table(
        io.BytesIO(_read_jar_resource(NAME_SKETCH_RESOURCE, jar_name))
    )
    row = table.to_pylist()[0]
    return CountMinSketch(row["cms_bytes"]), row["num_docs"]


def compute_name_idf(
    sketch: CountMinSketch, num_documents: int, tokens: list[str] | None
) -> dict[str, float]:
    """Inverse document frequency of each distinct token. Null gives an empty map."""
    return {
        token: math.log(num_documents / max(1, sketch.estimate_count(token))) + 1.0
        for token in tokens or ()
        if token is not None
    }


def bind_name_idf_udf(jar_name: str) -> Callable[[Column], Column]:
    """Bind ``compute_name_idf`` over an array-of-string column with the shipped jar's sketch."""

    def compute_idf_map(tokens: list[str] | None) -> dict[str, float]:
        return compute_name_idf(*load_name_idf_sketch(jar_name), tokens)

    return F.udf(compute_idf_map, MapType(StringType(), DoubleType(), False))


def predict_match_probability(
    booster, feature_columns: Sequence[pd.Series]
) -> pd.Series:
    """Probability of the positive class for each row. NaN means missing."""
    import numpy as np

    matrix = np.column_stack(feature_columns).astype(np.float32)
    return pd.Series(booster.inplace_predict(matrix).astype(np.float64))


def bind_match_probability_udf(
    jar_name: str, resource_path: str
) -> Callable[[list[Column]], Column]:
    """Bind ``predict_match_probability`` over feature columns in booster order.

    Args:
        jar_name: File name of the shipped model jar.
        resource_path: Booster path inside the jar.
    """

    def score_batch(*feature_columns: pd.Series) -> pd.Series:
        return predict_match_probability(
            load_xgboost_booster(jar_name, resource_path), feature_columns
        )

    udf = F.pandas_udf(score_batch, DoubleType())
    return lambda feature_inputs: udf(*feature_inputs)
