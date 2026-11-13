# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""The places model: settings read from the model jar, and the loaders workers use.

The driver reads the blocking settings once into ``PlacesModel``. Workers
read the booster and the name sketch out of the jar that ``sc.addFile``
shipped to them, once per process.
"""

from __future__ import annotations

import io
import json
import math
import os
import zipfile
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
from pyspark import SparkFiles
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, MapType, StringType

from overture_matchers.places import features
from overture_matchers.places.model_jar import open_jar
from overture_matchers.sketches import CountMinSketch

if TYPE_CHECKING:
    import pandas as pd
    from pyspark.sql import Column, SparkSession

BLOCKING_CONFIG_RESOURCE = "places_matcher_model/metadata/blocking_config"
BLOCKING_BOOSTER_RESOURCE = "places_matcher_model/data/blocking_model"
NAME_SKETCH_RESOURCE = "config/places/cms-names.parquet"


@dataclass(frozen=True)
class PlacesModel:
    """Settings of the model jar.

    Attributes:
        jar_name: File name of the jar, which is where workers find it.
        blocking_features: Blocking features in booster order.
        blocking_threshold: Smallest blocking score that keeps a pair.
        blocking_radius_m: Largest distance between the places of a pair.
    """

    jar_name: str
    blocking_features: tuple[str, ...]
    blocking_threshold: float
    blocking_radius_m: float

    @classmethod
    def load(cls, spark: SparkSession, model_jar: str) -> PlacesModel:
        """Read the settings from ``model_jar``.

        Raises:
            ValueError: The blocking feature order names a feature that
                ``features.py`` does not produce.
        """
        with open_jar(spark, model_jar) as jar:
            config = json.loads(jar.read(BLOCKING_CONFIG_RESOURCE))
        order = tuple(config["feature_order"])
        unknown = [
            f
            for f in order
            if f not in features.BLOCKING_FEATURES and f != features.GEO_DISTANCE
        ]
        if unknown:
            raise ValueError(
                f"blocking feature_order needs {unknown}, but features.py produces "
                f"only {sorted([*features.BLOCKING_FEATURES, features.GEO_DISTANCE])}"
            )
        return cls(
            jar_name=os.path.basename(model_jar.rstrip("/")),
            blocking_features=order,
            blocking_threshold=float(config["prediction_threshold"]),
            blocking_radius_m=float(config["spatial_radius_m"]),
        )


def _read_resource(resource: str, jar_name: str) -> bytes:
    with zipfile.ZipFile(SparkFiles.get(jar_name)) as jar:
        return jar.read(resource)


@cache
def load_booster(jar_name: str, resource: str):
    """The XGBoost booster stored at ``resource`` in the jar. It predicts on one thread."""
    import xgboost

    model = _read_resource(resource, jar_name)
    check_xgboost_version(model, xgboost.__version__)
    booster = xgboost.Booster({"nthread": 1})
    booster.load_model(bytearray(model))
    return booster


def check_xgboost_version(model: bytes, installed: str) -> None:
    """Raise unless ``installed`` XGBoost reads the JSON booster ``model`` as written.

    XGBoost reads models from older versions only. An older reader can misread
    newer fields and silently shift every score.
    """
    writer = tuple(json.loads(model)["version"][:2])
    reader = tuple(int(part) for part in installed.split(".")[:2])
    if reader < writer or reader[0] != writer[0]:
        raise RuntimeError(
            f"The model was saved by XGBoost {writer[0]}.{writer[1]}, "
            f"but XGBoost {installed} is installed. "
            f"Install XGBoost {writer[0]}.x at {writer[0]}.{writer[1]} or newer."
        )


@cache
def load_name_idf(jar_name: str) -> tuple[CountMinSketch, int]:
    """The name token sketch and the number of documents it counted."""
    table = pq.read_table(io.BytesIO(_read_resource(NAME_SKETCH_RESOURCE, jar_name)))
    row = table.to_pylist()[0]
    return CountMinSketch(row["cms_bytes"]), row["num_docs"]


def name_idf_udf(jar_name: str, tokens: Column) -> Column:
    """Map of each distinct name token to its inverse document frequency.

    Args:
        jar_name: File name of the shipped model jar.
        tokens: Array-of-string column. Null gives an empty map.
    """

    def idf_map(tokens: list[str] | None) -> dict[str, float]:
        sketch, num_docs = load_name_idf(jar_name)
        return {
            t: math.log(num_docs / max(1, sketch.estimate_count(t))) + 1.0
            for t in tokens or ()
            if t is not None
        }

    return F.udf(idf_map, MapType(StringType(), DoubleType(), False))(tokens)


def predict_udf(jar_name: str, resource: str, inputs: list[Column]) -> Column:
    """Probability of the positive class, from the booster at ``resource``.

    Args:
        jar_name: File name of the shipped model jar.
        resource: Booster path inside the jar.
        inputs: Feature columns in booster order. Null becomes missing.
    """
    import numpy as np
    import pandas as pd

    def predict(*columns: pd.Series) -> pd.Series:
        matrix = np.column_stack(columns).astype(np.float32)
        booster = load_booster(jar_name, resource)
        return pd.Series(booster.inplace_predict(matrix).astype(np.float64))

    udf = F.pandas_udf(predict, DoubleType(), F.PandasUDFType.SCALAR)
    return udf(*inputs)
