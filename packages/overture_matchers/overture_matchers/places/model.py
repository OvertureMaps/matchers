# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""The places model: settings read from the model jar, and the engines workers run.

The driver reads the blocking settings once into ``PlacesModel``. The engine functions return
``Column`` callables or lookups that read the boosters, sketches and Bloom
filters out of the jar ``sc.addFile`` shipped to workers, once per process.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import dataclass
from functools import cache, partial
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow.parquet as pq
from pyspark import SparkFiles
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    MapType,
    StringType,
    StructField,
    StructType,
)

from overture_matchers.places import features
from overture_matchers.places.model_jar import open_jar
from overture_matchers.sketches import BloomFilter, CountMinSketch

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pyspark.sql import Column, SparkSession

BLOCKING_CONFIG_RESOURCE = "places_matcher_model/metadata/blocking_config"
BLOCKING_BOOSTER_RESOURCE = "places_matcher_model/data/blocking_model"
SCORING_BOOSTER_RESOURCE = "places_matcher_model/data/model"
NAME_SKETCH_RESOURCE = "config/places/cms-names.parquet"
ADDRESS_SKETCH_RESOURCE = "config/places/cms-address.parquet"
# Keys of names repeated at least 2 and 10 times in one H3 cell.
NAME_H3_GE2_RESOURCE = "config/places/unique_name_h3_bin_ge_2.bin"
NAME_H3_GE10_RESOURCE = "config/places/unique_name_h3_bin_ge_10.bin"


@dataclass(frozen=True)
class PlacesModel:
    """Settings of the model jar.

    Attributes:
        blocking_features: Blocking features in booster order.
        blocking_threshold: Smallest blocking score that keeps a pair.
        blocking_radius_m: Largest distance between the places of a pair.
    """

    blocking_features: tuple[str, ...]
    blocking_threshold: float
    blocking_radius_m: float

    @classmethod
    def load(cls, spark: SparkSession, model_jar: str) -> PlacesModel:
        """Read the settings from ``model_jar`` and check its booster's XGBoost version."""
        with open_jar(spark, model_jar) as jar:
            return cls.from_jar(jar)

    @classmethod
    def from_jar(cls, jar: zipfile.ZipFile) -> PlacesModel:
        """Read the settings from an open jar and check its booster's XGBoost version."""
        import xgboost

        check_xgboost_version(jar.read(BLOCKING_BOOSTER_RESOURCE), xgboost.__version__)
        return cls.from_config(json.loads(jar.read(BLOCKING_CONFIG_RESOURCE)))

    @classmethod
    def from_config(cls, config: dict) -> PlacesModel:
        """Settings from the jar's parsed blocking config.

        Raises:
            ValueError: The blocking feature order names a feature that
                ``features.py`` does not produce.
        """
        order = tuple(config["feature_order"])
        unknown = [f for f in order if f not in features.BLOCKING_FEATURES]
        if unknown:
            raise ValueError(
                f"blocking feature_order needs {unknown}, but features.py produces "
                f"only {sorted(features.BLOCKING_FEATURES)}"
            )
        return cls(
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
    booster = xgboost.Booster({"nthread": 1})
    booster.load_model(bytearray(model))
    return booster


def check_xgboost_version(model: bytes, installed: str) -> None:
    """Raise unless ``installed`` XGBoost reads the JSON booster ``model`` as written.

    XGBoost reads models from older versions only. An older reader can misread
    newer fields and silently shift every score.
    """
    try:
        writer = tuple(json.loads(model)["version"][:2])
    except ValueError as e:
        raise RuntimeError("The booster must be saved as XGBoost JSON.") from e
    reader = tuple(int(part) for part in installed.split(".")[:2])
    if reader < writer or reader[0] != writer[0]:
        raise RuntimeError(
            f"The model was saved by XGBoost {writer[0]}.{writer[1]}, "
            f"but XGBoost {installed} is installed. "
            f"Install XGBoost {writer[0]}.x at {writer[0]}.{writer[1]} or newer."
        )


@cache
def load_sketch(jar_name: str, resource: str) -> tuple[CountMinSketch, int]:
    """The token sketch at ``resource`` and the number of documents it counted."""
    table = pq.read_table(io.BytesIO(_read_resource(resource, jar_name)))
    row = table.to_pylist()[0]
    return CountMinSketch(row["cms_bytes"]), row["num_docs"]


def load_name_idf(jar_name: str) -> tuple[CountMinSketch, int]:
    """The name token sketch and the number of documents it counted."""
    return load_sketch(jar_name, NAME_SKETCH_RESOURCE)


def token_idf(sketch: CountMinSketch, num_docs: int, token: str) -> float:
    """Inverse document frequency of ``token``."""
    return math.log(num_docs / max(1, sketch.estimate_count(token))) + 1.0


def name_idf(
    sketch: CountMinSketch, num_docs: int, tokens: list[str] | None
) -> dict[str, float]:
    """Inverse document frequency of each distinct token. Null gives an empty map."""
    return {t: token_idf(sketch, num_docs, t) for t in tokens or () if t is not None}


@cache
def scoring_lookups(jar_name: str) -> features.PairwiseLookups:
    """The token weights and name counts the pairwise features read from the jar."""

    def idf(resource: str):
        return partial(token_idf, *load_sketch(jar_name, resource))

    def bloom(resource: str):
        return BloomFilter(_read_resource(resource, jar_name)).might_contain_string

    return features.PairwiseLookups(
        name_idf=idf(NAME_SKETCH_RESOURCE),
        address_idf=idf(ADDRESS_SKETCH_RESOURCE),
        name_repeated=bloom(NAME_H3_GE2_RESOURCE),
        name_common=bloom(NAME_H3_GE10_RESOURCE),
    )


def name_idf_udf(jar_name: str) -> Callable[[Column], Column]:
    """``name_idf`` over an array-of-string column, with the shipped jar's sketch."""

    def idf_map(tokens: list[str] | None) -> dict[str, float]:
        return name_idf(*load_name_idf(jar_name), tokens)

    return F.udf(idf_map, MapType(StringType(), DoubleType(), False))


def _matrix(columns: Sequence[pd.Series]):
    import numpy as np

    return np.column_stack(columns).astype(np.float32)


def predict(booster, columns: Sequence[pd.Series]) -> pd.Series:
    """Probability of the positive class for each row. NaN means missing."""
    import numpy as np

    return pd.Series(booster.inplace_predict(_matrix(columns)).astype(np.float64))


def predict_udf(jar_name: str, resource: str) -> Callable[[list[Column]], Column]:
    """``predict`` over feature columns in booster order, with the shipped booster.

    Args:
        jar_name: File name of the shipped model jar.
        resource: Booster path inside the jar.
    """

    def score(*columns: pd.Series) -> pd.Series:
        return predict(load_booster(jar_name, resource), columns)

    udf = F.pandas_udf(score, DoubleType())
    return lambda inputs: udf(*inputs)


MARGIN = "margin"
PROBABILITY = "probability"


def margin_and_probability(booster, columns: Sequence[pd.Series]) -> pd.DataFrame:
    """Raw margin and positive-class probability for each row, widened to double."""
    import numpy as np
    import pandas as pd

    matrix = _matrix(columns)
    return pd.DataFrame(
        {
            MARGIN: booster.inplace_predict(matrix, predict_type="margin"),
            PROBABILITY: booster.inplace_predict(matrix),
        }
    ).astype(np.float64)


def margin_and_probability_udf(
    jar_name: str, resource: str
) -> Callable[[list[Column]], Column]:
    """Struct of ``margin_and_probability`` over feature columns, with the shipped booster.

    Args:
        jar_name: File name of the shipped model jar.
        resource: Booster path inside the jar.
    """

    def score(*columns: pd.Series) -> pd.DataFrame:
        return margin_and_probability(load_booster(jar_name, resource), columns)

    schema = StructType(
        [StructField(MARGIN, DoubleType()), StructField(PROBABILITY, DoubleType())]
    )
    udf = F.pandas_udf(score, schema, F.PandasUDFType.SCALAR)
    return lambda inputs: udf(*inputs)
