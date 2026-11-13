# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""The places model: settings read from the model jar, and the engines workers run.

The driver reads the blocking settings once into ``PlacesModel`` and the
scoring config with ``read_scoring_config``. The engine functions return
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
from overture_matchers.places.model_jar import CONFIG_RESOURCE, open_model_jar
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


def read_scoring_config(spark: SparkSession, model_jar: str) -> dict:
    """The jar's parsed scoring config."""
    with open_model_jar(spark, model_jar) as jar:
        return json.loads(jar.read(CONFIG_RESOURCE))


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
def load_sketch(jar_name: str, resource_path: str) -> tuple[CountMinSketch, int]:
    """The token sketch at ``resource_path`` and the number of documents it counted."""
    table = pq.read_table(io.BytesIO(_read_jar_resource(resource_path, jar_name)))
    row = table.to_pylist()[0]
    return CountMinSketch(row["cms_bytes"]), row["num_docs"]


def load_name_idf_sketch(jar_name: str) -> tuple[CountMinSketch, int]:
    """The name token sketch and the number of documents it counted."""
    return load_sketch(jar_name, NAME_SKETCH_RESOURCE)


def token_idf(sketch: CountMinSketch, num_documents: int, token: str) -> float:
    """Inverse document frequency of ``token``."""
    return math.log(num_documents / max(1, sketch.estimate_count(token))) + 1.0


def compute_name_idf(
    sketch: CountMinSketch, num_documents: int, tokens: list[str] | None
) -> dict[str, float]:
    """Inverse document frequency of each distinct token. Null gives an empty map."""
    return {
        token: token_idf(sketch, num_documents, token)
        for token in tokens or ()
        if token is not None
    }


@cache
def scoring_lookups(jar_name: str) -> features.PairwiseLookups:
    """The token weights and name counts the pairwise features read from the jar."""

    def idf(resource_path: str):
        return partial(token_idf, *load_sketch(jar_name, resource_path))

    def bloom(resource_path: str):
        return BloomFilter(
            _read_jar_resource(resource_path, jar_name)
        ).might_contain_string

    return features.PairwiseLookups(
        name_idf=idf(NAME_SKETCH_RESOURCE),
        address_idf=idf(ADDRESS_SKETCH_RESOURCE),
        name_repeated=bloom(NAME_H3_GE2_RESOURCE),
        name_common=bloom(NAME_H3_GE10_RESOURCE),
    )


def bind_name_idf_udf(jar_name: str) -> Callable[[Column], Column]:
    """Bind ``compute_name_idf`` over an array-of-string column with the shipped jar's sketch."""

    def compute_idf_map(tokens: list[str] | None) -> dict[str, float]:
        return compute_name_idf(*load_name_idf_sketch(jar_name), tokens)

    return F.udf(compute_idf_map, MapType(StringType(), DoubleType(), False))


def _matrix(feature_columns: Sequence[pd.Series]):
    import numpy as np

    return np.column_stack(feature_columns).astype(np.float32)


def predict_match_probability(
    booster, feature_columns: Sequence[pd.Series]
) -> pd.Series:
    """Probability of the positive class for each row. NaN means missing."""
    import numpy as np

    return pd.Series(
        booster.inplace_predict(_matrix(feature_columns)).astype(np.float64)
    )


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
        return margin_and_probability(load_xgboost_booster(jar_name, resource), columns)

    schema = StructType(
        [StructField(MARGIN, DoubleType()), StructField(PROBABILITY, DoubleType())]
    )
    udf = F.pandas_udf(score, schema, F.PandasUDFType.SCALAR)
    return lambda inputs: udf(*inputs)
