# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Shared pytest fixtures: one local Sedona session and a tiny places model jar."""

import io
import json
import os
import sys
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xgboost
from overture_spark import SparkPlatform, getSparkSedonaSession

from overture_matchers.places.job import PACKAGES

# Optional private inputs. Tests that need them skip when they are unset.
MODEL_JAR = os.environ.get("PLACES_MATCHER_MODEL_JAR")
LIBPOSTAL_DATA = os.environ.get("LIBPOSTAL_DATA_DIR")


def pytest_collection_modifyitems(items):
    # Set REQUIRE_PRIVATE_INPUTS where the inputs exist, so a missing one fails.
    if os.environ.get("REQUIRE_PRIVATE_INPUTS"):
        return
    skips = {
        "normalizer": (MODEL_JAR, "set PLACES_MATCHER_MODEL_JAR"),
        "libpostal": (LIBPOSTAL_DATA, "set LIBPOSTAL_DATA_DIR"),
    }
    for item in items:
        for marker, (value, reason) in skips.items():
            if marker in item.keywords and not value:
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="session")
def model_jar():
    return MODEL_JAR


@pytest.fixture(scope="session")
def spark():
    # UDF workers must run under this venv's interpreter, which is
    # sys.executable here.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    session = getSparkSedonaSession(
        spark_platform=SparkPlatform.LOCAL,
        extra_packages=list(PACKAGES),
        extra_spark_conf={
            "spark.master": "local[1]",
            "spark.app.name": "overture-matchers-tests",
            # The driver binds to loopback, because the hostname often fails to
            # resolve on laptops and VPNs.
            "spark.driver.bindAddress": "127.0.0.1",
            "spark.driver.host": "127.0.0.1",
            "spark.sql.shuffle.partitions": "1",
            "spark.ui.enabled": "false",
            # UTC keeps timestamp results independent of the machine's timezone.
            "spark.sql.session.timeZone": "UTC",
            **({"spark.jars": MODEL_JAR} if MODEL_JAR else {}),
            **(
                {"spark.sedona.libpostal.dataDir": LIBPOSTAL_DATA}
                if LIBPOSTAL_DATA
                else {}
            ),
        },
    )
    yield session
    session.stop()


FEATURES = ["name_tfidf", "name_similarity", "geo_distance"]
DOCS = 100
SKETCH = {"cafe": 100, "rare": 1}
CONFIG = {
    "feature_order": FEATURES,
    "prediction_threshold": 0.5,
    "spatial_radius_m": 400.0,
}


def train_booster(x, y):
    """A tiny binary booster saved as JSON."""
    booster = xgboost.train(
        {"objective": "binary:logistic", "max_depth": 2},
        xgboost.DMatrix(x, label=y),
        num_boost_round=10,
    )
    return bytes(booster.save_raw(raw_format="json"))


@pytest.fixture(scope="session")
def jar(spark, tmp_path_factory):
    """A jar holding a blocking booster that keeps pairs with similar names."""
    rng = np.random.default_rng(0)
    x = rng.random((400, 3)) * [2.0, 1.0, 400.0]
    booster = train_booster(x, (x[:, 1] > 0.5).astype(int))
    words = spark.createDataFrame(
        [(w,) for w, n in SKETCH.items() for _ in range(n)], ["word"]
    )
    sketch = words._jdf.stat().countMinSketch("word", 5, 1000, 1)
    cms = pa.table({"cms_bytes": [bytes(sketch.toByteArray())], "num_docs": [DOCS]})
    out = io.BytesIO()
    pq.write_table(cms, out)
    path = tmp_path_factory.mktemp("jar") / "tiny-model.jar"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(
            "places_matcher_model/metadata/blocking_config",
            json.dumps(CONFIG),
        )
        z.writestr("places_matcher_model/data/blocking_model", booster)
        z.writestr("config/places/cms-names.parquet", out.getvalue())
    return str(path)
