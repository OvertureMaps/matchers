# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Tests for history pairs and place blocking on a few rows and a tiny model."""

import io
import json
import math
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xgboost
from pyspark.sql import Row

from overture_matchers.places import features
from overture_matchers.places.blocking import PAIR_COLUMNS, blocking_pairs
from overture_matchers.places.model import PlacesModel, check_xgboost_version
from overture_matchers.prematch import history_pairs
from overture_matchers.sketches import CountMinSketch

FEATURES = ["name_tfidf", "name_similarity", "geo_distance"]
DOCS = 100
SKETCH = {"cafe": 100, "rare": 1}


@pytest.fixture(scope="module")
def jar(spark, tmp_path_factory):
    """A jar holding a blocking booster that keeps pairs with similar names."""
    rng = np.random.default_rng(0)
    x = rng.random((400, 3)) * [2.0, 1.0, 400.0]
    y = (x[:, 1] > 0.5).astype(int)
    booster = xgboost.train(
        {"objective": "binary:logistic", "max_depth": 2},
        xgboost.DMatrix(x, label=y),
        num_boost_round=10,
    )
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
            json.dumps(
                {
                    "feature_order": FEATURES,
                    "prediction_threshold": 0.5,
                    "spatial_radius_m": 400.0,
                }
            ),
        )
        z.writestr(
            "places_matcher_model/data/blocking_model",
            bytes(booster.save_raw(raw_format="json")),
        )
        z.writestr("config/places/cms-names.parquet", out.getvalue())
    return str(path)


def record(id, name, lon=0.0, lat=0.0, **extra):
    tokens = name.split() if name else None
    return Row(
        id=id,
        provider="p",
        longitude=lon,
        latitude=lat,
        house_number=None,
        address=None,
        address_clean=None,
        address_clean_tokens=None,
        country=None,
        normalized_phone=None,
        name=name,
        name_clean=name,
        name_clean_tokens=tokens,
        name_clean_stripped=name,
        name_clean_stripped_tokens=tokens,
        taxonomy_primary=None,
        website_domains=None,
        website_urls=None,
        **extra,
    )


def frame(spark, rows):
    return spark.createDataFrame(
        rows,
        "id string, provider string, longitude double, latitude double,"
        "house_number string, address string, address_clean string,"
        "address_clean_tokens array<string>, country string,"
        "normalized_phone string, name string, name_clean string,"
        "name_clean_tokens array<string>, name_clean_stripped string,"
        "name_clean_stripped_tokens array<string>, taxonomy_primary string,"
        "website_domains array<string>, website_urls array<string>",
    )


def test_count_min_sketch_matches_the_jvm_estimate(jar):
    with zipfile.ZipFile(jar) as z:
        table = pq.read_table(io.BytesIO(z.read("config/places/cms-names.parquet")))
    sketch = CountMinSketch(table.to_pylist()[0]["cms_bytes"])
    assert [sketch.estimate_count(w) for w in ("cafe", "rare", "absent")] == [
        100,
        1,
        0,
    ]


def test_model_rejects_an_unknown_blocking_feature(spark, jar, monkeypatch):
    monkeypatch.delitem(features.BLOCKING_FEATURES, "name_similarity")
    with pytest.raises(ValueError, match="name_similarity"):
        PlacesModel.load(spark, jar)


def test_booster_from_a_newer_xgboost_is_rejected():
    model = json.loads(
        xgboost.train({}, xgboost.DMatrix([[0.0]], label=[0])).save_raw("json")
    )
    major, minor = (int(p) for p in xgboost.__version__.split(".")[:2])
    model["version"] = [major, minor, 0]
    check_xgboost_version(json.dumps(model).encode(), xgboost.__version__)
    for newer in ([major, minor + 1, 0], [major + 1, 0, 0]):
        model["version"] = newer
        with pytest.raises(RuntimeError, match="saved by XGBoost"):
            check_xgboost_version(json.dumps(model).encode(), xgboost.__version__)


def test_history_pairs_explode_base_ids(spark):
    candidates = spark.createDataFrame(
        [("a", "p", ["x", "y"]), ("b", "p", None)],
        "id string, provider string, base_ids array<string>",
    )
    rows = history_pairs(candidates).collect()
    assert [(r.id, r.base_id, r.composite_score, r.prediction) for r in rows] == [
        ("a", "x", 1.0, 1),
        ("a", "y", 1.0, 1),
    ]
    assert {(r.record_type, r.base_record_type, r.base_provider) for r in rows} == {
        ("candidate", "baseline", "p")
    }


def test_blocking_keeps_close_similar_pairs_in_id_order(spark, jar):
    model = PlacesModel.load(spark, jar)
    candidates = frame(
        spark,
        [
            record("b", "blue cafe", 0.0, 0.0),
            record("c", "blue cafe", 0.0, 0.001),  # 111 m away
            record("d", "blue cafe", 0.0, 1.0),  # too far
        ],
    )
    baseline = frame(
        spark,
        [
            record("a", "blue cafe", 0.0, 0.0005),
            record("e", "zzzzzz", 0.0, 0.0),  # close, but a different name
        ],
    )
    out = blocking_pairs(candidates, baseline, model)
    assert out.columns[:3] == ["base_id", "base_provider", "base_record_type"]
    assert out.columns[-len(PAIR_COLUMNS) :] == list(PAIR_COLUMNS)
    pairs = {
        (r.id, r.base_id): (r.record_type, r.base_record_type) for r in out.collect()
    }
    assert pairs == {
        ("a", "b"): ("baseline", "candidate"),
        ("a", "c"): ("baseline", "candidate"),
        ("b", "c"): ("candidate", "candidate"),
    }


def test_name_tfidf_weighs_rare_tokens_more(spark, jar):
    from overture_matchers.places.model import name_idf_udf

    df = spark.createDataFrame(
        [(["cafe", "rare"],), (["cafe"],), (None,)], "tokens array<string>"
    )
    model = PlacesModel.load(spark, jar)
    got = [
        r.m
        for r in df.select(name_idf_udf(model.jar_name, df.tokens).alias("m")).collect()
    ]
    assert got[0]["rare"] == pytest.approx(math.log(DOCS / 1) + 1)
    assert got[1]["cafe"] == pytest.approx(math.log(DOCS / 100) + 1)
    assert got[2] == {}


def test_pair_features(spark):
    cols = [
        "name_clean",
        "base_name_clean",
        "normalized_phone",
        "base_normalized_phone",
        "country",
        "base_country",
        "house_number",
        "base_house_number",
    ]
    rows = [
        ("abcd", "abce", "123456789", "000456789", "US", "US", "1", "1"),
        (None, "", "123456789", "000456789", None, "US", "1", "2"),
        ("x", "x", "", "5", "US", "US", None, "2"),
    ]
    df = spark.createDataFrame(rows, ", ".join(f"{c} string" for c in cols))
    got = df.select(
        features.name_similarity().alias("n"),
        features.phone_number_similarity_availability().alias("p"),
        features.house_number_similarity_availability().alias("h"),
    ).collect()
    assert [tuple(r) for r in got] == [
        (0.75, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, -1.0, -1.0),
    ]
