# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Tests for history pairs and place blocking on a few rows and a tiny model.

Blocking rules run with stand-in engines. The jar fixture checks the real
engines and the stage that binds them.
"""

import io
import json
import math
import os
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xgboost
from pyspark.sql import Row
from pyspark.sql import functions as F

from overture_matchers.core import io as stage_io
from overture_matchers.core.stages import SetupOutput
from overture_matchers.places import features, job
from overture_matchers.places.blocking import PAIR_COLUMNS, blocking_pairs
from overture_matchers.places.model import (
    BLOCKING_BOOSTER_RESOURCE,
    PlacesModel,
    check_xgboost_version,
    load_booster,
    load_name_idf,
    name_idf,
    predict,
)
from overture_matchers.prematch import history_pairs
from overture_matchers.sketches import CountMinSketch

FEATURES = ["name_tfidf", "name_similarity", "geo_distance"]
DOCS = 100
SKETCH = {"cafe": 100, "rare": 1}
CONFIG = {
    "feature_order": FEATURES,
    "prediction_threshold": 0.5,
    "spatial_radius_m": 400.0,
}


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
            json.dumps(CONFIG),
        )
        z.writestr(
            "places_matcher_model/data/blocking_model",
            bytes(booster.save_raw(raw_format="json")),
        )
        z.writestr("config/places/cms-names.parquet", out.getvalue())
    return str(path)


@pytest.fixture(scope="module")
def shipped(spark, jar):
    """File name of the jar, shipped to the workers."""
    spark.sparkContext.addFile(jar)
    return os.path.basename(jar)


def unit_idf(tokens):
    return F.map_from_entries(
        F.transform(F.array_distinct(tokens), lambda t: F.struct(t, F.lit(1.0)))
    )


def name_similarity_score(columns):
    return columns[FEATURES.index("name_similarity")]


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
    with pytest.raises(ValueError, match="version 2"):
        CountMinSketch(b"\x00\x00\x00\x02" + bytes(20))


def test_model_rejects_an_unknown_blocking_feature(monkeypatch):
    monkeypatch.delitem(features.BLOCKING_FEATURES, "name_similarity")
    with pytest.raises(ValueError, match="name_similarity"):
        PlacesModel.from_config(CONFIG)


def test_model_reads_the_jar_config(spark, jar):
    assert PlacesModel.load(spark, jar) == PlacesModel(tuple(FEATURES), 0.5, 400.0)


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
    with pytest.raises(RuntimeError, match="JSON"):
        check_xgboost_version(b"binf", xgboost.__version__)


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


def test_blocking_keeps_close_similar_pairs_in_id_order(spark):
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
    out = blocking_pairs(
        candidates,
        baseline,
        PlacesModel.from_config(CONFIG),
        unit_idf,
        name_similarity_score,
    )
    plan = out._jdf.queryExecution().executedPlan().toString()
    assert "DistanceJoin" in plan
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


def test_matching_writes_blocking_with_the_jar_engines(spark, jar, tmp_path):
    candidates = frame(
        spark,
        [
            record("b", "blue cafe", 0.0, 0.0),
            record("c", "blue cafe", 0.0, 0.001),
            record("h", "blue cafe", 0.0, 0.0002),
        ],
    ).withColumn(
        "base_ids",
        F.when(F.col("id") == "h", F.array(F.lit("a"))).cast("array<string>"),
    )
    baseline = frame(spark, [record("a", "red bar", 0.0, 0.0005)])
    with pytest.raises(NotImplementedError, match="scoring"):
        job.matching(SetupOutput(candidates, baseline), jar, str(tmp_path))
    out = stage_io.read_output(spark, str(tmp_path), "blocking", job.FEATURE_TYPE)
    assert {(r.id, r.base_id) for r in out.collect()} == {("b", "c")}


def test_name_idf_weighs_rare_tokens_more(shipped):
    sketch, num_docs = load_name_idf(shipped)
    got = name_idf(sketch, num_docs, ["cafe", "rare", "cafe", None])
    assert got == pytest.approx(
        {"cafe": math.log(DOCS / 100) + 1, "rare": math.log(DOCS / 1) + 1}
    )
    assert name_idf(sketch, num_docs, None) == {}


def test_predict_matches_the_booster(shipped):
    booster = load_booster(shipped, BLOCKING_BOOSTER_RESOURCE)
    columns = [
        pd.Series([0.5, 0.0]),
        pd.Series([1.0, float("nan")]),
        pd.Series([10.0, 300.0]),
    ]
    expected = booster.inplace_predict(np.column_stack(columns).astype(np.float32))
    assert predict(booster, columns).tolist() == pytest.approx(expected.tolist())


@pytest.mark.parametrize("name", sorted(features.BLOCKING_FEATURES))
def test_features_read_only_their_declared_inputs(spark, name):
    inputs = features.FEATURE_INPUTS[name]
    types = {features.NAME_IDF: "map<string,double>"}
    schema = ", ".join(
        [
            f"{features.GEO_DISTANCE} double",
            *(
                f"{side}{c} {types.get(c, 'string')}"
                for c in inputs
                for side in ("", "base_")
            ),
        ]
    )
    df = spark.createDataFrame([], schema)
    df.select(features.BLOCKING_FEATURES[name]()).collect()


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
