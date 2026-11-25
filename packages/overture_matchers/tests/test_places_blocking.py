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
from overture_matchers.places.blocking import PAIR_COLUMNS, get_blocking_pairs
from overture_matchers.places.model import (
    BLOCKING_BOOSTER_RESOURCE,
    PlacesModel,
    check_xgboost_version,
    compute_name_idf,
    load_name_idf_sketch,
    load_xgboost_booster,
    predict_match_probability,
)
from overture_matchers.prematch import generate_history_pairs
from overture_matchers.sketches import CountMinSketch

FEATURES = ["name_tfidf", "name_similarity", "geo_distance"]
NUM_DOCUMENTS = 100
SKETCH = {"cafe": 100, "rare": 1}
CONFIG = {
    "feature_order": FEATURES,
    "prediction_threshold": 0.5,
    "spatial_radius_m": 400.0,
}


@pytest.fixture(scope="module")
def model_jar(spark, tmp_path_factory):
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
        [(word,) for word, count in SKETCH.items() for _ in range(count)], ["word"]
    )
    sketch = words._jdf.stat().countMinSketch("word", 5, 1000, 1)
    cms = pa.table(
        {"cms_bytes": [bytes(sketch.toByteArray())], "num_docs": [NUM_DOCUMENTS]}
    )
    out = io.BytesIO()
    pq.write_table(cms, out)
    path = tmp_path_factory.mktemp("jar") / "tiny-model.jar"
    with zipfile.ZipFile(path, "w") as jar_file:
        jar_file.writestr(
            "places_matcher_model/metadata/blocking_config",
            json.dumps(CONFIG),
        )
        jar_file.writestr(
            "places_matcher_model/data/blocking_model",
            bytes(booster.save_raw(raw_format="json")),
        )
        jar_file.writestr("config/places/cms-names.parquet", out.getvalue())
    return str(path)


@pytest.fixture(scope="module")
def shipped_jar_name(spark, model_jar):
    """File name of the jar, shipped to the workers."""
    spark.sparkContext.addFile(model_jar)
    return os.path.basename(model_jar)


def compute_unit_idf(tokens):
    return F.map_from_entries(
        F.transform(F.array_distinct(tokens), lambda token: F.struct(token, F.lit(1.0)))
    )


def select_name_similarity_score(feature_columns):
    return feature_columns[FEATURES.index("name_similarity")]


def build_place_record(record_id, name, longitude=0.0, latitude=0.0, **extra):
    tokens = name.split() if name else None
    return Row(
        id=record_id,
        provider="p",
        longitude=longitude,
        latitude=latitude,
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


def build_places_dataframe(spark, rows):
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


def test_count_min_sketch_matches_the_jvm_estimate(model_jar):
    with zipfile.ZipFile(model_jar) as jar_file:
        table = pq.read_table(
            io.BytesIO(jar_file.read("config/places/cms-names.parquet"))
        )
    sketch = CountMinSketch(table.to_pylist()[0]["cms_bytes"])
    assert [sketch.estimate_count(word) for word in ("cafe", "rare", "absent")] == [
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


def test_model_reads_the_jar_config(spark, model_jar):
    assert PlacesModel.load(spark, model_jar) == PlacesModel(
        tuple(FEATURES), 0.5, 400.0
    )


def test_booster_from_a_newer_xgboost_is_rejected():
    model = json.loads(
        xgboost.train({}, xgboost.DMatrix([[0.0]], label=[0])).save_raw("json")
    )
    major, minor = (int(part) for part in xgboost.__version__.split(".")[:2])
    model["version"] = [major, minor, 0]
    check_xgboost_version(json.dumps(model).encode(), xgboost.__version__)
    for newer in ([major, minor + 1, 0], [major + 1, 0, 0]):
        model["version"] = newer
        with pytest.raises(RuntimeError, match="saved by XGBoost"):
            check_xgboost_version(json.dumps(model).encode(), xgboost.__version__)
    with pytest.raises(RuntimeError, match="JSON"):
        check_xgboost_version(b"binf", xgboost.__version__)


def test_generate_history_pairs_explodes_base_ids(spark):
    candidates = spark.createDataFrame(
        [("a", "p", ["x", "y"]), ("b", "p", None)],
        "id string, provider string, base_ids array<string>",
    )
    rows = generate_history_pairs(candidates).collect()
    assert [
        (row.id, row.base_id, row.composite_score, row.prediction) for row in rows
    ] == [
        ("a", "x", 1.0, 1),
        ("a", "y", 1.0, 1),
    ]
    assert {
        (row.record_type, row.base_record_type, row.base_provider) for row in rows
    } == {("candidate", "baseline", "p")}


def test_get_blocking_pairs_keeps_close_similar_pairs_in_id_order(spark):
    candidates = build_places_dataframe(
        spark,
        [
            build_place_record("b", "blue cafe", 0.0, 0.0),
            build_place_record("c", "blue cafe", 0.0, 0.001),  # 111 m away
            build_place_record("d", "blue cafe", 0.0, 1.0),  # too far
        ],
    )
    baseline = build_places_dataframe(
        spark,
        [
            build_place_record("a", "blue cafe", 0.0, 0.0005),
            build_place_record("e", "zzzzzz", 0.0, 0.0),  # close, but a different name
        ],
    )
    out = get_blocking_pairs(
        candidates,
        baseline,
        PlacesModel.from_config(CONFIG),
        compute_unit_idf,
        select_name_similarity_score,
    )
    plan = out._jdf.queryExecution().executedPlan().toString()
    assert "DistanceJoin" in plan
    assert out.columns[:3] == ["base_id", "base_provider", "base_record_type"]
    assert out.columns[-len(PAIR_COLUMNS) :] == list(PAIR_COLUMNS)
    pairs = {
        (row.id, row.base_id): (row.record_type, row.base_record_type)
        for row in out.collect()
    }
    assert pairs == {
        ("a", "b"): ("baseline", "candidate"),
        ("a", "c"): ("baseline", "candidate"),
        ("b", "c"): ("candidate", "candidate"),
    }


def test_matching_writes_blocking_with_the_jar_engines(spark, model_jar, tmp_path):
    candidates = build_places_dataframe(
        spark,
        [
            build_place_record("b", "blue cafe", 0.0, 0.0),
            build_place_record("c", "blue cafe", 0.0, 0.001),
            build_place_record("h", "blue cafe", 0.0, 0.0002),
        ],
    ).withColumn(
        "base_ids",
        F.when(F.col("id") == "h", F.array(F.lit("a"))).cast("array<string>"),
    )
    baseline = build_places_dataframe(
        spark, [build_place_record("a", "red bar", 0.0, 0.0005)]
    )
    with pytest.raises(NotImplementedError, match="scoring"):
        job.matching(SetupOutput(candidates, baseline), model_jar, str(tmp_path))
    out = stage_io.read_output(spark, str(tmp_path), "blocking", job.FEATURE_TYPE)
    assert {(row.id, row.base_id) for row in out.collect()} == {("b", "c")}


def test_compute_name_idf_weighs_rare_tokens_more(shipped_jar_name):
    sketch, num_documents = load_name_idf_sketch(shipped_jar_name)
    got = compute_name_idf(sketch, num_documents, ["cafe", "rare", "cafe", None])
    assert got == pytest.approx(
        {
            "cafe": math.log(NUM_DOCUMENTS / 100) + 1,
            "rare": math.log(NUM_DOCUMENTS / 1) + 1,
        }
    )
    assert compute_name_idf(sketch, num_documents, None) == {}


def test_predict_match_probability_matches_the_booster(shipped_jar_name):
    booster = load_xgboost_booster(shipped_jar_name, BLOCKING_BOOSTER_RESOURCE)
    feature_columns = [
        pd.Series([0.5, 0.0]),
        pd.Series([1.0, float("nan")]),
        pd.Series([10.0, 300.0]),
    ]
    expected = booster.inplace_predict(
        np.column_stack(feature_columns).astype(np.float32)
    )
    assert predict_match_probability(
        booster, feature_columns
    ).tolist() == pytest.approx(expected.tolist())


@pytest.mark.parametrize("name", sorted(features.BLOCKING_FEATURES))
def test_features_read_only_their_declared_inputs(spark, name):
    inputs = features.FEATURE_INPUTS[name]
    types = {features.NAME_IDF: "map<string,double>"}
    schema = ", ".join(
        [
            f"{features.GEO_DISTANCE} double",
            *(
                f"{side}{column_name} {types.get(column_name, 'string')}"
                for column_name in inputs
                for side in ("", "base_")
            ),
        ]
    )
    df = spark.createDataFrame([], schema)
    df.select(features.BLOCKING_FEATURES[name]()).collect()


def test_calculate_pair_features(spark):
    column_names = [
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
    df = spark.createDataFrame(
        rows, ", ".join(f"{column_name} string" for column_name in column_names)
    )
    got = df.select(
        features.calculate_name_similarity().alias("n"),
        features.calculate_phone_number_similarity_availability().alias("p"),
        features.calculate_house_number_similarity_availability().alias("h"),
    ).collect()
    assert [tuple(row) for row in got] == [
        (0.75, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, -1.0, -1.0),
    ]
