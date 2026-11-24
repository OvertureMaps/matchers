# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Tests for place-pair scoring features and scores on a tiny model.

Scoring rules run with stand-in engines. The scoring jar fixture checks the
real engines and the stage that binds them.
"""

import hashlib
import json
import math
import os
import random
import shutil
import string
import zipfile

import numpy as np
import pandas as pd
import pytest
import xgboost
from conftest import MODEL_JAR, train_xgboost_booster
from pyspark.sql import Row
from pyspark.sql import functions as F
from test_places_blocking import build_place_record, build_places_dataframe

from overture_matchers.core import io as stage_io
from overture_matchers.core.stages import SetupOutput
from overture_matchers.places import features, job, model, model_jar, similarity
from overture_matchers.places.model import MARGIN, PROBABILITY
from overture_matchers.places.scoring import PlacesPairScorer, score_pairs

SCORING_FEATURES = sorted(features.SCORING_FEATURES)
SCORING_THRESHOLD = 0.5
# Names repeated at least 2 and 10 times in the H3 resolution-5 cell of (0, 0).
NAME_H3_GE2 = ["blue cafe", "rare"]
NAME_H3_GE10 = ["blue cafe"]

CACHE = "test_embeddings_cache"


def build_name_h3_key_at_origin(spark, name):
    cell = spark.sql("SELECT ST_H3CellIDs(ST_Point(0.0, 0.0), 5, false)[0]")
    return f"5:{cell.first()[0]}:{name}"


def serialize_name_h3_bloom_filter(spark, names):
    df = spark.createDataFrame(
        [(build_name_h3_key_at_origin(spark, name),) for name in names], ["k"]
    )
    bloom = df._jdf.stat().bloomFilter("k", 100, 0.001)
    out = spark._jvm.java.io.ByteArrayOutputStream()
    bloom.writeTo(out)
    return bytes(out.toByteArray())


def build_stand_in_unique_name_bin(name, latitude, longitude):
    normalized = F.trim(F.lower(name))
    valid_point = latitude.isNotNull() & longitude.isNotNull()
    return (
        F.when(~valid_point | normalized.isNull() | (normalized == ""), F.lit(0))
        .when(normalized.isin(*NAME_H3_GE10), F.lit(2))
        .when(normalized.isin(*NAME_H3_GE2), F.lit(1))
        .otherwise(F.lit(0))
    )


@pytest.fixture(scope="module")
def scoring_jar(spark, jar, tmp_path_factory):
    """The blocking jar plus a scoring model, shipped to the workers."""
    rng = np.random.default_rng(0)
    feature_matrix = rng.random((400, len(SCORING_FEATURES)))
    booster = train_xgboost_booster(
        feature_matrix,
        (feature_matrix[:, SCORING_FEATURES.index("name_similarity")] > 0.5).astype(
            int
        ),
    )
    path = tmp_path_factory.mktemp("jar") / "tiny-scoring-model.jar"
    shutil.copy(jar, path)
    with zipfile.ZipFile(path, "a") as jar_file:
        names = jar_file.read(model.NAME_SKETCH_RESOURCE)
        jar_file.writestr(
            "places_matcher_model/metadata/config",
            json.dumps(
                {
                    "feature_order": SCORING_FEATURES,
                    "prediction_threshold": SCORING_THRESHOLD,
                }
            ),
        )
        jar_file.writestr(model.SCORING_BOOSTER_RESOURCE, booster)
        jar_file.writestr(model.ADDRESS_SKETCH_RESOURCE, names)
        jar_file.writestr(
            model.UNIQUE_NAME_H3_BIN_GE2_RESOURCE,
            serialize_name_h3_bloom_filter(spark, NAME_H3_GE2),
        )
        jar_file.writestr(
            model.UNIQUE_NAME_H3_BIN_GE10_RESOURCE,
            serialize_name_h3_bloom_filter(spark, NAME_H3_GE10),
        )
    spark.sparkContext.addFile(str(path))
    if MODEL_JAR:
        yield str(path)
        return
    patcher = pytest.MonkeyPatch()
    patcher.setattr(job, "register_unique_name_h3_bin_udf", lambda *_: None)
    patcher.setattr(
        job, "bind_unique_name_h3_bin_udf", lambda: build_stand_in_unique_name_bin
    )
    try:
        yield str(path)
    finally:
        patcher.undo()


IDF = {"blue": 3.0, "cafe": 1.0, "main": 2.0, "st": 1.5}.get
LOOKUPS = features.PairwiseLookups(
    compute_name_idf=lambda token: IDF(token, 1.0),
    compute_address_idf=lambda token: IDF(token, 1.0),
)


def select_first_feature_as_scores(feature_columns):
    return F.struct(
        feature_columns[0].alias(MARGIN), feature_columns[0].alias(PROBABILITY)
    )


def build_stand_in_scorer(feature_order, prediction_threshold=0.95):
    return PlacesPairScorer(
        feature_order=feature_order,
        prediction_threshold=prediction_threshold,
        get_lookups=lambda: LOOKUPS,
        compute_unique_name_bin=build_stand_in_unique_name_bin,
        predict_scores=select_first_feature_as_scores,
    )


def load_scorer_from_jar(spark, jar_path):
    with model_jar.open_model_jar(spark, jar_path) as jar_file:
        config = json.loads(jar_file.read(model_jar.CONFIG_RESOURCE))
        job.register_unique_name_h3_bin_udf(
            spark,
            jar_file.read(model.UNIQUE_NAME_H3_BIN_GE2_RESOURCE),
            jar_file.read(model.UNIQUE_NAME_H3_BIN_GE10_RESOURCE),
        )
    return job.build_places_pair_scorer(config, os.path.basename(jar_path))


def build_scored_place_side(record_id, name, latitude, **fields):
    row = build_place_record(record_id, name, 0.0, latitude).asDict()
    row.update(fields)
    return Row(**row)


def build_pair_dataframe(spark, left_row, right_row):
    base = build_places_dataframe(spark, [right_row])
    base = base.select(
        [
            F.col(column_name).alias(f"base_{column_name}")
            for column_name in base.columns
        ]
    )
    return build_places_dataframe(spark, [left_row]).crossJoin(base)


@pytest.fixture
def pair(spark):
    shared = {"taxonomy_primary": "coffee_shop", "website_domains": ["x.com"]}
    left = build_scored_place_side(
        "a",
        "blue cafe",
        0.0,
        normalized_phone="5551234",
        website_urls=["x.com/a/b"],
        **shared,
    )
    right = build_scored_place_side(
        "b",
        "blue cafes",
        0.001,
        normalized_phone="0001234",
        website_urls=["x.com/a"],
        **shared,
    )
    # The H3 name bin trims and lowercases the raw name.
    left = Row(**{**left.asDict(), "name": "  Blue Cafe "})
    return build_pair_dataframe(spark, left, right)


@pytest.fixture
def cache(spark):
    def compute_sha256(text):
        return hashlib.sha256(text.encode()).hexdigest()

    rows = [
        ("name", compute_sha256("blue cafe"), bytes([127, 0])),
        ("taxonomy", compute_sha256("coffee shop"), bytes([0, 127])),
        ("taxonomy", compute_sha256("coffee shop"), bytes([0, 127])),
    ]
    spark.createDataFrame(
        rows, "field_type string, input_hash string, embedding binary"
    ).createOrReplaceTempView(CACHE)
    return spark.table(CACHE)


def test_score_pairs_adds_every_feature_and_the_score(spark, scoring_jar, pair, cache):
    out = score_pairs(pair, load_scorer_from_jar(spark, scoring_jar), cache)
    assert out.columns == [
        *pair.columns,
        *SCORING_FEATURES,
        "features",
        "rawPrediction",
        "probability",
        "prediction",
        "composite_score",
    ]
    (row,) = out.collect()
    assert row.name_similarity == pytest.approx(0.9)
    assert row.phone_last_digits_match == 1.0
    assert row.website_domain_match == 1
    assert row.website_path_similarity == 0.5
    assert row.geo_distance == pytest.approx(111.2, abs=0.1)
    assert row.unique_name_h3_cell_count_bin == 2.0
    assert row.address_similarity_availability == -1.0
    assert row.name_cosine_similarity is None
    assert row.taxonomy_cosine_similarity == 1.0

    with zipfile.ZipFile(scoring_jar) as jar_file:
        booster = xgboost.Booster(
            model_file=bytearray(jar_file.read(model.SCORING_BOOSTER_RESOURCE))
        )
    feature_matrix = np.array(
        [
            [
                np.nan if row[feature_name] is None else row[feature_name]
                for feature_name in SCORING_FEATURES
            ]
        ],
        np.float32,
    )
    expected = float(booster.inplace_predict(feature_matrix)[0])
    margin = float(booster.inplace_predict(feature_matrix, predict_type="margin")[0])
    assert row.composite_score == expected
    assert row.prediction == int(expected >= SCORING_THRESHOLD)
    assert row.probability.toArray().tolist() == [1.0 - expected, expected]
    assert row.rawPrediction.toArray().tolist() == [-margin, margin]
    np.testing.assert_array_equal(
        row.features.toArray(),
        [
            np.nan if row[feature_name] is None else float(row[feature_name])
            for feature_name in SCORING_FEATURES
        ],
    )


def test_scoring_runs_each_python_udf_once(spark, scoring_jar, pair, cache):
    out = score_pairs(pair, load_scorer_from_jar(spark, scoring_jar), cache)
    plan = out._jdf.queryExecution().executedPlan().toString()
    assert plan.count("BatchEvalPython") == 1
    assert plan.count("ArrowEvalPython") == 1


def test_cosine_features_need_a_cache_table(pair):
    scorer = build_stand_in_scorer((features.NAME_COSINE_SIMILARITY,))
    with pytest.raises(ValueError, match="embeddingsCacheTable"):
        score_pairs(pair, scorer, None)


def test_embeddings_cache_is_resolved_only_for_cosine_features(spark):
    plain = build_stand_in_scorer(("name_similarity",))
    assert job.load_embeddings_cache(spark, plain, "no_such_table") is None
    cosine = build_stand_in_scorer((features.NAME_COSINE_SIMILARITY,))
    with pytest.raises(ValueError, match="embeddingsCacheTable"):
        job.load_embeddings_cache(spark, cosine, None)


def test_scorer_rejects_an_unknown_feature():
    config = {"feature_order": ["geo_distance", "nope"], "prediction_threshold": 0.5}
    with pytest.raises(ValueError, match="nope"):
        PlacesPairScorer.from_config(
            config,
            lambda: LOOKUPS,
            build_stand_in_unique_name_bin,
            select_first_feature_as_scores,
        )


def test_string_features_count_utf16_units():
    # The emoji is one code point but two UTF-16 units.
    assert similarity.compute_normalized_similarity(
        "a\U0001f600", "a"
    ) == pytest.approx(1 / 3)
    assert similarity.get_string_prefix_score(
        "a\U0001f600", "a\U0001f601"
    ) == pytest.approx(2 / 3)
    assert similarity.get_string_suffix_score("\U0001f600b", "\U0001f600b") == 1.0
    # Reversal splits the surrogate pair, so the low units differ first.
    assert similarity.get_string_suffix_score("x\U0001f600", "x\U0001f601") == 0.0


def test_tfidf_sums_in_jvm_set_order(spark):
    jvm = spark._jvm
    if not jvm.scala.util.Properties.versionNumberString().startswith("2.12."):
        pytest.skip("requires the JVM 2.12 hash-trie set runtime")

    def compute_jvm_intersection(left, right):
        sets = []
        for tokens in (left, right):
            items = jvm.java.util.ArrayList()
            for token in tokens:
                items.add(token)
            sets.append(
                jvm.scala.collection.JavaConverters.asScalaBufferConverter(items)
                .asScala()
                .toSet()
            )
        iterator = sets[0].intersect(sets[1]).iterator()
        out = []
        while iterator.hasNext():
            out.append(iterator.next())
        return out

    rng = random.Random(0)
    pool = ["".join(rng.choices(string.ascii_lowercase, k=4)) for _ in range(30)]
    for _ in range(200):
        left = rng.sample(pool, rng.randint(1, 12))
        right = set(rng.sample(pool, rng.randint(1, 12)))
        expected = compute_jvm_intersection(left, right)
        assert [
            token for token in similarity._order_distinct_tokens(left) if token in right
        ] == expected


def test_token_features_treat_null_as_empty():
    compute_idf = {"cafe": 1.0, "blue": 3.0}.__getitem__
    assert similarity.get_tfidf_overlap(["blue", None], None, compute_idf) == 0.0
    assert (
        similarity.get_tfidf_overlap(["blue", "cafe", None], ["cafe"], compute_idf)
        == 1.0
    )
    assert (
        similarity.compute_residual_idf_max(["blue", "cafe"], ["cafe"], compute_idf)
        == 3.0
    )
    assert similarity.compute_residual_idf_max(None, None, compute_idf) == -1.0
    assert similarity.compute_fuzzy_jaccard_similarity(None, []) == 1.0
    assert (
        similarity.compute_website_path_similarity(["x.com", None], ["x.com/"]) == 1.0
    )
    assert similarity.compute_website_path_similarity(["x.com"], ["y.com"]) == -1.0


def test_score_pairs_without_cosine_features_skips_the_cache(pair):
    scorer = build_stand_in_scorer(("name_similarity", "phone_last_digits_match"))
    (row,) = score_pairs(pair, scorer, None).collect()
    assert row.features.toArray().tolist() == pytest.approx([0.9, 1.0])
    assert row.rawPrediction.toArray().tolist() == pytest.approx([-0.9, 0.9])
    assert row.probability.toArray().tolist() == pytest.approx([0.1, 0.9])
    assert row.prediction == 0
    assert row.composite_score == pytest.approx(0.9)
    assert features.NAME_VECTOR not in row


def build_pairwise_input_row(left_fields, right_fields):
    empty = dict.fromkeys(features.PAIRWISE_INPUTS)
    return Row(
        **{**empty, **left_fields},
        **{f"base_{key}": value for key, value in {**empty, **right_fields}.items()},
    )


def test_pairwise_features_weigh_names_by_their_difference():
    pair_row = build_pairwise_input_row(
        {
            "name_clean": "blue cafe",
            "name_clean_tokens": ["blue", "cafe"],
            "name_vector": bytes([127, 0]),
            "taxonomy_vector": bytes([127]),
        },
        {
            "name_clean": "blue cafes",
            "name_clean_tokens": ["blue", "cafes"],
            "name_vector": bytes([0, 127]),
            "taxonomy_vector": bytes([127, 0]),
            "address_clean": "1 main st",
            "address_clean_tokens": ["1", "main", "st"],
        },
    )
    out = features._compute_pairwise_features_for_row(pair_row, LOOKUPS)
    assert out.keys() == set(features.PAIRWISE_FEATURES.fieldNames())
    strict = 1.0 - 0.9
    assert out["name_similarity"] == pytest.approx(0.9)
    assert out["name_strict_prefix_score"] == pytest.approx(strict)
    # "cafe" and "cafes" fall below the token threshold, so only "blue" pairs up.
    assert out["name_strict_token_subset"] == 0.0
    assert out["name_strict_shared_token_ratio"] == pytest.approx(strict / 2)
    assert out["address_similarity_availability"] == -1.0
    assert out["address_token_subset"] == 0
    assert out["name_tfidf"] == 3.0 / 2
    assert out["residual_idf_max"] == 1.0
    assert out[features.NAME_COSINE_SIMILARITY] == 0.0
    assert out[features.TAXONOMY_COSINE_SIMILARITY] is None


def test_calculate_unique_name_h3_cell_count_bin_takes_larger_side(spark):
    pairs = spark.createDataFrame(
        [
            ("  Blue Cafe ", 0.0, 0.0, "other", 0.0, 0.0),
            ("other", 0.0, 0.0, "rare", 0.0, 0.0),
            (" \t", 0.0, 0.0, "other", 0.0, 0.0),
            ("blue cafe", None, 0.0, "other", 0.0, 0.0),
        ],
        "name string, latitude double, longitude double, "
        "base_name string, base_latitude double, base_longitude double",
    )
    bins = [
        row.bin
        for row in pairs.select(
            features.calculate_unique_name_h3_cell_count_bin(
                build_stand_in_unique_name_bin
            ).alias("bin")
        ).collect()
    ]
    assert bins == [2.0, 1.0, 0.0, 0.0]


def test_load_scoring_lookups_reads_the_jar(scoring_jar):
    lookups = model.load_scoring_lookups(os.path.basename(scoring_jar))
    assert lookups.compute_name_idf("rare") == pytest.approx(math.log(100) + 1)
    assert lookups.compute_address_idf("cafe") == 1.0


def test_token_similarity_claims_each_token_once():
    assert similarity.compute_normalized_similarity("", None) == 1.0
    assert similarity.get_string_prefix_score("", "a") == 0.0
    assert similarity.get_string_suffix_score("ab", None) == 0.0
    assert similarity.compute_fuzzy_jaccard_similarity(["ab", "ab"], ["ab"]) == 0.5
    assert similarity.compute_fuzzy_jaccard_similarity(["ab"], []) == 0.0
    assert similarity.compute_shared_token_ratio(["ab", "ab"], ["ab", "ab"]) == 0.5
    assert similarity.compute_shared_token_ratio(None, ["ab"]) == 0.0
    assert similarity.is_token_subset(["ab"], ["ab", "cd"])
    assert not similarity.is_token_subset(["ab", "ef"], ["cd"])
    assert not similarity.is_token_subset(None, ["ab"])
    assert similarity.compute_website_path_similarity(["x.com/a"], ["x.com"]) == 0.0
    assert (
        similarity.compute_website_path_similarity(["x.com/a/b"], ["x.com/a/c"])
        == 1 / 3
    )
    assert similarity.compute_website_path_similarity(None, ["x.com"]) == -1.0


def test_matching_writes_checkpoints_and_matches_with_the_jar_engines(
    graph_session, scoring_jar, cache, tmp_path
):
    candidates = build_places_dataframe(
        graph_session,
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
        graph_session, [build_place_record("a", "red bar", 0.0, 0.0005)]
    )
    out = job.matching(
        SetupOutput(candidates, baseline), scoring_jar, str(tmp_path), CACHE
    )
    for name in ("blocking", "scores"):
        written = stage_io.read_output(
            graph_session, str(tmp_path), name, job.FEATURE_TYPE
        )
        assert {(row.id, row.base_id) for row in written.collect()} == {("b", "c")}
    validated = stage_io.read_output(
        graph_session, str(tmp_path), "matches_validated", job.FEATURE_TYPE
    )
    assert {(row.id, row.base_id) for row in validated.collect()} == {
        ("b", "c"),
        ("h", "a"),
    }
    expected = {("b", "c", "candidate_candidate"), ("h", "a", "candidate_baseline")}
    assert {
        (row.id, row.base_id, row.match_sub_type) for row in out.matches.collect()
    } == expected
    quiet = tmp_path / "quiet"
    quiet_out = job.matching(
        SetupOutput(candidates, baseline),
        scoring_jar,
        str(quiet),
        CACHE,
        output_debug=False,
    )
    assert {
        (row.id, row.base_id, row.match_sub_type) for row in quiet_out.matches.collect()
    } == expected
    assert (quiet / "scores").exists()
    assert not (quiet / "matches_validated").exists()


def test_predict_margin_and_probability_matches_the_booster(scoring_jar):
    booster = model.load_xgboost_booster(
        os.path.basename(scoring_jar), model.SCORING_BOOSTER_RESOURCE
    )
    rng = np.random.default_rng(1)
    feature_columns = [
        pd.Series(column) for column in rng.random((len(SCORING_FEATURES), 3))
    ]
    feature_columns[0][1] = float("nan")
    matrix = np.column_stack(feature_columns).astype(np.float32)
    out = model.predict_margin_and_probability(booster, feature_columns)
    assert out.dtypes.tolist() == [np.float64, np.float64]
    assert out[MARGIN].tolist() == pytest.approx(
        booster.inplace_predict(matrix, predict_type="margin").tolist()
    )
    assert out[PROBABILITY].tolist() == pytest.approx(
        booster.inplace_predict(matrix).tolist()
    )
