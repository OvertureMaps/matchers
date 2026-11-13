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
from conftest import train_booster
from pyspark.sql import Row
from pyspark.sql import functions as F
from test_places_blocking import frame, record

from overture_matchers.core import io as stage_io
from overture_matchers.core.stages import SetupOutput
from overture_matchers.places import features, job, model, model_jar, similarity
from overture_matchers.places.model import MARGIN, PROBABILITY
from overture_matchers.places.scoring import Scorer, score_pairs

SCORING_FEATURES = sorted(features.SCORING_FEATURES)
SCORING_THRESHOLD = 0.5
# Names repeated at least 2 and 10 times in the H3 cell of (0, 0).
NAME_H3_GE2 = ["blue cafe", "rare"]
NAME_H3_GE10 = ["blue cafe"]

CACHE = "test_embeddings_cache"


def name_key_at_origin(spark, name):
    res = features.NAME_H3_RESOLUTION
    cell = spark.sql(f"SELECT ST_H3CellIDs(ST_Point(0.0, 0.0), {res}, false)[0]")
    return f"{res}:{cell.first()[0]}:{name}"


def _name_h3_filter(spark, names):
    df = spark.createDataFrame([(name_key_at_origin(spark, n),) for n in names], ["k"])
    bloom = df._jdf.stat().bloomFilter("k", 100, 0.001)
    out = spark._jvm.java.io.ByteArrayOutputStream()
    bloom.writeTo(out)
    return bytes(out.toByteArray())


@pytest.fixture(scope="module")
def scoring_jar(spark, jar, tmp_path_factory):
    """The blocking jar plus a scoring model, shipped to the workers."""
    rng = np.random.default_rng(0)
    x = rng.random((400, len(SCORING_FEATURES)))
    booster = train_booster(
        x, (x[:, SCORING_FEATURES.index("name_similarity")] > 0.5).astype(int)
    )
    path = tmp_path_factory.mktemp("jar") / "tiny-scoring-model.jar"
    shutil.copy(jar, path)
    with zipfile.ZipFile(path, "a") as z:
        names = z.read(model.NAME_SKETCH_RESOURCE)
        z.writestr(
            "places_matcher_model/metadata/config",
            json.dumps(
                {
                    "feature_order": SCORING_FEATURES,
                    "prediction_threshold": SCORING_THRESHOLD,
                }
            ),
        )
        z.writestr(model.SCORING_BOOSTER_RESOURCE, booster)
        z.writestr(model.ADDRESS_SKETCH_RESOURCE, names)
        z.writestr(model.NAME_H3_GE2_RESOURCE, _name_h3_filter(spark, NAME_H3_GE2))
        z.writestr(model.NAME_H3_GE10_RESOURCE, _name_h3_filter(spark, NAME_H3_GE10))
    spark.sparkContext.addFile(str(path))
    return str(path)


IDF = {"blue": 3.0, "cafe": 1.0, "main": 2.0, "st": 1.5}.get
NAME_KEYS = {"5:7:blue cafe": 1, "5:7:red cafe": 2}
LOOKUPS = features.PairwiseLookups(
    name_idf=lambda t: IDF(t, 1.0),
    address_idf=lambda t: IDF(t, 1.0),
    name_repeated=NAME_KEYS.__contains__,
    name_common=lambda key: NAME_KEYS.get(key) == 2,
)


def first_feature_as_probability(columns):
    return F.struct(columns[0].alias(MARGIN), columns[0].alias(PROBABILITY))


def stand_in_scorer(order, threshold=0.95):
    return Scorer(order, threshold, lambda: LOOKUPS, first_feature_as_probability)


def jar_scorer(spark, path):
    with model_jar.open_jar(spark, path) as jar:
        config = json.loads(jar.read(model_jar.CONFIG_RESOURCE))
    return job.jar_scorer(config, os.path.basename(path))


def side(id, name, lat, **fields):
    row = record(id, name, 0.0, lat).asDict()
    row.update(fields)
    return Row(**row)


def pairs_of(spark, left, right):
    base = frame(spark, [right])
    base = base.select([F.col(c).alias(f"base_{c}") for c in base.columns])
    return frame(spark, [left]).crossJoin(base)


@pytest.fixture
def pair(spark):
    shared = {"taxonomy_primary": "coffee_shop", "website_domains": ["x.com"]}
    left = side(
        "a",
        "blue cafe",
        0.0,
        normalized_phone="5551234",
        website_urls=["x.com/a/b"],
        **shared,
    )
    right = side(
        "b",
        "blue cafes",
        0.001,
        normalized_phone="0001234",
        website_urls=["x.com/a"],
        **shared,
    )
    # The H3 name bin trims and lowercases the raw name.
    left = Row(**{**left.asDict(), "name": "  Blue Cafe "})
    return pairs_of(spark, left, right)


@pytest.fixture
def cache(spark):
    def sha(text):
        return hashlib.sha256(text.encode()).hexdigest()

    rows = [
        ("name", sha("blue cafe"), bytes([127, 0])),
        ("taxonomy", sha("coffee shop"), bytes([0, 127])),
        ("taxonomy", sha("coffee shop"), bytes([0, 127])),
    ]
    spark.createDataFrame(
        rows, "field_type string, input_hash string, embedding binary"
    ).createOrReplaceTempView(CACHE)
    return spark.table(CACHE)


def test_score_pairs_adds_every_feature_and_the_score(spark, scoring_jar, pair, cache):
    out = score_pairs(pair, jar_scorer(spark, scoring_jar), cache)
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

    with zipfile.ZipFile(scoring_jar) as z:
        booster = xgboost.Booster(
            model_file=bytearray(z.read(model.SCORING_BOOSTER_RESOURCE))
        )
    x = np.array(
        [[np.nan if row[f] is None else row[f] for f in SCORING_FEATURES]], np.float32
    )
    expected = float(booster.inplace_predict(x)[0])
    margin = float(booster.inplace_predict(x, predict_type="margin")[0])
    assert row.composite_score == expected
    assert row.prediction == int(expected >= SCORING_THRESHOLD)
    assert row.probability.toArray().tolist() == [1.0 - expected, expected]
    assert row.rawPrediction.toArray().tolist() == [-margin, margin]
    np.testing.assert_array_equal(
        row.features.toArray(),
        [np.nan if row[f] is None else float(row[f]) for f in SCORING_FEATURES],
    )


def test_scoring_runs_each_python_udf_once(spark, scoring_jar, pair, cache):
    out = score_pairs(pair, jar_scorer(spark, scoring_jar), cache)
    plan = out._jdf.queryExecution().executedPlan().toString()
    assert plan.count("BatchEvalPython") == 1
    assert plan.count("ArrowEvalPython") == 1


def test_cosine_features_need_a_cache_table(pair):
    scorer = stand_in_scorer((features.NAME_COSINE,))
    with pytest.raises(ValueError, match="embeddingsCacheTable"):
        score_pairs(pair, scorer, None)


def test_embeddings_cache_is_resolved_only_for_cosine_features(spark):
    plain = stand_in_scorer(("name_similarity",))
    assert job.embeddings_cache(spark, plain, "no_such_table") is None
    cosine = stand_in_scorer((features.NAME_COSINE,))
    with pytest.raises(ValueError, match="embeddingsCacheTable"):
        job.embeddings_cache(spark, cosine, None)


def test_scorer_rejects_an_unknown_feature():
    config = {"feature_order": ["geo_distance", "nope"], "prediction_threshold": 0.5}
    with pytest.raises(ValueError, match="nope"):
        Scorer.from_config(config, lambda: LOOKUPS, first_feature_as_probability)


def test_string_features_count_utf16_units():
    # The emoji is one code point but two UTF-16 units.
    assert similarity.normalized_similarity("a\U0001f600", "a") == pytest.approx(1 / 3)
    assert similarity.prefix_score("a\U0001f600", "a\U0001f601") == pytest.approx(2 / 3)
    assert similarity.suffix_score("\U0001f600b", "\U0001f600b") == 1.0
    # Reversal splits the surrogate pair, so the low units differ first.
    assert similarity.suffix_score("x\U0001f600", "x\U0001f601") == 0.0


def test_tfidf_sums_in_jvm_set_order(spark):
    jvm = spark._jvm
    if not jvm.scala.util.Properties.versionNumberString().startswith("2.12."):
        pytest.skip("the served model's sets iterate in Scala 2.12 order")

    def jvm_intersection(left, right):
        sets = []
        for tokens in (left, right):
            items = jvm.java.util.ArrayList()
            for t in tokens:
                items.add(t)
            sets.append(
                jvm.scala.collection.JavaConverters.asScalaBufferConverter(items)
                .asScala()
                .toSet()
            )
        it = sets[0].intersect(sets[1]).iterator()
        out = []
        while it.hasNext():
            out.append(it.next())
        return out

    rng = random.Random(0)
    pool = ["".join(rng.choices(string.ascii_lowercase, k=4)) for _ in range(30)]
    for _ in range(200):
        left = rng.sample(pool, rng.randint(1, 12))
        right = set(rng.sample(pool, rng.randint(1, 12)))
        expected = jvm_intersection(left, right)
        assert [t for t in similarity._set_order(left) if t in right] == expected


def test_token_features_treat_null_as_empty():
    idf = {"cafe": 1.0, "blue": 3.0}.__getitem__
    assert similarity.tfidf_overlap(["blue", None], None, idf) == 0.0
    assert similarity.tfidf_overlap(["blue", "cafe", None], ["cafe"], idf) == 1.0
    assert similarity.residual_idf_max(["blue", "cafe"], ["cafe"], idf) == 3.0
    assert similarity.residual_idf_max(None, None, idf) == -1.0
    assert similarity.fuzzy_jaccard(None, []) == 1.0
    assert similarity.website_path_similarity(["x.com", None], ["x.com/"]) == 1.0
    assert similarity.website_path_similarity(["x.com"], ["y.com"]) == -1.0


def test_score_pairs_without_cosine_features_skips_the_cache(pair):
    scorer = stand_in_scorer(("name_similarity", "phone_last_digits_match"))
    (row,) = score_pairs(pair, scorer, None).collect()
    assert row.features.toArray().tolist() == pytest.approx([0.9, 1.0])
    assert row.rawPrediction.toArray().tolist() == pytest.approx([-0.9, 0.9])
    assert row.probability.toArray().tolist() == pytest.approx([0.1, 0.9])
    assert row.prediction == 0
    assert row.composite_score == pytest.approx(0.9)
    assert features.NAME_VECTOR not in row


def pairwise_row(left, right):
    empty = dict.fromkeys(features.PAIRWISE_INPUTS)
    return Row(
        **{**empty, **left},
        **{f"base_{k}": v for k, v in {**empty, **right}.items()},
    )


def test_pairwise_features_weigh_names_by_their_difference():
    p = pairwise_row(
        {
            "name": " Blue Cafe\t",
            "name_clean": "blue cafe",
            "name_clean_tokens": ["blue", "cafe"],
            "name_h3_cell": 7,
            "name_vector": bytes([127, 0]),
            "taxonomy_vector": bytes([127]),
        },
        {
            "name": "Red Cafe",
            "name_clean": "blue cafes",
            "name_clean_tokens": ["blue", "cafes"],
            "name_h3_cell": 7,
            "name_vector": bytes([0, 127]),
            "taxonomy_vector": bytes([127, 0]),
            "address_clean": "1 main st",
            "address_clean_tokens": ["1", "main", "st"],
        },
    )
    out = features._pairwise(p, LOOKUPS)
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
    # The base name is common in its cell.
    assert out["unique_name_h3_cell_count_bin"] == 2.0
    assert out[features.NAME_COSINE] == 0.0
    assert out[features.TAXONOMY_COSINE] is None


def test_name_h3_bin_ignores_blank_and_missing_names():
    assert features._name_h3_bin("blue cafe", 7, LOOKUPS) == 1
    assert features._name_h3_bin(" \t", 7, LOOKUPS) == 0
    assert features._name_h3_bin("blue cafe", None, LOOKUPS) == 0


def test_scoring_lookups_read_the_jar(spark, scoring_jar):
    lookups = model.scoring_lookups(os.path.basename(scoring_jar))
    assert lookups.name_idf("rare") == pytest.approx(math.log(100) + 1)
    assert lookups.address_idf("cafe") == 1.0
    assert lookups.name_repeated(name_key_at_origin(spark, "rare"))
    assert not lookups.name_common(name_key_at_origin(spark, "rare"))


def test_token_similarity_claims_each_token_once():
    assert similarity.normalized_similarity("", None) == 1.0
    assert similarity.prefix_score("", "a") == 0.0
    assert similarity.suffix_score("ab", None) == 0.0
    assert similarity.fuzzy_jaccard(["ab", "ab"], ["ab"]) == 0.5
    assert similarity.fuzzy_jaccard(["ab"], []) == 0.0
    assert similarity.shared_token_ratio(["ab", "ab"], ["ab", "ab"]) == 0.5
    assert similarity.shared_token_ratio(None, ["ab"]) == 0.0
    assert similarity.is_token_subset(["ab"], ["ab", "cd"])
    assert not similarity.is_token_subset(["ab", "ef"], ["cd"])
    assert not similarity.is_token_subset(None, ["ab"])
    assert similarity.website_path_similarity(["x.com/a"], ["x.com"]) == 0.0
    assert similarity.website_path_similarity(["x.com/a/b"], ["x.com/a/c"]) == 1 / 3
    assert similarity.website_path_similarity(None, ["x.com"]) == -1.0


def test_matching_writes_checkpoints_and_matches_with_the_jar_engines(
    spark, scoring_jar, cache, tmp_path
):
    spark.sparkContext.setCheckpointDir(str(tmp_path / "checkpoints"))
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
    out = job.matching(
        SetupOutput(candidates, baseline), scoring_jar, str(tmp_path), CACHE
    )
    for name in ("blocking", "scores"):
        out_ = stage_io.read_output(spark, str(tmp_path), name, job.FEATURE_TYPE)
        assert {(r.id, r.base_id) for r in out_.collect()} == {("b", "c")}
    validated = stage_io.read_output(
        spark, str(tmp_path), "matches_validated", job.FEATURE_TYPE
    )
    assert {(r.id, r.base_id) for r in validated.collect()} == {("b", "c"), ("h", "a")}
    expected = {("b", "c", "candidate_candidate"), ("h", "a", "candidate_baseline")}
    assert {(r.id, r.base_id, r.match_sub_type) for r in out.matches.collect()} == (
        expected
    )
    quiet = tmp_path / "quiet"
    out = job.matching(
        SetupOutput(candidates, baseline),
        scoring_jar,
        str(quiet),
        CACHE,
        output_debug=False,
    )
    assert {(r.id, r.base_id, r.match_sub_type) for r in out.matches.collect()} == (
        expected
    )
    assert (quiet / "scores").exists()
    assert not (quiet / "matches_validated").exists()


def test_margin_and_probability_match_the_booster(scoring_jar):
    booster = model.load_booster(
        os.path.basename(scoring_jar), model.SCORING_BOOSTER_RESOURCE
    )
    rng = np.random.default_rng(1)
    columns = [pd.Series(c) for c in rng.random((len(SCORING_FEATURES), 3))]
    columns[0][1] = float("nan")
    matrix = np.column_stack(columns).astype(np.float32)
    out = model.margin_and_probability(booster, columns)
    assert out.dtypes.tolist() == [np.float64, np.float64]
    assert out[MARGIN].tolist() == pytest.approx(
        booster.inplace_predict(matrix, predict_type="margin").tolist()
    )
    assert out[PROBABILITY].tolist() == pytest.approx(
        booster.inplace_predict(matrix).tolist()
    )
