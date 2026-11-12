# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Tests for the places setup stage and the helpers it uses."""

import math

import pytest
from pyspark.sql import functions as F

from overture_matchers.core.stages import RawInputs
from overture_matchers.places.coreword import h3_parent
from overture_matchers.places.job import SetupOutput, setup
from overture_matchers.places.prepare import normalize_phone, prepare_baseline
from overture_matchers.quadkey import focal_quadkey
from overture_matchers.sketches import BloomFilter

POINTS = [(0.0, 0.0), (40.7128, -74.006), (-33.8688, 151.2093), (51.5, -0.12)]


def _reference_quadkey(lat, lng, zoom):
    n = 2**zoom
    x = int((lng + 180) / 360 * n)
    sin = math.sin(math.radians(lat))
    y = int((0.5 - math.log((1 + sin) / (1 - sin)) / (4 * math.pi)) * n)
    return "".join(
        str((x >> b & 1) + 2 * (y >> b & 1)) for b in range(zoom - 1, -1, -1)
    )


def test_focal_quadkey_matches_the_tile_scheme(spark):
    df = spark.createDataFrame(POINTS + [(None, 1.0)], "lat double, lng double")
    got = df.select(focal_quadkey(F.col("lat"), F.col("lng"), 16).alias("k")).collect()
    expected = [_reference_quadkey(lat, lng, 16) for lat, lng in POINTS] + [None]
    assert [r.k for r in got] == expected


def test_bloom_filter_agrees_with_spark(spark):
    jvm = spark._jvm
    jfilter = jvm.org.apache.spark.util.sketch.BloomFilter.create(1000, 0.01)
    members = [f"9:{i}:coffee" for i in range(50)]
    for m in members:
        jfilter.putString(m)
    out = jvm.java.io.ByteArrayOutputStream()
    jfilter.writeTo(out)
    ours = BloomFilter(bytes(out.toByteArray()))
    others = [f"3:{i}:other" for i in range(200)]
    for value in members + others:
        assert ours.might_contain_string(value) == jfilter.mightContainString(value)


def test_h3_parent_matches_h3_core(spark):
    h3 = spark._jvm.com.uber.h3core.H3Core.newInstance()
    cell = (
        spark.sql("select ST_H3CellIDs(ST_Point(-74.006, 40.7128), 9, false)[0] c")
        .first()
        .c
    )
    for res in (7, 5, 3):
        assert h3_parent(cell, res) == h3.cellToParent(cell, res)


@pytest.mark.parametrize(
    ("number", "region", "digits"),
    [
        ("(212) 555-0100", "US", "12125550100"),
        ("+44 20 7946 0958", "GB", "442079460958"),
        ("no digits", "US", ""),
        (None, "US", None),
    ],
)
def test_normalize_phone(number, region, digits):
    assert normalize_phone(number, region) == digits


def _address(freeform, locality=None):
    return [{"freeform": freeform, "locality": locality, "country": "US"}]


@pytest.mark.normalizer
@pytest.mark.libpostal
def test_setup_columns_and_history(spark, model_jar):
    schema = (
        "id string, names struct<primary:string>, "
        "addresses array<struct<freeform:string,locality:string,country:string>>, "
        "geometry binary, sources array<struct<property:string,record_id:string>>, "
        "phones array<string>, websites array<string>, "
        "taxonomy struct<primary:string>"
    )
    base = "ST_AsBinary(ST_Point(-74.006, 40.7128))"
    rows = [
        (
            "c1",
            ("Joe's Pizza",),
            _address("781 Franklin Ave", "Brooklyn"),
            None,
            [(None, "Src@1"), ("x", "other@2")],
            ["(212) 555-0100"],
            ["https://www.Example.com:8080/a/?q=1", "nodot"],
            ("pizza_restaurant",),
        ),
        ("c2", ("Nameless",), None, None, [], None, None, None),
    ]
    df = spark.createDataFrame(rows, schema).withColumn("geometry", F.expr(base))
    baseline = (
        df.where("id = 'c1'")
        .withColumn("id", F.lit("b1"))
        .withColumn("provider", F.lit("acme"))
        .withColumn("corpus_state", F.lit("retired"))
    )
    out = setup(RawInputs(df, baseline), "acme", model_jar)
    assert isinstance(out, SetupOutput)
    cand = {r.id: r for r in out.candidates.collect()}
    assert out.candidates.columns[0] == "id"
    assert out.candidates.columns[-1] == "base_ids"
    assert out.baseline.columns[-1] == "active"
    c1 = cand["c1"]
    assert c1.house_number == "781"
    assert c1.address == "franklin ave"
    assert c1.normalized_phone == "12125550100"
    assert c1.primary_source == "src"
    assert c1.website_domains == ["example.com", "nodot"]
    assert c1.website_urls == ["example.com:8080/a"]
    assert c1.blocking_keys == [_reference_quadkey(40.7128, -74.006, 16)]
    assert c1.base_ids == ["b1"]
    assert cand["c2"].base_ids is None
    assert out.baseline.first().active == 0
    no_history = setup(RawInputs(df, baseline), "acme", model_jar, match_history=False)
    assert no_history.candidates.where("base_ids is not null").count() == 0


def test_baseline_provider_rules(spark):
    plain = spark.createDataFrame([("b1",)], "id string")
    with_provider = plain.withColumn("provider", F.lit("acme"))
    for df, given in ((plain, None), (with_provider, "other")):
        with pytest.raises(ValueError, match="baselineProvider"):
            prepare_baseline(df, "model.jar", given)
