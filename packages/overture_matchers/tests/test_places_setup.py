# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Tests for the places setup stage and the helpers it uses."""

import math
import struct

import pytest
from pyspark.sql import functions as F

from overture_matchers.core.stages import RawInputs
from overture_matchers.places import prepare
from overture_matchers.places.job import SetupOutput, setup
from overture_matchers.places.prepare import (
    TextFunctions,
    _build_signature,
    _extract_website_domains_and_urls,
    _get_primary_source,
    _normalize_address,
    _prepare_columns_for_matching,
    _replace_invalid_utf8,
    _stage_reused_columns,
    add_base_ids,
    get_baseline_provider_column,
    normalize_phone_number,
    prepare_baseline,
    prepare_candidates,
)
from overture_matchers.quadkey import get_focal_quadkey

POINTS = [
    (0.0, 0.0),
    (40.7128, -74.006),
    (-33.8688, 151.2093),
    (51.5, -0.12),
    (85.0, 179.999),
    (-85.0, -179.999),
    (89.9, 0.0),
]


def _reference_quadkey(lat, lng, zoom):
    n = 2**zoom
    x = int((lng + 180) / 360 * n)
    sin = math.sin(math.radians(max(-85.05112878, min(85.05112878, lat))))
    y = int((0.5 - math.log((1 + sin) / (1 - sin)) / (4 * math.pi)) * n)
    x, y = min(max(x, 0), n - 1), min(max(y, 0), n - 1)
    return "".join(
        str((x >> b & 1) + 2 * (y >> b & 1)) for b in range(zoom - 1, -1, -1)
    )


def test_get_focal_quadkey_matches_the_tile_scheme(spark):
    df = spark.createDataFrame(POINTS + [(None, 1.0)], "lat double, lng double")
    got = df.select(
        get_focal_quadkey(F.col("lat"), F.col("lng"), 16).alias("k")
    ).collect()
    expected = [_reference_quadkey(lat, lng, 16) for lat, lng in POINTS] + [None]
    assert [r.k for r in got] == expected


@pytest.mark.parametrize(
    ("number", "region", "digits"),
    [
        ("(212) 555-0100", "US", "12125550100"),
        ("+44 20 7946 0958", "GB", "442079460958"),
        ("no digits", "US", ""),
        (None, "US", None),
    ],
)
def test_normalize_phone_number(number, region, digits):
    assert normalize_phone_number(number, region) == digits


def _address(freeform, locality=None):
    return [{"freeform": freeform, "locality": locality, "country": "US"}]


@pytest.mark.normalizer
@pytest.mark.libpostal
def test_setup_columns_and_history(spark):
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
    out = setup(RawInputs(df, baseline), "acme")
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
    no_history = setup(RawInputs(df, baseline), "acme", match_history=False)
    assert no_history.candidates.where("base_ids is not null").count() == 0


def test_baseline_provider_rules(spark):
    plain = spark.createDataFrame([("b1",)], "id string")
    with_provider = plain.withColumn("provider", F.lit("acme"))
    for df, given in ((plain, None), (with_provider, "other")):
        with pytest.raises(ValueError, match="baselineProvider"):
            get_baseline_provider_column(df, given)
    assert (
        plain.select(get_baseline_provider_column(plain, "acme")).first()[0] == "acme"
    )
    assert (
        with_provider.select(get_baseline_provider_column(with_provider, None)).first()[
            0
        ]
        == "acme"
    )


def test_setup_validates_before_shipping(spark):
    plain = spark.createDataFrame([("b1",)], "id string")
    raw = RawInputs(plain, plain)
    with pytest.raises(ValueError, match="baselineProvider"):
        setup(raw, "acme")


def test_add_base_ids(spark):
    cands = spark.createDataFrame(
        [("c1", "acme", "s1"), ("c2", "acme", "s2"), ("c3", "other", "s1")],
        "id string, provider string, signature string",
    )
    base = spark.createDataFrame(
        [("b2", "acme", "s1"), ("b1", "acme", "s1")],
        "id string, provider string, signature string",
    )
    out = {r.id: r.base_ids for r in add_base_ids(cands, base, True).collect()}
    assert out == {"c1": ["b1", "b2"], "c2": None, "c3": None}
    off = add_base_ids(cands, base, False)
    assert off.where("base_ids is not null").count() == 0


def test_replace_invalid_utf8_replaces_invalid_bytes(spark):
    raw = F.unhex(F.lit("41FF42")).cast("string")
    row = (
        spark.range(1)
        .select(
            F.hex(raw.cast("binary")).alias("raw"),
            F.hex(F.encode(_replace_invalid_utf8(raw), "UTF-8")).alias("clean"),
        )
        .first()
    )
    assert (row.raw, row.clean) == ("41FF42", "41EFBFBD42")


def test_build_signature_joins_lowercased_non_null_parts(spark):
    row = (
        spark.range(1)
        .select(
            _build_signature(
                F.lit("Src"),
                F.lit("0123"),
                F.lit(None),
                F.lit("Main St"),
                F.lit("Cafe"),
            ).alias("s")
        )
        .first()
    )
    assert row.s == "src@0123@main st@cafe"


def _components(*pairs):
    parts = [
        F.struct(F.lit(k).alias("label"), F.lit(v).alias("value")) for k, v in pairs
    ]
    return F.array(*parts).cast("array<struct<label:string,value:string>>")


def test_normalize_address_fallbacks(spark):
    cases = [
        ("1 Main St", [("house_number", "1"), ("road", "main st")], "main st"),
        ("1 Main St", [("house_number", "1"), ("city", "x")], ""),
        ("1 Main St", [("house_number", "1")], "Main St"),
        ("Main St", [], "Main St"),
    ]
    rows = [
        spark.range(1)
        .select(
            F.array(F.struct(F.lit(f).alias("freeform"))).alias("addresses"),
            _components(*c).alias("c"),
        )
        .select(
            _normalize_address(F.col("c"), F.col("addresses")[0]["freeform"]).alias("a")
        )
        .first()
        .a
        for f, c, _ in cases
    ]
    assert rows == [e for *_, e in cases]


def test_extract_website_domains_and_urls(spark):
    df = spark.createDataFrame(
        [(["https://www.Example.com:8080/a/?q=1", "nodot", ""],)],
        "websites array<string>",
    )
    domains, urls = _extract_website_domains_and_urls(df)
    row = df.select(domains.alias("d"), urls.alias("u")).first()
    assert (row.d, row.u) == (["example.com", "nodot"], ["example.com:8080/a"])
    no_col = spark.range(1)
    empty_domains, empty_urls = _extract_website_domains_and_urls(no_col)
    empty_row = no_col.select(empty_domains.alias("d"), empty_urls.alias("u")).first()
    assert (empty_row.d, empty_row.u) == ([], [])


def test_get_primary_source_returns_lowercased_prefix_of_first_unqualified_source(
    spark,
):
    df = spark.createDataFrame(
        [
            ([(None, "Src@1"), ("x", "other@2")],),
            ([("x", "other@2"), ("", "Second@3")],),
            ([("x", "other@2")],),
        ],
        "sources array<struct<property:string,record_id:string>>",
    )
    assert [
        r.s
        for r in df.select(_get_primary_source(F.col("sources")).alias("s")).collect()
    ] == [
        "src",
        "second",
        "",
    ]


def _raw_place(spark):
    schema = (
        "id string, names struct<primary:string>, "
        "addresses array<struct<freeform:string,locality:string,country:string>>, "
        "geometry binary, sources array<struct<property:string,record_id:string>>, "
        "phones array<string>, websites array<string>, "
        "taxonomy struct<primary:string>"
    )
    point_wkb = struct.pack("<BIdd", 1, 1, -74.006, 40.7128)
    rows = [
        (
            "c1",
            ("Joe's Pizza",),
            _address("781 Franklin Ave", "Brooklyn"),
            point_wkb,
            [(None, "Src@1")],
            ["(212) 555-0100"],
            ["https://www.Example.com:8080/a/?q=1", "nodot"],
            ("pizza_restaurant",),
        )
    ]
    return spark.createDataFrame(rows, schema)


_STAND_INS = TextFunctions(
    normalize_text=F.lower,
    tokenize_text=lambda c: F.split(c, " "),
    strip_background_tokens=lambda tokens, lat, lng: F.when(
        lat.isNotNull() & lng.isNotNull(), tokens
    ).otherwise(tokens),
)


def test_prepare_columns_for_matching_without_the_model_jar(spark):
    staged = _stage_reused_columns(
        _raw_place(spark),
        _components(("house_number", "781"), ("road", "franklin ave")),
    )
    columns = _prepare_columns_for_matching(staged, F.lit("acme"), _STAND_INS)
    assert list(columns) == [
        "id",
        "name",
        "address",
        "name_clean",
        "name_clean_tokens",
        "name_clean_stripped",
        "name_clean_stripped_tokens",
        "address_clean",
        "address_clean_tokens",
        "latitude",
        "longitude",
        "house_number",
        "country",
        "normalized_phone",
        "blocking_keys",
        "primary_source",
        "signature",
        "provider",
        "taxonomy_primary",
        "website_domains",
        "website_urls",
    ]
    projected = staged.select(*[c.alias(n) for n, c in columns.items()])
    plan = projected._jdf.queryExecution().optimizedPlan().toString().lower()
    assert plan.count("st_geomfromwkb") == 1
    row = projected.first()
    quadkey = _reference_quadkey(40.7128, -74.006, 16)
    assert row.address == "franklin ave"
    assert row.house_number == "781"
    assert row.normalized_phone == "12125550100"
    assert row.blocking_keys == [quadkey]
    assert row.primary_source == "src"
    assert row.signature == f"src@{quadkey}@joe's pizza@franklin ave@pizza_restaurant"
    assert row.provider == "acme"
    assert row.website_domains == ["example.com", "nodot"]
    assert row.website_urls == ["example.com:8080/a"]


def test_prepare_sides_add_their_extra_columns(spark, monkeypatch):
    # Parsing needs libpostal data, so the parsed column is supplied directly.
    components = _components(("house_number", "781"), ("road", "franklin ave"))
    monkeypatch.setattr(prepare, "_parse_first_address", lambda: components)
    raw = _raw_place(spark)
    candidates = prepare_candidates(raw, "acme", _STAND_INS, blocking_resolution=12)
    cand_row = candidates.first()
    assert cand_row.provider == "acme"
    assert cand_row.blocking_keys == [_reference_quadkey(40.7128, -74.006, 12)]
    retired = raw.withColumn("corpus_state", F.lit("retired"))
    assert prepare_baseline(retired, F.lit("b"), _STAND_INS).first().active == 0
    assert prepare_baseline(raw, F.lit("b"), _STAND_INS).first().active == 1
