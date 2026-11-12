# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Places setup steps: the columns matching reads, and the history join.

Each side's columns come from one ``select``. The JVM UDFs (name normalization and
tokenizing) come from the ``places-matcher-model`` jar, which must be on the
Spark classpath. Python UDFs live here, not in the job script, so that workers
can unpickle them.
"""

from __future__ import annotations

import contextlib
import os
import re
from typing import TYPE_CHECKING

import phonenumbers
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType
from sedona.spark.sql.st_functions import ParseAddress

from overture_matchers.places.coreword import strip_background_udf
from overture_matchers.quadkey import focal_quadkey

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

# Resolution of the signature's quadkey and the blocking keys.
BLOCKING_RESOLUTION = 16
LIBPOSTAL_DATA_DIR_CONF = "spark.sedona.libpostal.dataDir"
_NORMALIZE_UDF = "normalize_text"
_TOKENIZE_UDF = "tokenize_text"
_COMPONENTS = "_components"
_SIGNATURE_SEPARATOR = "@"
_ACTIVE_STATE = "active"
_NON_DIGITS = re.compile(r"[^0-9]")
_HOST = r"^(?:[a-z][a-z0-9+.-]*://)?(?:www\.)?([^/:?#]+)"
_HOST_AND_PATH = r"^(?:[a-z][a-z0-9+.-]*://)?(?:www\.)?([^?#]+)"
# Java's String.trim: every character up to U+0020.
_JAVA_TRIM = r"^[\x00-\x20]+|[\x00-\x20]+$"


def register_text_udfs(spark: SparkSession) -> None:
    """Register ``PlaceTextNormalizer`` as the JVM UDFs ``normalize_text`` and ``tokenize_text``."""
    udfs = "org.overturemaps.places.text"
    spark.udf.registerJavaFunction(
        _NORMALIZE_UDF, f"{udfs}.NormalizeTextUdf", StringType()
    )
    spark.udf.registerJavaFunction(
        _TOKENIZE_UDF, f"{udfs}.TokenizeTextUdf", ArrayType(StringType(), False)
    )


def ship_resources(spark: SparkSession, model_jar: str) -> None:
    """Send the model jar and the libpostal data to the workers.

    Args:
        spark: The session. ``spark.sedona.libpostal.dataDir`` must name the
            unpacked v1.0.0 libpostal data.
        model_jar: Path of the ``places-matcher-model`` jar.

    Raises:
        ValueError: The libpostal data dir is not set. Sedona's default
            model would change addresses outside the US.
        FileNotFoundError: A local ``model_jar`` does not exist.
    """
    if "://" not in model_jar and not os.path.isfile(model_jar):
        raise FileNotFoundError(model_jar)
    data_dir = spark.conf.get(LIBPOSTAL_DATA_DIR_CONF, None)
    if not data_dir:
        raise ValueError(f"set {LIBPOSTAL_DATA_DIR_CONF} to the libpostal v1.0.0 data")
    spark.sparkContext.addFile(model_jar)
    # Local data dirs work as-is; only remote data needs shipping.
    if "://" in data_dir and not data_dir.startswith("file:"):
        spark.sparkContext.addFile(data_dir, recursive=True)


def normalize_phone(number: str | None, region: str | None) -> str | None:
    """Digits of the international format of ``number``.

    Args:
        number: A phone number, usually with punctuation.
        region: ISO country code to read a national number in.

    Returns:
        Digits only. An unparseable number keeps its own digits. None for None.
    """
    if number is None:
        return None
    with contextlib.suppress(
        phonenumbers.NumberParseException
    ):  # Unparseable numbers keep their raw digits.
        number = phonenumbers.format_number(
            phonenumbers.parse(number, region),
            phonenumbers.PhoneNumberFormat.INTERNATIONAL,
        )
    return _NON_DIGITS.sub("", number)


def _clean_utf8(col: Column) -> Column:
    """Replace invalid UTF-8 in the JVM, so Python UDFs never receive it."""
    return F.decode(F.encode(col, "UTF-8"), "UTF-8")


def _java_trim(col: Column) -> Column:
    return F.regexp_replace(col, _JAVA_TRIM, "")


def _source(sources: Column) -> Column:
    """Lowercased record ID prefix of the first source without a property."""
    primary = F.filter(
        sources, lambda s: s["property"].isNull() | (s["property"] == "")
    )[0]
    return F.coalesce(
        F.lower(F.split(primary["record_id"], _SIGNATURE_SEPARATOR)[0]), F.lit("")
    )


def _component(components: Column, label: str) -> Column:
    first = F.filter(components, lambda c: c["label"] == label)[0]
    return F.coalesce(first["value"], F.lit(""))


def _freeform() -> Column:
    return _clean_utf8(F.col("addresses")[0]["freeform"])


def _parsed_components() -> Column:
    """Libpostal components of the first address. Projected once as ``_COMPONENTS``."""
    locality = _clean_utf8(F.col("addresses")[0]["locality"])
    query = F.concat_ws(" ", _freeform(), locality)
    return F.when(F.trim(query) != "", ParseAddress(query))


def _address_columns(components: Column) -> tuple[Column, Column]:
    """``house_number`` and ``address`` from the first address and its parsed ``components``."""
    freeform = _freeform()
    house_number = _component(components, "house_number")
    street = _java_trim(
        F.regexp_replace(
            F.regexp_replace(_component(components, "road"), "[,;]", " "), r"\s+", " "
        )
    )
    raw_freeform = F.coalesce(freeform, F.lit(""))
    # A street wins. A parse with a locality but no street gives "" rather than
    # leaking the locality. Otherwise the freeform address minus its house number.
    address = (
        F.when(street != "", street)
        .when(_java_trim(_component(components, "city")) != "", F.lit(""))
        .when(
            house_number != "",
            _java_trim(F.replace(raw_freeform, house_number, F.lit(""))),
        )
        .otherwise(raw_freeform)
    )
    return house_number, address


def _website_columns(df: DataFrame) -> tuple[Column, Column]:
    """Hosts and host-plus-path URLs of ``websites``, lowercased, per website."""
    no_websites = F.array().cast("array<string>")
    websites = (
        F.coalesce(F.col("websites"), no_websites)
        if "websites" in df.columns
        else no_websites
    )
    hosts = F.transform(websites, lambda u: F.regexp_extract(F.lower(u), _HOST, 1))
    urls = F.transform(
        websites,
        lambda u: F.regexp_replace(
            F.regexp_extract(F.lower(u), _HOST_AND_PATH, 1), "/+$", ""
        ),
    )
    return hosts, urls


def _signature(*parts: Column) -> Column:
    """Lowercased parts joined by ``@``. Null parts are skipped."""
    return F.lower(F.concat_ws(_SIGNATURE_SEPARATOR, *parts))


def _matching_columns(
    df: DataFrame, provider: Column, model_jar_name: str
) -> dict[str, Column]:
    """Column expressions for one side."""
    country = _clean_utf8(F.col("addresses")[0]["country"])
    house_number, address = _address_columns(F.col(_COMPONENTS))
    name = F.col("names.primary")
    name_clean = F.call_udf(_NORMALIZE_UDF, name)
    address_clean = F.call_udf(_NORMALIZE_UDF, address)
    name_tokens = F.call_udf(_TOKENIZE_UDF, name_clean)
    lat = F.expr("ST_Y(ST_GeomFromWKB(geometry))")
    lng = F.expr("ST_X(ST_GeomFromWKB(geometry))")
    quadkey = focal_quadkey(lat, lng, BLOCKING_RESOLUTION)
    source = _source(F.col("sources"))
    cell = F.expr("ST_H3CellIDs(ST_GeomFromWKB(geometry), 9, false)[0]")
    stripped_tokens = strip_background_udf(model_jar_name, name_tokens, cell)
    hosts, urls = _website_columns(df)
    return {
        "id": F.col("id"),
        "name": name,
        "address": address,
        "name_clean": name_clean,
        "name_clean_tokens": name_tokens,
        "name_clean_stripped": F.concat_ws(" ", stripped_tokens),
        "name_clean_stripped_tokens": stripped_tokens,
        "address_clean": address_clean,
        "address_clean_tokens": F.call_udf(_TOKENIZE_UDF, address_clean),
        "latitude": lat,
        "longitude": lng,
        "house_number": house_number,
        "country": country,
        "normalized_phone": F.udf(normalize_phone, StringType())(
            _clean_utf8(F.col("phones")[0]), country
        ),
        "blocking_keys": F.array(quadkey),
        "primary_source": source,
        "signature": _signature(
            source, quadkey, name, address, F.col("taxonomy.primary")
        ),
        "provider": provider,
        "taxonomy_primary": F.coalesce(F.col("taxonomy.primary"), F.lit("")),
        "website_domains": F.array_distinct(F.filter(hosts, lambda h: h != "")),
        "website_urls": F.array_distinct(F.filter(urls, lambda u: u.contains("."))),
    }


def _prepare(
    df: DataFrame, provider: Column, model_jar: str, extra: dict[str, Column]
) -> DataFrame:
    columns = _matching_columns(df, provider, os.path.basename(model_jar)) | extra
    # One shuffle by id up front so later joins start co-partitioned.
    # Parsing is expensive, so it is projected once and every field reads that column.
    parsed = df.repartition("id").withColumn(_COMPONENTS, _parsed_components())
    return parsed.select(*[c.alias(n) for n, c in columns.items()])


def prepare_candidates(df: DataFrame, provider: str, model_jar: str) -> DataFrame:
    """Candidates with the columns matching reads, before ``base_ids``.

    Args:
        df: Raw candidate places.
        provider: Stamped on every row.
        model_jar: Path of the ``places-matcher-model`` jar, whose file name
            is where workers find it after ``ship_resources``.
    """
    return _prepare(df, F.lit(provider), model_jar, {})


def baseline_provider_column(df: DataFrame, baseline_provider: str | None) -> Column:
    """The baseline's provider: its ``provider`` column, or ``baseline_provider`` as a literal.

    Raises:
        ValueError: The baseline has a ``provider`` column and
            ``baseline_provider`` is set, or has neither.
    """
    has_provider = "provider" in df.columns
    if has_provider == (baseline_provider is not None):
        raise ValueError(
            "set baselineProvider when the baseline has no provider column, "
            "and only then"
        )
    return F.col("provider") if has_provider else F.lit(baseline_provider)


def prepare_baseline(
    df: DataFrame, model_jar: str, baseline_provider: str | None = None
) -> DataFrame:
    """Baseline with the columns matching reads, plus ``active``.

    Args:
        df: Raw baseline places.
        model_jar: Path of the ``places-matcher-model`` jar.
        baseline_provider: Provider to stamp when the baseline has no
            ``provider`` column.

    Raises:
        ValueError: The baseline has a ``provider`` column and
            ``baseline_provider`` is set, or has neither.
    """
    provider = baseline_provider_column(df, baseline_provider)
    active = (
        F.when(F.col("corpus_state") == _ACTIVE_STATE, F.lit(1)).otherwise(F.lit(0))
        if "corpus_state" in df.columns
        else F.lit(1)
    )
    return _prepare(df, provider, model_jar, {"active": active})


def add_base_ids(
    candidates: DataFrame, baseline: DataFrame, match_history: bool
) -> DataFrame:
    """Add ``base_ids``: the sorted baseline IDs with the candidate's provider and signature.

    Args:
        candidates: Output of ``prepare_candidates``.
        baseline: Output of ``prepare_baseline``.
        match_history: When False, ``base_ids`` is null on every row.

    Returns:
        ``candidates`` plus ``base_ids``, null where nothing matched. Retired
        baseline records count; ``active`` only matters later.
    """
    if not match_history:
        return candidates.withColumn("base_ids", F.lit(None).cast("array<string>"))
    ids = F.array_sort(F.collect_list("id"))
    history = baseline.groupBy("provider", "signature").agg(
        F.when(F.size(ids) > 0, ids).alias("base_ids")
    )
    c, h = candidates.alias("c"), history.alias("h")
    return c.join(
        h,
        (F.col("c.provider") == F.col("h.provider"))
        & (F.col("c.signature") == F.col("h.signature")),
        "left",
    ).select("c.*", "h.base_ids")
