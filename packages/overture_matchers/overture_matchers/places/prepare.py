# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Places setup steps: the columns matching reads, and the history join.

Each side's columns come from one ``select``. Name normalization, tokenizing
and background-token stripping come in as ``TextFns``, so the matching rules
here run without the ``places-matcher-model`` jar. Python UDFs live here, not
in the job script, so that workers can unpickle them.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Callable
from typing import NamedTuple

import phonenumbers
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from sedona.spark.sql.st_functions import ParseAddress

from overture_matchers.quadkey import focal_quadkey

# Resolution of the signature's quadkey and the blocking keys.
BLOCKING_RESOLUTION = 16
_COMPONENTS = "_components"
_SIGNATURE_SEPARATOR = "@"
_ACTIVE_STATE = "active"
_NON_DIGITS = re.compile(r"[^0-9]")
_HOST = r"^(?:[a-z][a-z0-9+.-]*://)?(?:www\.)?([^/:?#]+)"
_HOST_AND_PATH = r"^(?:[a-z][a-z0-9+.-]*://)?(?:www\.)?([^?#]+)"
# Java's String.trim: every character up to U+0020. The Scala setup trims with
# String.trim, so this matches it for parity (Spark's trim only strips spaces).
_JAVA_TRIM = r"^[\x00-\x20]+|[\x00-\x20]+$"


class TextFns(NamedTuple):
    """The text functions the matching columns use, each over Columns."""

    normalize: Callable[[Column], Column]
    tokenize: Callable[[Column], Column]
    # Takes the name tokens and the H3 resolution 9 cell.
    strip_background: Callable[[Column, Column], Column]


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


def _address_columns(freeform: Column, components: Column) -> tuple[Column, Column]:
    """``house_number`` and ``address`` from a ``freeform`` address and its parsed ``components``."""
    house_number = _component(components, "house_number")
    street = _java_trim(
        F.regexp_replace(
            F.regexp_replace(_component(components, "road"), "[,;]", " "), r"\s+", " "
        )
    )
    raw_freeform = F.coalesce(freeform, F.lit(""))
    # Pick the first rule that applies:
    # 1. The parser found a street: use it.
    # 2. It found a city but no street: use "" so the city does not end up in the
    #    address field.
    # 3. It found a house number: use the raw address with that number removed.
    # 4. Otherwise use the raw address as given.
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
    df: DataFrame, provider: Column, text: TextFns
) -> dict[str, Column]:
    """Column expressions for one side. ``df`` must carry the parsed ``_COMPONENTS``."""
    country = _clean_utf8(F.col("addresses")[0]["country"])
    house_number, address = _address_columns(_freeform(), F.col(_COMPONENTS))
    name = F.col("names.primary")
    name_clean = text.normalize(name)
    address_clean = text.normalize(address)
    name_tokens = text.tokenize(name_clean)
    lat = F.expr("ST_Y(ST_GeomFromWKB(geometry))")
    lng = F.expr("ST_X(ST_GeomFromWKB(geometry))")
    quadkey = focal_quadkey(lat, lng, BLOCKING_RESOLUTION)
    source = _source(F.col("sources"))
    cell = F.expr("ST_H3CellIDs(ST_GeomFromWKB(geometry), 9, false)[0]")
    stripped_tokens = text.strip_background(name_tokens, cell)
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
        "address_clean_tokens": text.tokenize(address_clean),
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


def _select_matching_columns(
    df: DataFrame, provider: Column, text: TextFns, extra_columns: dict[str, Column]
) -> DataFrame:
    """Project ``df`` to the matching columns plus ``extra_columns``."""
    columns = _matching_columns(df, provider, text) | extra_columns
    # Parsing is expensive, so it is projected once and every field reads that column.
    parsed = df.withColumn(_COMPONENTS, _parsed_components())
    return parsed.select(*[c.alias(n) for n, c in columns.items()])


def prepare_candidates(df: DataFrame, provider: str, text: TextFns) -> DataFrame:
    """Candidates with the columns matching reads, before ``base_ids``.

    Args:
        df: Raw candidate places.
        provider: Stamped on every row.
        text: The text functions.
    """
    # One shuffle by id up front so later joins start co-partitioned.
    return _select_matching_columns(df.repartition("id"), F.lit(provider), text, {})


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


def prepare_baseline(df: DataFrame, provider: Column, text: TextFns) -> DataFrame:
    """Baseline with the columns matching reads, plus ``active``.

    Args:
        df: Raw baseline places.
        provider: The baseline's provider, from ``baseline_provider_column``.
        text: The text functions.
    """
    active = (
        F.when(F.col("corpus_state") == _ACTIVE_STATE, F.lit(1)).otherwise(F.lit(0))
        if "corpus_state" in df.columns
        else F.lit(1)
    )
    return _select_matching_columns(
        df.repartition("id"), provider, text, {"active": active}
    )


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
