# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Places setup steps: the columns matching reads, and the history join.

Each side's columns come from one ``select``. Name normalization, tokenizing
and background-token stripping come in as ``TextFunctions``, so the matching
rules here run without the ``places-matcher-model`` jar. Python UDFs live here,
not in the job script, so that workers can unpickle them.
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

from overture_matchers.quadkey import get_focal_quadkey

# Default resolution of the signature's quadkey and the blocking keys.
BLOCKING_RESOLUTION = 16
_PARSED_ADDRESS = "_parsed_address"
_GEOMETRY = "_geometry"
_QUADKEY = "_quadkey"
_SIGNATURE_SEPARATOR = "@"
_ACTIVE_STATE = "active"
_NON_DIGITS = re.compile(r"[^0-9]")
_HOST = r"^(?:[a-z][a-z0-9+.-]*://)?(?:www\.)?([^/:?#]+)"
_HOST_AND_PATH = r"^(?:[a-z][a-z0-9+.-]*://)?(?:www\.)?([^?#]+)"
# Java's String.trim: every character up to U+0020. The Scala setup trims with
# String.trim, so this matches it for parity (Spark's trim only strips spaces).
_JAVA_TRIM = r"^[\x00-\x20]+|[\x00-\x20]+$"


class TextFunctions(NamedTuple):
    """The text functions that build the columns for matching, each over Columns."""

    normalize_text: Callable[[Column], Column]
    tokenize_text: Callable[[Column], Column]
    # Takes the name tokens, latitude and longitude.
    strip_background_tokens: Callable[[Column, Column, Column], Column]


def normalize_phone_number(
    phone_number: str | None, country_code: str | None
) -> str | None:
    """Digits of the international format of ``phone_number``.

    Args:
        phone_number: A phone number, usually with punctuation.
        country_code: ISO country code to read a national number in.

    Returns:
        Digits only. An unparseable number keeps its own digits. None for None.
    """
    if phone_number is None:
        return None
    with contextlib.suppress(
        phonenumbers.NumberParseException
    ):  # Unparseable numbers keep their raw digits.
        phone_number = phonenumbers.format_number(
            phonenumbers.parse(phone_number, country_code),
            phonenumbers.PhoneNumberFormat.INTERNATIONAL,
        )
    return _NON_DIGITS.sub("", phone_number)


def _replace_invalid_utf8(col: Column) -> Column:
    """Replace invalid UTF-8 in the JVM, so Python UDFs never receive it."""
    return F.decode(F.encode(col, "UTF-8"), "UTF-8")


def _java_trim(col: Column) -> Column:
    return F.regexp_replace(col, _JAVA_TRIM, "")


def _get_primary_source(sources: Column) -> Column:
    """Lowercased record ID prefix of the first source without a property."""
    primary = F.filter(
        sources, lambda s: s["property"].isNull() | (s["property"] == "")
    )[0]
    return F.coalesce(
        F.lower(F.split(primary["record_id"], _SIGNATURE_SEPARATOR)[0]), F.lit("")
    )


def _get_address_component(components: Column, label: str) -> Column:
    first = F.filter(components, lambda c: c["label"] == label)[0]
    return F.coalesce(first["value"], F.lit(""))


def _get_first_address_freeform() -> Column:
    return _replace_invalid_utf8(F.col("addresses")[0]["freeform"])


def _parse_first_address() -> Column:
    """Libpostal components of the first address. Projected once as ``_PARSED_ADDRESS``."""
    locality = _replace_invalid_utf8(F.col("addresses")[0]["locality"])
    query = F.concat_ws(" ", _get_first_address_freeform(), locality)
    return F.when(F.trim(query) != "", ParseAddress(query))


def _normalize_address(parsed_address: Column, freeform: Column) -> Column:
    """Street-only ``address`` from ``parsed_address`` components and raw ``freeform``."""
    house_number = _get_address_component(parsed_address, "house_number")
    street = _java_trim(
        F.regexp_replace(
            F.regexp_replace(
                _get_address_component(parsed_address, "road"), "[,;]", " "
            ),
            r"\s+",
            " ",
        )
    )
    raw_freeform = F.coalesce(freeform, F.lit(""))
    # Pick the first rule that applies:
    # 1. The parser found a street: use it.
    # 2. It found a city but no street: use "" so the city does not end up in the
    #    address field.
    # 3. It found a house number: use the raw address with that number removed.
    # 4. Otherwise use the raw address as given.
    return (
        F.when(street != "", street)
        .when(
            _java_trim(_get_address_component(parsed_address, "city")) != "", F.lit("")
        )
        .when(
            house_number != "",
            _java_trim(F.replace(raw_freeform, house_number, F.lit(""))),
        )
        .otherwise(raw_freeform)
    )


def _extract_website_domains_and_urls(df: DataFrame) -> tuple[Column, Column]:
    """Distinct lowercased hosts and ``host/path`` URLs from ``websites``."""
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
    website_domains = F.array_distinct(F.filter(hosts, lambda h: h != ""))
    website_urls = F.array_distinct(F.filter(urls, lambda u: u.contains(".")))
    return website_domains, website_urls


def _build_signature(*parts: Column) -> Column:
    """Lowercased parts joined by ``@``. Null parts are skipped."""
    return F.lower(F.concat_ws(_SIGNATURE_SEPARATOR, *parts))


def _stage_reused_columns(
    df: DataFrame,
    parsed_address: Column,
    blocking_resolution: int = BLOCKING_RESOLUTION,
) -> DataFrame:
    """Stage columns read by multiple outputs so Spark evaluates each one once."""
    return (
        df.withColumn(_PARSED_ADDRESS, parsed_address)
        .withColumn(_GEOMETRY, F.expr("ST_GeomFromWKB(geometry)"))
        .withColumn("latitude", F.expr(f"ST_Y({_GEOMETRY})"))
        .withColumn("longitude", F.expr(f"ST_X({_GEOMETRY})"))
        .withColumn(
            _QUADKEY,
            get_focal_quadkey(
                F.col("latitude"), F.col("longitude"), blocking_resolution
            ),
        )
    )


def _prepare_columns_for_matching(
    df: DataFrame, provider: Column, text_functions: TextFunctions
) -> dict[str, Column]:
    """Column expressions for matching on one side, over ``_stage_reused_columns``."""
    country = _replace_invalid_utf8(F.col("addresses")[0]["country"])
    parsed_address = F.col(_PARSED_ADDRESS)
    house_number = _get_address_component(parsed_address, "house_number")
    address = _normalize_address(parsed_address, _get_first_address_freeform())
    name = F.col("names.primary")
    name_clean = text_functions.normalize_text(name)
    address_clean = text_functions.normalize_text(address)
    name_clean_tokens = text_functions.tokenize_text(name_clean)
    latitude = F.col("latitude")
    longitude = F.col("longitude")
    quadkey = F.col(_QUADKEY)
    primary_source = _get_primary_source(F.col("sources"))
    name_clean_stripped_tokens = text_functions.strip_background_tokens(
        name_clean_tokens, latitude, longitude
    )
    website_domains, website_urls = _extract_website_domains_and_urls(df)
    return {
        "id": F.col("id"),
        "name": name,
        "address": address,
        "name_clean": name_clean,
        "name_clean_tokens": name_clean_tokens,
        "name_clean_stripped": F.concat_ws(" ", name_clean_stripped_tokens),
        "name_clean_stripped_tokens": name_clean_stripped_tokens,
        "address_clean": address_clean,
        "address_clean_tokens": text_functions.tokenize_text(address_clean),
        "latitude": latitude,
        "longitude": longitude,
        "house_number": house_number,
        "country": country,
        "normalized_phone": F.udf(normalize_phone_number, StringType())(
            _replace_invalid_utf8(F.col("phones")[0]), country
        ),
        "blocking_keys": F.array(quadkey),
        "primary_source": primary_source,
        "signature": _build_signature(
            primary_source, quadkey, name, address, F.col("taxonomy.primary")
        ),
        "provider": provider,
        "taxonomy_primary": F.coalesce(F.col("taxonomy.primary"), F.lit("")),
        "website_domains": website_domains,
        "website_urls": website_urls,
    }


def _select_columns_for_matching(
    df: DataFrame,
    provider: Column,
    text_functions: TextFunctions,
    extra_columns: dict[str, Column],
    blocking_resolution: int = BLOCKING_RESOLUTION,
) -> DataFrame:
    """Project ``df`` to the columns for matching plus ``extra_columns``."""
    staged = _stage_reused_columns(df, _parse_first_address(), blocking_resolution)
    columns = (
        _prepare_columns_for_matching(staged, provider, text_functions) | extra_columns
    )
    return staged.select(*[c.alias(n) for n, c in columns.items()])


def prepare_candidates(
    df: DataFrame,
    provider: str,
    text_functions: TextFunctions,
    blocking_resolution: int = BLOCKING_RESOLUTION,
) -> DataFrame:
    """Candidates with the columns matching reads, before ``base_ids``.

    Args:
        df: Raw candidate places.
        provider: Stamped on every row.
        text_functions: The text functions.
        blocking_resolution: Quadkey zoom for ``blocking_keys`` and ``signature``.
    """
    return _select_columns_for_matching(
        df, F.lit(provider), text_functions, {}, blocking_resolution
    )


def get_baseline_provider_column(
    df: DataFrame, baseline_provider: str | None
) -> Column:
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
    df: DataFrame,
    provider: Column,
    text_functions: TextFunctions,
    blocking_resolution: int = BLOCKING_RESOLUTION,
) -> DataFrame:
    """Baseline with the columns matching reads, plus ``active``.

    Args:
        df: Raw baseline places.
        provider: The baseline's provider, from ``get_baseline_provider_column``.
        text_functions: The text functions.
        blocking_resolution: Quadkey zoom for ``blocking_keys`` and ``signature``.
    """
    active = (
        F.when(F.col("corpus_state") == _ACTIVE_STATE, F.lit(1)).otherwise(F.lit(0))
        if "corpus_state" in df.columns
        else F.lit(1)
    )
    return _select_columns_for_matching(
        df, provider, text_functions, {"active": active}, blocking_resolution
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
