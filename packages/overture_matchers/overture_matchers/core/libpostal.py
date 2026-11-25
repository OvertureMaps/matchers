# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Ship libpostal model data to Spark workers for Sedona's ``ParseAddress``.

Any theme that parses addresses calls this once per session, before parsing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse

from overture_matchers.core.config import rewrite_s3_to_s3a

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

LIBPOSTAL_DATA_DIR_CONF = "spark.sedona.libpostal.dataDir"


def ship_libpostal_data_to_workers(spark: SparkSession) -> None:
    """Send remote libpostal data to the workers.

    Args:
        spark: The session. ``spark.sedona.libpostal.dataDir`` must name the
            unpacked v1.0.0 libpostal data.

    Raises:
        ValueError: The libpostal data dir is not set. Sedona's default
            model would change addresses outside the US.
    """
    data_dir = spark.conf.get(LIBPOSTAL_DATA_DIR_CONF, None)
    if not data_dir:
        raise ValueError(f"set {LIBPOSTAL_DATA_DIR_CONF} to the libpostal v1.0.0 data")
    # Local data dirs work as-is. Sedona 1.9+ finds remote data among the
    # shipped files by its folder name.
    if urlparse(data_dir).scheme not in ("", "file"):
        spark.sparkContext.addFile(rewrite_s3_to_s3a(data_dir), recursive=True)
