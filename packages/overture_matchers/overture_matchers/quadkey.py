# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Web Mercator quadkeys as native Spark expressions."""

from __future__ import annotations

from pyspark.sql import Column
from pyspark.sql import functions as F

_MAX_LATITUDE = 85.05112878
_TILE_SIZE = 256


def focal_quadkey(lat: Column, lng: Column, resolution: int) -> Column:
    """Quadkey of the tile at ``resolution`` containing the given point.

    Args:
        lat: Latitude in degrees.
        lng: Longitude in degrees.
        resolution: Tile zoom, from 1 to 23.

    Returns:
        A string column with one digit per zoom level. It is null when either
        coordinate is null.
    """
    grid_size = _TILE_SIZE << resolution
    clamped_lat = F.least(F.greatest(lat, F.lit(-_MAX_LATITUDE)), F.lit(_MAX_LATITUDE))
    sin_lat = F.sin(F.radians(clamped_lat))
    x_grid = (F.least(F.greatest(lng, F.lit(-180.0)), F.lit(180.0)) + 180) / 360
    y_grid = 0.5 - F.log((1 + sin_lat) / (1 - sin_lat)) / (4 * F.lit(3.141592653589793))

    def cell(grid: Column) -> Column:
        pixel = F.least(
            F.greatest(grid * grid_size + 0.5, F.lit(0.0)), F.lit(grid_size - 1.0)
        )
        # The pixel is non-negative, so the casts truncate like integer division.
        return (pixel.cast("int") / _TILE_SIZE).cast("int")

    x_cell, y_cell = cell(x_grid), cell(y_grid)
    return F.when(
        lat.isNotNull() & lng.isNotNull(),
        F.concat(
            *[
                (
                    F.shiftright(x_cell, bit).bitwiseAND(1)
                    + 2 * F.shiftright(y_cell, bit).bitwiseAND(1)
                ).cast("string")
                for bit in range(resolution - 1, -1, -1)
            ]
        ),
    )
