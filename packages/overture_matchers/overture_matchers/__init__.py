# SPDX-License-Identifier: MIT
# Copyright (c) 2026 by the Overture Maps Foundation
"""Matchers that conflate Overture Maps data with incoming sources.

Each theme (places, divisions, addresses) matches candidate records against
the Overture baseline and assigns GERS IDs. The code runs as PySpark jobs.
"""
