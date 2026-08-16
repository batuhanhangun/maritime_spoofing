"""Geodesic helpers shared across the pipeline.

All functions are NaN-safe: invalid inputs propagate NaN instead of raising.
"""

import math

EARTH_RADIUS_M = 6371000.0
MPS_TO_KNOTS = 1.94384


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance between two WGS84 points, in meters."""
    if any(v is None or math.isnan(v) for v in (lat1, lon1, lat2, lon2)):
        return float('nan')
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2.0) ** 2)
    return EARTH_RADIUS_M * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing from point 1 to point 2, degrees [0, 360).

    This replaces the old ``atan2(dlon, dlat)`` approximation, which ignored
    the cos(latitude) compression of longitude and biased derived headings on
    any course that is not due north/south. The full formula is exact for
    arbitrary courses and latitudes, which matters once real vessel logs
    (arbitrary maneuvering) are processed.
    """
    if any(v is None or math.isnan(v) for v in (lat1, lon1, lat2, lon2)):
        return float('nan')
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(phi2)
    y = (math.cos(phi1) * math.sin(phi2)
         - math.sin(phi1) * math.cos(phi2) * math.cos(dlon))
    return math.degrees(math.atan2(x, y)) % 360.0


def angular_diff_deg(a, b):
    """Smallest absolute angular difference in degrees, handling wraparound."""
    if a is None or b is None:
        return float('nan')
    if isinstance(a, float) and math.isnan(a):
        return float('nan')
    if isinstance(b, float) and math.isnan(b):
        return float('nan')
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)
