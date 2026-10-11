"""City / route presets for Indian context + US source context.

Includes timezone mapping and canonical holiday rules.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

CITY_TZ: dict[str, str] = {
    "Mumbai": "Asia/Kolkata",
    "Bengaluru": "Asia/Kolkata",
    "Delhi NCR": "Asia/Kolkata",
    "Pune": "Asia/Kolkata",
    "Minneapolis (Source / US)": "America/Chicago",
}

CITIES: dict[str, dict] = {
    "Mumbai": {
        "center": (19.0760, 72.8777),
        "routes": [
            {"name": "Andheri → BKC", "lat": 19.1136, "lon": 72.8697, "free_flow": 50, "length_km": 8.5},
            {"name": "Dadar → Sion", "lat": 19.0176, "lon": 72.8562, "free_flow": 45, "length_km": 6.2},
            {"name": "Bandra-Worli Sea Link", "lat": 19.0350, "lon": 72.8170, "free_flow": 80, "length_km": 5.6},
            {"name": "Thane → Mulund (Eastern Expy)", "lat": 19.1726, "lon": 72.9560, "free_flow": 60, "length_km": 9.1},
            {"name": "Colaba → Nariman Point", "lat": 18.9220, "lon": 72.8347, "free_flow": 40, "length_km": 3.4},
        ],
    },
    "Bengaluru": {
        "center": (12.9716, 77.5946),
        "routes": [
            {"name": "Silk Board → Whitefield", "lat": 12.9177, "lon": 77.6226, "free_flow": 45, "length_km": 12.4},
            {"name": "Hebbal → Majestic", "lat": 13.0358, "lon": 77.5970, "free_flow": 50, "length_km": 7.8},
            {"name": "Electronic City → Koramangala", "lat": 12.8452, "lon": 77.6602, "free_flow": 55, "length_km": 10.2},
            {"name": "Airport Rd (KR Puram)", "lat": 13.0050, "lon": 77.6950, "free_flow": 60, "length_km": 6.9},
        ],
    },
    "Delhi NCR": {
        "center": (28.6139, 77.2090),
        "routes": [
            {"name": "Connaught Place → Gurgaon", "lat": 28.5355, "lon": 77.0830, "free_flow": 60, "length_km": 14.5},
            {"name": "Noida → Akshardham", "lat": 28.6127, "lon": 77.2773, "free_flow": 55, "length_km": 8.0},
            {"name": "Rohini → Karol Bagh", "lat": 28.6519, "lon": 77.1907, "free_flow": 45, "length_km": 9.3},
            {"name": "IGI Airport → Dhaula Kuan", "lat": 28.5562, "lon": 77.1000, "free_flow": 60, "length_km": 7.1},
        ],
    },
    "Pune": {
        "center": (18.5204, 73.8567),
        "routes": [
            {"name": "Hinjewadi → Shivajinagar", "lat": 18.5679, "lon": 73.7700, "free_flow": 50, "length_km": 13.8},
            {"name": "Kharadi → Magarpatta", "lat": 18.5200, "lon": 73.9300, "free_flow": 45, "length_km": 5.5},
            {"name": "Swargate → Katraj", "lat": 18.4575, "lon": 73.8678, "free_flow": 45, "length_km": 4.9},
        ],
    },
    "Minneapolis (Source / US)": {
        "center": (44.9778, -93.2650),
        "routes": [
            {"name": "I-94 Westbound (Source)", "lat": 44.9778, "lon": -93.2650, "free_flow": 100, "length_km": 10.0},
        ],
    },
}

INDIAN_HOLIDAYS = {
    (1, 26),   # Republic Day
    (5, 1),    # Labour Day / Maharashtra Day
    (8, 15),   # Independence Day
    (10, 2),   # Gandhi Jayanti
    (11, 1),   # Diwali approx
    (12, 25),  # Christmas
}

US_HOLIDAYS = {
    (1, 1), (5, 30), (7, 4), (9, 5), (11, 24), (12, 25),
}


def is_holiday(d: dt.date | dt.datetime, city: str = "Mumbai") -> int:
    """Returns 1 if date is a holiday in given city, else 0."""
    key = (d.month, d.day)
    holidays = US_HOLIDAYS if "Minneapolis" in city or "US" in city else INDIAN_HOLIDAYS
    return int(key in holidays)


def get_city_now(city: str = "Mumbai") -> dt.datetime:
    """Returns local time for specified city, preventing UTC displacement on cloud hosts."""
    tz_str = CITY_TZ.get(city, "Asia/Kolkata")
    return dt.datetime.now(ZoneInfo(tz_str)).replace(second=0, microsecond=0, tzinfo=None)
