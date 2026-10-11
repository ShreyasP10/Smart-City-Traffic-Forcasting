"""Quick TomTom key check. Usage:  python test_tomtom.py <YOUR_KEY_VALUE>"""
from __future__ import annotations
import sys
from traffic_api import fetch_tomtom_flow, fetch_tomtom_route, fetch_tomtom_incidents

key = sys.argv[1].strip() if len(sys.argv) > 1 else ""
if not key:
    print("Usage: python test_tomtom.py <KEY_VALUE>")
    print("Copy the KEY VALUE (eye/copy icon), not the Key ID eceb3134-…")
    raise SystemExit(1)
if "-" in key and len(key) < 30:
    print("That looks like the Key ID, not the key value. Open TomTom Developer > My Apps > "
          "My First API key > show/copy the key value (~30+ chars, no dashes).")
    raise SystemExit(1)

print("1) Flow Segment (Mumbai Andheri):")
print("  ", fetch_tomtom_flow(19.1136, 72.8697, 50, key, 0.5))
print("2) Routing with traffic (Andheri -> BKC):")
print("  ", fetch_tomtom_route((19.1136, 72.8697), (19.0596, 72.8656), key))
print("3) Incidents near BKC:")
inc = fetch_tomtom_incidents(19.0596, 72.8656, key)
print(f"   count={inc['count']} source={inc['source']}")
print("If sources say live:*, your key works — paste it in the Streamlit sidebar.")
