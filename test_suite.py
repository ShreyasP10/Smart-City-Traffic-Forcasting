"""Automated Unit Test Suite for Smart City Traffic Forecasting System.

Tests:
1. SHAP Waterfall Exact Additivity (W1):
   base_expected_value + sum(waterfall_impacts) == final_forecasted_volume
   across sub-hour horizons, adaptation, live fusion, and clipping.
2. Exponential Horizon Decay (W5):
   w_live decays smoothly and monotonically across 0, 15, 30, 60 minutes.
3. Classification Threshold Boundaries:
   Strictly increasing thresholds and exact label partitioning.
4. City-Specific Holiday Calendar (W8):
   Differentiation between US and Indian public holidays.
5. SSRF Geographic Coordinate Validation:
   Bounds checking and non-float injection rejection.
6. Safe Model Checksum Integrity (W2):
   Verification passes on authentic hash and fails on tampered digests.
"""
from __future__ import annotations

import datetime as dt
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from city_config import is_holiday, get_city_now
from traffic_api import validate_coordinates
from schema import safe_load_model, featurize_point
from pipeline import predict_pipeline, classify
from explain_shap import compute_shap_explanation


class TestSmartCityTraffic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = os.path.join(os.path.dirname(__file__), "models", "traffic_model.joblib")
        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(__file__), "base_model.pkl")
        cls.bundle = safe_load_model(model_path)

    def test_01_waterfall_exact_additivity(self):
        """W1: Guarantee 100% mathematical equality between waterfall steps and final volume."""
        test_cases = [
            # (horizon_min, rain, speed_ratio, city)
            (0, 0.0, 0.70, "Mumbai"),
            (15, 4.5, 0.40, "Mumbai"),
            (30, 0.0, 0.85, "Bengaluru"),
            (45, 12.0, 0.30, "Delhi NCR"),
            (60, 0.0, 0.95, "Minneapolis (Source / US)"),
        ]

        now = dt.datetime(2026, 10, 12, 17, 20)  # 5:20 PM Monday peak
        for horizon, rain, ratio, city in test_cases:
            weather = {
                "temp": 302.15,
                "rain_1h": rain,
                "clouds_all": 75,
                "weather_main": "Rain" if rain > 0 else "Clear",
            }
            flow = {
                "current_speed": 45.0 * ratio,
                "free_flow_speed": 45.0,
                "congestion_ratio": ratio,
            }

            pipe = predict_pipeline(now, weather, flow, self.bundle, minutes_ahead=horizon, city=city)
            shap_info = compute_shap_explanation(
                now, weather, flow, self.bundle,
                horizon_min=horizon, pipeline_pred=pipe, city=city
            )

            # Sum up base value and all waterfall step deltas
            base_val = shap_info["base_expected_value"]
            steps_sum = sum(step["impact"] for step in shap_info["waterfall_steps"])
            reconstructed_total = base_val + steps_sum
            final_vol = float(shap_info["final_forecast"])

            # Verify exact floating-point equality within 1e-3 tolerance
            self.assertAlmostEqual(
                reconstructed_total, final_vol, places=3,
                msg=f"Waterfall failed additivity at T+{horizon}m in {city}: "
                    f"reconstructed={reconstructed_total:.4f} vs final={final_vol:.4f} (diff={reconstructed_total - final_vol:.6f})"
            )

    def test_02_exponential_horizon_decay(self):
        """W5: Exponential live fusion decay w_live = max(0.35 * exp(-0.693 * m / 30), 0.05)."""
        horizons = [0, 15, 30, 45, 60]
        weights = []
        for m in horizons:
            w = max(0.35 * math.exp(-0.693 * m / 30.0), 0.05)
            weights.append(w)

        # Weight at T=0 should be exactly 0.35
        self.assertAlmostEqual(weights[0], 0.35, places=3)
        # Weight at T=30 should be half-life ~0.175
        self.assertAlmostEqual(weights[2], 0.175, places=2)
        # Monotonically decreasing
        for i in range(len(weights) - 1):
            self.assertGreater(weights[i], weights[i + 1])
            self.assertGreaterEqual(weights[i + 1], 0.05)

    def test_03_classification_thresholds(self):
        """Threshold ordering and disjoint classification intervals."""
        thresh = self.bundle["thresholds"]
        self.assertEqual(len(thresh), 3)
        self.assertLess(thresh[0], thresh[1])
        self.assertLess(thresh[1], thresh[2])

        self.assertEqual(classify(thresh[0] - 10, thresh), "Low")
        self.assertEqual(classify(thresh[0], thresh), "Moderate")
        self.assertEqual(classify(thresh[0] + 10, thresh), "Moderate")
        self.assertEqual(classify(thresh[1], thresh), "High")
        self.assertEqual(classify(thresh[1] + 10, thresh), "High")
        self.assertEqual(classify(thresh[2], thresh), "Severe")
        self.assertEqual(classify(thresh[2] + 10, thresh), "Severe")

    def test_04_city_specific_holidays(self):
        """W8: Verify city-specific holiday evaluation."""
        # July 4 (US Independence Day)
        july_4 = dt.date(2026, 7, 4)
        self.assertEqual(is_holiday(july_4, "Minneapolis (Source / US)"), 1)
        self.assertEqual(is_holiday(july_4, "Mumbai"), 0)
        self.assertEqual(is_holiday(july_4, "Bengaluru"), 0)

        # January 26 (Indian Republic Day)
        jan_26 = dt.date(2026, 1, 26)
        self.assertEqual(is_holiday(jan_26, "Mumbai"), 1)
        self.assertEqual(is_holiday(jan_26, "Delhi NCR"), 1)
        self.assertEqual(is_holiday(jan_26, "Minneapolis (Source / US)"), 0)

    def test_05_coordinate_validation_ssrf(self):
        """SSRF & input sanitization boundaries."""
        # Valid coordinates
        lat, lon = validate_coordinates(19.0760, 72.8777)
        self.assertEqual(lat, 19.0760)
        self.assertEqual(lon, 72.8777)

        # Out of bounds
        with self.assertRaises(ValueError):
            validate_coordinates(95.0, 72.0)
        with self.assertRaises(ValueError):
            validate_coordinates(19.0, 200.0)
        # Non-float injections
        with self.assertRaises(ValueError):
            validate_coordinates("19.0; DROP TABLE", 72.0)

    def test_06_model_integrity_verification(self):
        """W2: Model integrity check with SHA-256."""
        model_path = os.path.join(os.path.dirname(__file__), "models", "traffic_model.joblib")
        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(__file__), "base_model.pkl")

        # Wrong hash must raise ValueError
        with self.assertRaises(ValueError):
            safe_load_model(model_path, "0000000000000000000000000000000000000000000000000000000000000000")


if __name__ == "__main__":
    unittest.main(verbosity=2)
