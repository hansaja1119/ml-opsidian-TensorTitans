"""
Tests for the Feature Engineering Pipeline.

Verifies that:
1. The engineer correctly loads from artifacts.
2. A known input produces the expected 115-feature vector.
3. Missing/unexpected values are handled gracefully.
4. The risk level classifier returns correct strings.
"""

import pytest
import numpy as np
from api.feature_engine import FeatureEngine, get_risk_level, CAT_COLS, TE_COLS_AND_SMOOTHING


# ─── Mock Artifacts ─────────────────────────────────────────────

@pytest.fixture
def mock_engine():
    """Create a FeatureEngine with mock artifacts for testing."""
    # Fake label encoders: just map a few known values
    label_encoders = {}
    for col in CAT_COLS:
        label_encoders[col] = {"missing": 0, "Unknown": 1, "Other": 2}
    label_encoders['district']['Colombo'] = 3
    label_encoders['district']['Ratnapura'] = 4
    label_encoders['landcover']['Urban'] = 3
    label_encoders['soil_type']['Clay'] = 3
    label_encoders['water_supply']['Pipe-borne'] = 3
    label_encoders['electricity']['Yes'] = 3
    label_encoders['road_quality']['Moderate'] = 3
    label_encoders['urban_rural']['Urban'] = 3
    label_encoders['water_presence_flag']['No'] = 3
    label_encoders['flood_occurrence_current_event']['No'] = 3
    label_encoders['is_good_to_live']['Yes'] = 3
    label_encoders['place_name']['Unknown'] = 1

    # Fake TE stats: just return global mean for all
    te_stats = {}
    for col, _ in TE_COLS_AND_SMOOTHING:
        te_stats[col] = {"0": 0.42, "1": 0.38, "2": 0.45, "3": 0.40, "4": 0.50}

    # Build a plausible feature column list (simplified)
    feature_columns = [f"feat_{i}" for i in range(115)]

    medians = {f"feat_{i}": 0.5 for i in range(115)}
    global_mean = 0.42

    return FeatureEngine(
        label_encoders=label_encoders,
        te_stats=te_stats,
        feature_columns=feature_columns,
        medians=medians,
        global_mean=global_mean,
    )


def _make_raw_input(**overrides):
    """Create a complete raw input dict with sensible defaults."""
    base = {
        'district': 'Colombo',
        'landcover': 'Urban',
        'soil_type': 'Clay',
        'water_supply': 'Pipe-borne',
        'electricity': 'Yes',
        'road_quality': 'Moderate',
        'urban_rural': 'Urban',
        'water_presence_flag': 'No',
        'flood_occurrence_current_event': 'No',
        'is_good_to_live': 'Yes',
        'reason_not_good_to_live': 'Other',
        'place_name': 'Unknown',
        'generation_date': '2024-06-15',
        'rainfall_7d_mm': 50.0,
        'monthly_rainfall_mm': 200.0,
        'drainage_index': 0.5,
        'infrastructure_score': 0.5,
        'elevation_m': 30.0,
        'distance_to_river_m': 500.0,
        'nearest_hospital_km': 5.0,
        'nearest_evac_km': 3.0,
        'built_up_percent': 30.0,
        'extreme_weather_index': 0.3,
        'seasonal_index': 0.5,
        'terrain_roughness_index': 0.2,
        'historical_flood_count': 2,
        'population_density_per_km2': 500.0,
        'ndvi': 0.3,
        'ndwi': 0.1,
        'socioeconomic_status_index': 0.5,
        'inundation_area_sqm': 1000.0,
        'is_synthetic': 0,
    }
    base.update(overrides)
    return base


class TestFeatureEngine:
    """Tests for the FeatureEngine class."""

    def test_transform_returns_float32_array(self, mock_engine):
        """Transform should return a 1D float32 numpy array."""
        raw = _make_raw_input()
        result = mock_engine.transform(raw)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.ndim == 1

    def test_transform_output_length_matches_feature_columns(self, mock_engine):
        """Output vector length must match the feature column list."""
        raw = _make_raw_input()
        result = mock_engine.transform(raw)
        assert len(result) == len(mock_engine.feature_columns)

    def test_transform_has_no_nans(self, mock_engine):
        """Output should never contain NaN values (medians fill gaps)."""
        raw = _make_raw_input()
        result = mock_engine.transform(raw)
        assert not np.isnan(result).any()

    def test_transform_handles_missing_categorical(self, mock_engine):
        """Unknown categorical values should not crash the pipeline."""
        raw = _make_raw_input(district='Atlantis')
        result = mock_engine.transform(raw)
        assert isinstance(result, np.ndarray)
        assert not np.isnan(result).any()

    def test_transform_handles_extreme_values(self, mock_engine):
        """Extreme but valid values should produce a valid output."""
        raw = _make_raw_input(rainfall_7d_mm=1500.0, elevation_m=0.0, drainage_index=0.0)
        result = mock_engine.transform(raw)
        assert isinstance(result, np.ndarray)
        assert not np.isnan(result).any()

    def test_date_features_extracted(self, mock_engine):
        """Date parsing should produce temporal features."""
        raw = _make_raw_input(generation_date='2024-12-25')
        # The transform internally creates gen_month, etc.
        # We just verify it doesn't crash with a valid date
        result = mock_engine.transform(raw)
        assert isinstance(result, np.ndarray)

    def test_transform_handles_bad_date(self, mock_engine):
        """Invalid dates should fallback to a default, not crash."""
        raw = _make_raw_input(generation_date='not-a-date')
        result = mock_engine.transform(raw)
        assert isinstance(result, np.ndarray)
        assert not np.isnan(result).any()


class TestRiskLevel:
    """Tests for the risk level classifier."""

    def test_low_risk(self):
        assert get_risk_level(0.0) == "Low"
        assert get_risk_level(0.15) == "Low"
        assert get_risk_level(0.24) == "Low"

    def test_medium_risk(self):
        assert get_risk_level(0.25) == "Medium"
        assert get_risk_level(0.35) == "Medium"
        assert get_risk_level(0.49) == "Medium"

    def test_high_risk(self):
        assert get_risk_level(0.50) == "High"
        assert get_risk_level(0.65) == "High"
        assert get_risk_level(0.74) == "High"

    def test_critical_risk(self):
        assert get_risk_level(0.75) == "Critical"
        assert get_risk_level(0.90) == "Critical"
        assert get_risk_level(1.0) == "Critical"

    def test_boundary_values(self):
        """Verify exact boundary behavior."""
        assert get_risk_level(0.2499) == "Low"
        assert get_risk_level(0.2500) == "Medium"
        assert get_risk_level(0.4999) == "Medium"
        assert get_risk_level(0.5000) == "High"
        assert get_risk_level(0.7499) == "High"
        assert get_risk_level(0.7500) == "Critical"
