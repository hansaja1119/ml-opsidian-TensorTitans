"""
Tests for Pydantic input validation schemas.

Verifies that:
1. Valid inputs are accepted.
2. Invalid inputs (out of bounds, wrong types) are rejected.
3. Default values produce a valid request.
"""

import pytest
from pydantic import ValidationError
from api.schemas import SimulationRequest


class TestSimulationRequest:
    """Tests for the SimulationRequest schema."""

    def test_default_values_are_valid(self):
        """Default values should create a valid request."""
        req = SimulationRequest()
        assert req.district.value == "Colombo"
        assert req.model_version.value == "v13"
        assert 0 <= req.rainfall_7d_mm <= 1500
        assert 0 <= req.drainage_index <= 1

    def test_valid_full_request(self):
        """A fully specified valid request should be accepted."""
        req = SimulationRequest(
            district="Ratnapura",
            model_version="v18_titan",
            rainfall_7d_mm=120.0,
            drainage_index=0.35,
            infrastructure_score=0.6,
            elevation_m=45.0,
            distance_to_river_m=280.0,
            nearest_hospital_km=8.0,
            nearest_evac_km=6.0,
            built_up_percent=22.0,
        )
        assert req.district.value == "Ratnapura"
        assert req.model_version.value == "v18_titan"

    def test_negative_rainfall_rejected(self):
        """Negative rainfall should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SimulationRequest(rainfall_7d_mm=-10.0)
        assert "rainfall_7d_mm" in str(exc_info.value)

    def test_rainfall_over_max_rejected(self):
        """Rainfall over 1500mm should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SimulationRequest(rainfall_7d_mm=2000.0)
        assert "rainfall_7d_mm" in str(exc_info.value)

    def test_drainage_index_out_of_bounds(self):
        """Drainage index must be between 0 and 1."""
        with pytest.raises(ValidationError):
            SimulationRequest(drainage_index=1.5)
        with pytest.raises(ValidationError):
            SimulationRequest(drainage_index=-0.1)

    def test_invalid_district_rejected(self):
        """An unknown district should be rejected."""
        with pytest.raises(ValidationError):
            SimulationRequest(district="Atlantis")

    def test_invalid_model_version_rejected(self):
        """An unknown model version should be rejected."""
        with pytest.raises(ValidationError):
            SimulationRequest(model_version="v999")

    def test_elevation_allows_negative(self):
        """Elevation can be slightly negative (below sea level areas)."""
        req = SimulationRequest(elevation_m=-5.0)
        assert req.elevation_m == -5.0

    def test_elevation_rejects_extreme_negative(self):
        """Elevation below -10m should be rejected."""
        with pytest.raises(ValidationError):
            SimulationRequest(elevation_m=-20.0)

    def test_all_model_versions_valid(self):
        """All three model versions should be accepted."""
        for version in ["v13", "v18_titan", "v20_colossus"]:
            req = SimulationRequest(model_version=version)
            assert req.model_version.value == version
