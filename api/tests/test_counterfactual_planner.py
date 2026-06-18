"""Verification tests for the counterfactual planner and API registration."""

import asyncio
import hashlib
from pathlib import Path
from unittest.mock import patch

from api.counterfactual_planner import generate_counterfactual_plan
from api.schemas import OptimizePlanRequest, OptimizePlanResponse


REQUIRED_RESPONSE_FIELDS = {
    "district",
    "model_version",
    "baseline_score",
    "baseline_risk_level",
    "target_risk_level",
    "target_score_threshold",
    "target_reached",
    "optimized_score",
    "optimized_risk_level",
    "risk_reduction_pct",
    "recommended_plan",
    "search_trace",
    "alternatives",
}


def _mock_predict(features: dict, model_version: str):
    """Deterministic black-box score with improving actionable directions."""
    score = 0.90
    score -= features["drainage_index"] * 0.20
    score -= features["infrastructure_score"] * 0.15
    score += features["nearest_evac_km"] * 0.01
    score += features["nearest_hospital_km"] * 0.005
    score -= features["distance_to_river_m"] * 0.00002
    score += features["built_up_percent"] * 0.001
    return max(0.0, min(1.0, score)), 0.0, []


def _artifact_hashes() -> dict[str, str]:
    artifacts_dir = Path(__file__).parents[2] / "mlops" / "artifacts"
    hashes = {}
    if not artifacts_dir.exists():
        return hashes

    for path in sorted(artifacts_dir.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as artifact:
                for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[str(path.relative_to(artifacts_dir))] = digest.hexdigest()
    return hashes


def test_plan_has_required_response_fields():
    with patch(
        "api.counterfactual_planner.model_manager.predict",
        side_effect=_mock_predict,
    ):
        result = generate_counterfactual_plan(
            district="Colombo",
            model_version="v13",
            target_risk_level="Low",
        )

    assert set(result) == REQUIRED_RESPONSE_FIELDS
    OptimizePlanResponse.model_validate(result)
    assert isinstance(result["target_reached"], bool)


def test_unknown_allowed_features_are_ignored_safely():
    with patch(
        "api.counterfactual_planner.model_manager.predict",
        side_effect=_mock_predict,
    ):
        result = generate_counterfactual_plan(
            district="Colombo",
            model_version="v13",
            target_risk_level="Low",
            allowed_features=["not_a_feature", "drainage_index"],
        )

    assert all(
        step["feature"] == "drainage_index"
        for step in result["recommended_plan"]
    )
    assert all(
        item["tried_feature"] == "drainage_index"
        for item in result["search_trace"]
    )


def test_max_steps_is_respected():
    max_steps = 2
    with patch(
        "api.counterfactual_planner.model_manager.predict",
        side_effect=_mock_predict,
    ):
        result = generate_counterfactual_plan(
            district="Colombo",
            model_version="v13",
            target_risk_level="Low",
            max_steps=max_steps,
        )

    assert len(result["recommended_plan"]) <= max_steps
    accepted_steps = {
        item["step"] for item in result["search_trace"] if item["accepted"]
    }
    assert len(accepted_steps) <= max_steps
    assert max(
        (item["step"] for item in result["search_trace"]),
        default=0,
    ) <= max_steps


def test_planner_does_not_modify_model_artifacts():
    before = _artifact_hashes()

    with patch(
        "api.counterfactual_planner.model_manager.predict",
        side_effect=_mock_predict,
    ):
        generate_counterfactual_plan(
            district="Colombo",
            model_version="v13",
            target_risk_level="Medium",
            max_steps=1,
        )

    after = _artifact_hashes()
    assert after == before


def test_optimize_route_and_existing_endpoint_imports():
    from api.main import app, health_check, optimize_plan, simulate

    assert callable(health_check)
    assert callable(simulate)
    assert callable(optimize_plan)

    route = next(
        route
        for route in app.routes
        if getattr(route, "path", None) == "/api/optimize-plan"
    )
    assert "POST" in route.methods
    assert route.response_model is OptimizePlanResponse

    request = OptimizePlanRequest(
        district="Colombo",
        model_version="v13",
        target_risk_level="Low",
        max_steps=2,
        allowed_features=["drainage_index", "not_a_feature"],
    )
    with patch(
        "api.counterfactual_planner.model_manager.predict",
        side_effect=_mock_predict,
    ):
        response = asyncio.run(optimize_plan(request))

    assert isinstance(response, OptimizePlanResponse)
    assert isinstance(response.target_reached, bool)
    assert len(response.recommended_plan) <= request.max_steps
