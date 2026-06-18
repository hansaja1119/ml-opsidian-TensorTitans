# Feature 1: AI Counterfactual Flood Mitigation Planner

## 1. Feature Goal

Generate a practical, ranked mitigation plan that changes a limited number of controllable district features to reduce the model-predicted flood risk to a requested level. The planner must reuse the existing inference pipeline without changing model behavior.

## 2. User Story

As an urban planner, I want to select a district, target risk level, intervention limit, budget profile, and allowed mitigation features so that I can receive an explainable plan showing the smallest practical set of changes likely to reach the target.

## 3. Backend API Contract

### Endpoint

`POST /api/optimize-plan`

### Request

```json
{
  "district": "Colombo",
  "model_version": "v13",
  "target_risk_level": "Medium",
  "max_steps": 3,
  "budget_profile": "balanced",
  "allowed_features": [
    "drainage_index",
    "infrastructure_score",
    "nearest_evac_km",
    "nearest_hospital_km",
    "distance_to_river_m",
    "built_up_percent"
  ]
}
```

Validation:

- `district` must be one of the existing 25 districts.
- `model_version` must be an existing supported model version.
- `target_risk_level` must be `Low`, `Medium`, or `High`.
- `max_steps` must be between 1 and 6.
- `budget_profile` must be `conservative`, `balanced`, or `aggressive`.
- `allowed_features` must be non-empty, unique, and limited to supported actionable features.

### Response

```json
{
  "district": "Colombo",
  "model_version": "v13",
  "baseline_score": 0.8123,
  "baseline_risk_level": "Critical",
  "target_risk_level": "Medium",
  "target_score_threshold": 0.5,
  "target_reached": true,
  "optimized_score": 0.4217,
  "optimized_risk_level": "Medium",
  "risk_reduction_pct": 48.09,
  "recommended_plan": [
    {
      "step": 1,
      "feature": "drainage_index",
      "feature_label": "Drainage System",
      "category": "Infrastructure",
      "from_value": 0.38,
      "to_value": 0.65,
      "score_before": 0.8123,
      "score_after": 0.6901,
      "marginal_reduction_pct": 15.04,
      "estimated_cost": 0.54
    }
  ],
  "search_trace": [
    {
      "iteration": 1,
      "candidates_evaluated": 18,
      "selected_feature": "drainage_index",
      "selected_value": 0.65,
      "resulting_score": 0.6901
    }
  ],
  "alternatives": [
    {
      "rank": 2,
      "optimized_score": 0.4472,
      "risk_reduction_pct": 44.95,
      "target_reached": true,
      "total_estimated_cost": 1.61,
      "plan": []
    }
  ]
}
```

Scores are rounded to four decimal places and percentages to two. `estimated_cost` is a normalized planning cost from 0 to 1 per action, not a currency value. If the target cannot be reached, return the best valid plan with `target_reached: false`. Invalid input returns HTTP 422; an unavailable model version returns HTTP 400.

Risk thresholds remain consistent with `get_risk_level()`:

| Target | Score threshold |
|---|---:|
| Low | 0.25 |
| Medium | 0.50 |
| High | 0.75 |

## 4. Optimization Algorithm

Use deterministic, bounded beam search over counterfactual feature changes:

1. Build the baseline from existing district defaults using `_build_raw_features()`.
2. Score the baseline through the unchanged `model_manager.predict()` method.
3. Generate candidate values for each allowed feature between its current value and configured improvement bound. Candidate density and maximum change depend on the budget profile.
4. At each iteration, expand plans by one unused feature change, score candidates, and rank them by:
   - target reached;
   - lower predicted risk;
   - fewer actions;
   - lower normalized cost.
5. Keep a small fixed beam of the best unique plans and continue until the target is reached, `max_steps` is exhausted, or no candidate improves the score.
6. Return the best plan, a compact iteration trace, and up to three materially distinct alternatives.

Budget profiles control permitted change magnitude and cost penalty:

| Profile | Change magnitude | Cost penalty |
|---|---|---|
| Conservative | Small | High |
| Balanced | Moderate | Medium |
| Aggressive | Full configured range | Low |

The search must be deterministic for identical requests, cap total inference calls, reject non-improving actions, avoid changing the same feature twice, and never mutate district defaults or request data. The model score is an estimate, not a causal guarantee.

## 5. Frontend UX

Add a `/planner` page containing:

- district, model, target risk, maximum steps, and budget selectors;
- checkboxes for allowed intervention features;
- an **Optimize Plan** action with loading and validation states;
- baseline versus optimized risk gauges;
- target reached/not reached status;
- an ordered recommended-plan timeline showing before/after values, marginal impact, and normalized cost;
- a concise search summary and expandable alternatives;
- a disclaimer that recommendations are model-based planning scenarios.

Add a **Planner** link to the existing navbar. Reuse the current visual system and components where practical, and preserve responsive behavior.

## 6. Files to Modify

- `api/schemas.py` — optimization request, plan-step, trace, alternative, and response schemas.
- `api/counterfactual_planner.py` — candidate generation, cost profiles, beam search, and response assembly.
- `api/main.py` — register `POST /api/optimize-plan` and reuse existing district/actionable-feature configuration.
- `api/tests/test_counterfactual_planner.py` — deterministic algorithm and edge-case tests using mocked inference.
- `api/tests/test_api.py` — endpoint contract and validation tests.
- `frontend/src/app/planner/page.tsx` — planner form and results.
- `frontend/src/components/Navbar.tsx` — Planner navigation link.
- `frontend/src/app/globals.css` — planner-specific responsive styling.

No changes are expected in `api/model_manager.py`, `api/feature_engine.py`, `mlops/`, or model artifact directories.

## 7. Acceptance Criteria

- A valid request returns HTTP 200 and the documented response structure.
- Baseline and candidate scores are produced only through `model_manager.predict()`.
- Returned plans use only requested `allowed_features` and contain at most `max_steps` actions.
- Each selected action strictly improves the score relative to its preceding step.
- `target_reached` matches the requested score threshold and optimized score.
- If the baseline already satisfies the target, the response has an empty plan and unchanged optimized score.
- If no plan reaches the target, the endpoint returns the best improving plan with `target_reached: false`.
- Identical requests produce identical plans and ordering.
- Search execution has a fixed inference-call cap and does not perform unbounded combinatorial search.
- Unsupported features and invalid constraints return validation errors.
- The `/planner` page displays success, unreachable-target, loading, empty, and API-error states.
- Existing simulator, intervention, comparison, district, analytics, model, and health tests continue to pass.
- The feature runs locally and in Docker without internet access.

## 8. Explicit Non-Goals

- Do not retrain, fine-tune, or recalibrate any model.
- Do not change model files, model artifacts, feature engineering, MLflow setup, or existing prediction behavior.
- Do not claim causal effectiveness or replace engineering, hydrological, financial, or policy review.
- Do not produce currency-level project estimates, construction schedules, or procurement plans.
- Do not remove or break existing simulator, interventions, comparison, districts, analytics, or health pages.
- Do not add external APIs or require internet access at runtime.
