# ML Opsidian Genesis: Flood Risk Prediction
**Current Best Leaderboard Score:** `0.38130` (The Final Geo-Blend)

## 1. Problem Statement
Cyclone events and rapid environmental changes frequently turn normal situations into severe flood disasters in Sri Lanka. Despite having historical records, this data is rarely used to compute actionable long-term risk intelligence for habitability. 

This challenge focuses on moving beyond simple flood prediction to **risk intelligence**. The goal is to accurately predict the `flood_risk_score` (a continuous variable between 0 and 1) for various regions using historical, geographical, and meteorological data.

### Custom Evaluation Metric
The leaderboard evaluates submissions on a highly specific custom metric that strictly shapes our modeling strategy:
1. **Balanced Error Assessment**: Calculates a foundational error score that *strictly penalizes large, unexpected outliers*.
2. **Explained Variance Penalty**: The foundational error is dynamically scaled based on how well the predictions *track the actual fluctuations* of the target variable.

**Strategic Implication**: The model must take enough risks (high variance) to track complex fluctuations, but must be strictly bounded so that it never predicts a catastrophic outlier. 

---

## 2. Dataset Description
- **Train Data**: 20,886 rows
- **Test Data**: 5,300 rows
- **Target**: `flood_risk_score`

### Core Features
- **Categorical**: `district`, `landcover`, `soil_type`, `place_name`, `water_supply`, `flood_occurrence_current_event`, etc.
- **Meteorological**: `rainfall_7d_mm`, `monthly_rainfall_mm`, `extreme_weather_index`, `seasonal_index`.
- **Geographical**: `elevation_m`, `distance_to_river_m`, `drainage_index`, `terrain_roughness_index`.
- **Infrastructural**: `infrastructure_score`, `nearest_hospital_km`, `nearest_evac_km`.
- **Satellite**: `ndvi` (Vegetation), `ndwi` (Water).

### Feature Engineering (The "115-Feature Set")
Through rigorous testing, we found that exactly **115 features** is the sweet spot for balancing signal and noise. Key additions include:
- **Fold-Safe Target Encoding**: Applied to high-cardinality categorical features (e.g., `district`, `place_name`, `soil_type`) to extract rich target correlations without data leakage.
- **Interaction Terms**: Multiplying encoded categoricals by raw meteorological data (e.g., `dist_te_x_rain`, `soil_te_x_extreme`).
- **Domain Ratios**: `rain_ratio` (7-day / monthly), `river_rain_risk` (rain / distance to river), `inundation_per_pop`.

---

## 3. Version History & Insights

> [!TIP]
> **The OOF Overfitting Paradox**
> During V16, we discovered that optimizing models strictly to achieve the lowest possible Out-Of-Fold (OOF) RMSE actually *decreased* the Public Leaderboard score. The evaluation metric explicitly rewards models that track high-variance fluctuations. If a model is too "smooth" or conservative (like V16), it fails the Explained Variance Penalty. We must explicitly include high-variance models (deep trees) but stack them securely to prevent outliers.

### Early Baselines (V1 - V7)
- **Scores**: Ranged from `0.40765` to `0.38363`.
- **Insights**: Initial EDA, baseline LightGBM and CatBoost models. Established that ensemble stacking is necessary to reduce prediction variance.

### V8: The First Breakthrough
- **Score**: `0.38215`
- **Insights**: Used 104 features and a simple stacked ensemble. Proved that combining multiple models significantly outperforms single estimators. 

### V11: The Feature Champion
- **Score**: `0.38194` | **OOF RMSE**: `0.23447`
- **Insights**: Introduced the definitive 115-feature set (including fold-safe target encoding). Swept CatBoost across depths (5, 6, 7, 8, 10). Used a standard `Ridge` meta-model for stacking.

### V12 & V13: Hypothesis Testing
- **Scores**: V12 (`0.38258`), V13 (`0.38205`)
- **Insights**: V12 reduced features to 101, resulting in a score drop. V13 introduced Huber/MAE loss functions natively into CatBoost, but still underperformed V11. This confirmed that the **115-feature set + RMSE base models** is optimal.

### V14: The Meta-Ensemble
- **Score**: `0.38195`
- **Insights**: A weighted blend of V11, V13, and V8. Showed that blending diverse, top-tier submissions yields highly stable results.

### V15: The Robust Stacker
- **Score**: `0.38189` | **OOF RMSE**: `0.23452`
- **Insights**: Used the exact same 115 features and base models as V11, but replaced the `Ridge` meta-model with a `HuberRegressor` (Epsilon=1.35). Huber is mathematically designed to ignore outliers. By capping the predictions of deep trees, it perfectly satisfied the custom metric's "Balanced Error Assessment", securing a massive LB jump.

### V16: The OOF Paradox
- **Score**: `0.38202` | **OOF RMSE**: `0.23445` (Best at the time)
- **Insights**: Since V15's Huber stacker heavily penalized deep trees (depth 8, 10), V16 was built strictly with shallow trees (depths 3 to 7). It achieved the best internal OOF score but failed on the Leaderboard. This proved that deep trees are essential for capturing target fluctuations, even if their weights are small.

### V17 Genesis: The Ultimate Ensemble
- **Score**: `0.38163` (Current Champion)
- **Insights**: The culmination of all prior insights.
  - **20-Fold CV**: For extreme baseline stability.
  - **Shallow Models**: CatBoost (3,4,5) and LGB (MAE) for stable anchors.
  - **Deep Models**: CatBoost (8,10,12) and LGB (255 leaves) to track high-variance fluctuations.
  - **Robust Meta-Model**: `HuberRegressor` to fuse them securely without generating outliers.
  - **Final Action**: Blended V17 (30%) + V15 (40%) + V11 (30%) to create the `submission_v17_genesis_ultimate.csv` file, resulting in the massive leap to `0.38163`.

### V18 Titan: The Scale Champion
- **Score**: `0.38149` (Current Champion) | **OOF RMSE**: `0.23388` | **R2**: `0.0405`
- **Insights**: Proved that sheer scale and diversity is the winning formula for this metric.
  - **36 distinct models**: CatBoost (depth 4,6,8,10,12), LightGBM (MAE, RMSE, Deep-255, DART), XGBoost (SquaredError, PseudoHuber), HistGradientBoosting.
  - **3 random seeds per model**: Seed averaging (42, 142, 242) neutralized initialization variance.
  - **20-Fold CV with checkpointing**: Allowed crash-safe multi-hour training.
  - **Raw target prediction**: Models predict `flood_risk_score` directly (no transformation).
  - **HuberRegressor (ε=1.35) stacker**: Proven meta-model.
  - **Training time**: ~12 hours.

> [!CAUTION]
> **V19 Excalibur: The Transformation Failure**
> - **Score**: `0.38415` (Major Regression!) | **OOF RMSE**: `0.23428` | **R2**: `0.0372`
> - **What was tried**:
>   - **Logit Target Transformation**: `log(y / (1-y))` to map target to unbounded space, then sigmoid back. This distorted the optimization landscape and introduced systematic bias in the predictions.
>   - **Pseudo-Labeling**: Used V18's test predictions as "soft labels" to augment training data. This created confirmation bias — the models learned to replicate V18's errors rather than discovering new signal.
>   - **60-Fold CV** (3 repeats × 20 folds): Did not compensate for the above failures.
> - **Critical Lesson**: For this specific metric, **raw target prediction on the original scale is essential**. Any transformation of the target or injection of noisy pseudo-labels degrades explained variance tracking. The winning formula remains: diverse raw-target models + Huber stacking.

### V20 Colossus: The Multi-GPU Swarm
- **Score**: `0.38139` (Current Champion - 50/50 blend with V18) | **OOF RMSE**: `0.23197` | **R2**: `0.0561`
- **Insights**: Reverted to V18's raw-target approach but pushed scale to an absolute extreme.
  - **100-Fold CV**: 5 repeats × 20 folds. Squeezed every drop of signal out of the data.
  - **10 Seeds per Config**: Annihilated random initialization variance.
  - **Multi-GPU Parallelism**: The training was distributed across 3 separate machines simultaneously (Ubuntu VM, RTX 4070, GTX 3060) to meet the deadline.
  - **The OOF Paradox Confirmed Again**: Early in training, when only HGB models were finished, OOF RMSE dropped to `0.23287` but LB was `0.38327`. Only after the GPU machines injected Deep CatBoost trees (depths 6, 8, 10) into the stack did the LB score shatter the plateau and reach `0.38139`.

### The Final Geo-Blend (The Ultimate Submission)
- **Score**: `0.38130` (The Absolute Champion)
- **Insights**: In the final 30 minutes of the competition, we fused the three historical champion architectures together:
  1. **V17 Genesis** (`0.38163`) - The 20-fold meta-ensemble pioneer.
  2. **V18 Titan** (`0.38149`) - The massive 36-model, 3-seed scale champion.
  3. **V20 Colossus** (`0.38225` standalone, but vastly superior OOF metrics) - The 100-fold multi-GPU swarm.
- **The Technique**: Instead of a simple weighted average, we used a **Geometric Mean** `(V17 * V18 * V20) ^ (1/3)`. 
- **Why it Won**: The Geometric Mean mathematically penalizes extreme spikes more harshly than an arithmetic mean. Because the three models learned fundamentally distinct signals across the competition, the geometric blend cleanly squashed their respective outliers while perfectly preserving their underlying variance tracking, completely maximizing the "Balanced Error Assessment" metric.

