"""
V20 Colossus Model Export Script

Uses pre-computed OOF/test checkpoints from v20_colossus_solution.py to:
1. Fit the HuberRegressor meta-model on OOF predictions
2. Retrain base models on full training data for real-time inference
3. Serialize the complete pipeline + feature engineering artifacts

Architecture:
- 270 models (LGB × 60, HGB × 50, CatBoost × 120, XGB × 40)
- 100-fold cross-validation with 10 seeds per config
- HuberRegressor(ε=1.35) meta-model

Checkpoint directory: colossus_checkpoints/
Output: mlops/artifacts/v20/ directory
"""

import numpy as np
import pandas as pd
import os
import json
import glob
import joblib
import warnings
import time
warnings.filterwarnings('ignore')

from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

import lightgbm as lgb
from catboost import CatBoostRegressor

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed. XGB models will be skipped.")

# ─── Configuration ──────────────────────────────────────────────

SEED = 42
N_FOLDS = 10   # For TE computation during export (the OOF checkpoints used 100-fold)
DATA_DIR = "data"
CHECKPOINT_DIR = "colossus_checkpoints"
OUTPUT_DIR = os.path.join("mlops", "artifacts", "v20")
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(SEED)

TARGET = 'flood_risk_score'
ALL_SEEDS = [42, 142, 242, 342, 442, 542, 642, 742, 842, 942]

CAT_COLS = ['district', 'landcover', 'soil_type', 'water_supply', 'electricity',
            'road_quality', 'urban_rural', 'water_presence_flag',
            'flood_occurrence_current_event', 'is_good_to_live',
            'reason_not_good_to_live', 'place_name']
DROP_COLS = ['record_id', 'gen_date', 'generation_date', 'is_synthetic', TARGET]

TE_COLS_AND_SMOOTHING = [
    ('place_name', 5), ('district', 10), ('soil_type', 10),
    ('landcover', 10), ('road_quality', 10), ('flood_occurrence_current_event', 10),
]

print("=" * 60)
print("  V20 COLOSSUS MODEL EXPORT")
print("=" * 60)

# ─── Step 1: Load & Verify Checkpoints ─────────────────────────

oof_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*_oof.npy")))
if not oof_files:
    print(f"\nERROR: No checkpoints found in {CHECKPOINT_DIR}/")
    print("Run `python v20_colossus_solution.py` first, or copy checkpoints here.")
    exit(1)

model_names = [os.path.basename(f).replace("_oof.npy", "") for f in oof_files
               if "features_cached" not in os.path.basename(f)]
print(f"\nFound {len(model_names)} completed model checkpoints")

families = {}
for m in model_names:
    family = m.split('_')[0]
    families.setdefault(family, []).append(m)
for fam, members in sorted(families.items()):
    print(f"  {fam}: {len(members)} models")

# ─── Step 2: Feature Engineering ───────────────────────────────

print("\nLoading data...")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
print(f"Train: {train.shape}, Test: {test.shape}")

for df in [train, test]:
    df['gen_date'] = pd.to_datetime(df['generation_date'])
    df['gen_month'] = df['gen_date'].dt.month
    df['gen_year'] = df['gen_date'].dt.year
    df['gen_day_of_year'] = df['gen_date'].dt.dayofyear
    df['gen_quarter'] = df['gen_date'].dt.quarter
    df['is_ne_monsoon'] = df['gen_month'].isin([12, 1, 2]).astype(int)
    df['is_sw_monsoon'] = df['gen_month'].isin([5, 6, 7, 8, 9]).astype(int)
    df['gen_month_sin'] = np.sin(2 * np.pi * df['gen_month'] / 12)
    df['gen_month_cos'] = np.cos(2 * np.pi * df['gen_month'] / 12)
    reason = df['reason_not_good_to_live'].fillna('Other')
    df['reason_flood_flag'] = reason.str.contains('flood', case=False).astype(int)
    df['reason_infra_flag'] = reason.str.contains('infrastructure', case=False).astype(int)
    df['reason_road_flag'] = reason.str.contains('road', case=False).astype(int)
    df['reason_other_flag'] = (reason == 'Other').astype(int)
    df['is_good_binary'] = (df['is_good_to_live'] == 'Yes').astype(int)
    df['log_inundation'] = np.log1p(df['inundation_area_sqm'])
    df['sqrt_inundation'] = np.sqrt(df['inundation_area_sqm'])
    df['inundation_per_pop'] = df['inundation_area_sqm'] / (df['population_density_per_km2'] + 1)
    df['record_id_num'] = df['record_id'].str.replace('F', '', regex=False).astype(int)


def engineer_features(df):
    df = df.copy()
    eps = 1e-6
    df['rainfall_x_flood'] = df['rainfall_7d_mm'] * df['historical_flood_count']
    df['monthly_x_flood'] = df['monthly_rainfall_mm'] * df['historical_flood_count']
    df['rain_ratio'] = df['rainfall_7d_mm'] / (df['monthly_rainfall_mm'] + eps)
    df['rain_cum'] = df['rainfall_7d_mm'] + df['monthly_rainfall_mm']
    df['river_clip'] = df['distance_to_river_m'].clip(lower=0)
    df['river_rain_risk'] = df['rainfall_7d_mm'] / (df['river_clip'] + 1)
    df['river_monthly_risk'] = df['monthly_rainfall_mm'] / (df['river_clip'] + 1)
    df['elev_clip'] = df['elevation_m'].clip(lower=0)
    df['elev_rain_ratio'] = df['rainfall_7d_mm'] / (df['elev_clip'] + 1)
    df['low_elev_flag'] = (df['elevation_m'] < 30).astype(int)
    df['infra_socio'] = df['infrastructure_score'] * df['socioeconomic_status_index']
    df['water_veg_balance'] = df['ndwi'] - df['ndvi']
    df['ndvi_ndwi_product'] = df['ndvi'] * df['ndwi']
    df['ndwi_sq'] = df['ndwi'] ** 2
    df['drainage_x_rain'] = df['drainage_index'] * df['rainfall_7d_mm']
    df['bad_drainage_rain'] = (df['drainage_index'] < 0.35).astype(int) * df['rainfall_7d_mm']
    df['urban_runoff'] = df['built_up_percent'] * df['rainfall_7d_mm'] / 100
    df['evac_hosp_sum'] = df['nearest_hospital_km'] + df['nearest_evac_km']
    df['max_dist_help'] = df[['nearest_hospital_km', 'nearest_evac_km']].max(axis=1)
    df['pop_x_rain'] = df['population_density_per_km2'] * df['rainfall_7d_mm']
    df['pop_x_flood'] = df['population_density_per_km2'] * df['historical_flood_count']
    df['extreme_x_rain'] = df['extreme_weather_index'] * df['rainfall_7d_mm']
    df['extreme_x_flood'] = df['extreme_weather_index'] * df['historical_flood_count']
    df['extreme_x_monthly'] = df['extreme_weather_index'] * df['monthly_rainfall_mm']
    df['seasonal_rain'] = df['seasonal_index'] * df['rainfall_7d_mm']
    df['seasonal_extreme'] = df['seasonal_index'] * df['extreme_weather_index']
    df['terrain_rain'] = df['terrain_roughness_index'] * df['rainfall_7d_mm']
    df['inundation_x_rain'] = df['inundation_area_sqm'] * df['rainfall_7d_mm']
    df['log_inundation_x_extreme'] = df['log_inundation'] * df['extreme_weather_index']
    df['composite_vuln'] = (
        df['rainfall_7d_mm'] * 0.3 + df['historical_flood_count'] * 15.0 +
        df['extreme_weather_index'] * 50.0 + (1 - df['drainage_index']) * 30.0 +
        df['built_up_percent'] * 0.10)
    return df


print("Engineering features...")
train = engineer_features(train)
test = engineer_features(test)

# Label encoding
all_data = pd.concat([train, test], axis=0, ignore_index=True)
label_encoder_maps = {}
for col in CAT_COLS:
    if col in all_data.columns:
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col].astype(str).fillna('missing'))
        label_encoder_maps[col] = {cls: int(idx) for idx, cls in enumerate(le.classes_)}

with open(os.path.join(OUTPUT_DIR, "label_encoders.json"), "w") as f:
    json.dump(label_encoder_maps, f, indent=2)

n_train = len(train)
train_enc = all_data.iloc[:n_train].copy()
test_enc = all_data.iloc[n_train:].copy()
train_enc[TARGET] = train[TARGET].values

EXCLUDE = set(DROP_COLS + [TARGET])
feature_cols = [c for c in train_enc.columns if c not in EXCLUDE]
X = train_enc[feature_cols].copy()
y = train_enc[TARGET].copy()
X_test = test_enc[feature_cols].copy()
medians = X.median(numeric_only=True)
X = X.fillna(medians)
X_test = X_test.fillna(medians)

medians_dict = {k: float(v) for k, v in medians.to_dict().items()}
with open(os.path.join(OUTPUT_DIR, "medians.json"), "w") as f:
    json.dump(medians_dict, f, indent=2)

global_mean = float(y.mean())
with open(os.path.join(OUTPUT_DIR, "global_mean.json"), "w") as f:
    json.dump(global_mean, f)

# Target encoding (10-fold for export, matching the TE used in V20 training)
def fold_safe_te(X_df, y_s, X_te_df, col, smooth, gm, n_folds=N_FOLDS, seed=SEED):
    tr_te = np.zeros(len(X_df))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf.split(X_df):
        s = pd.DataFrame({'k': X_df.iloc[tr_idx][col].values, 'y': y_s.iloc[tr_idx].values}).groupby('k')['y'].agg(['mean', 'count'])
        s['enc'] = (s['mean'] * s['count'] + gm * smooth) / (s['count'] + smooth)
        tr_te[val_idx] = X_df.iloc[val_idx][col].map(s['enc']).fillna(gm).values
    s_all = pd.DataFrame({'k': X_df[col].values, 'y': y_s.values}).groupby('k')['y'].agg(['mean', 'count'])
    s_all['enc'] = (s_all['mean'] * s_all['count'] + gm * smooth) / (s_all['count'] + smooth)
    return tr_te, X_te_df[col].map(s_all['enc']).fillna(gm).values, s_all['enc'].to_dict()


gm = y.mean()
print("Computing target encodings + interactions...")
te_store = {}
te_stats_all = {}

for col, smooth in TE_COLS_AND_SMOOTHING:
    tr_te, te_te, stats = fold_safe_te(X, y, X_test, col, smooth, gm, N_FOLDS, SEED)
    X[f'{col}_te'] = tr_te
    X_test[f'{col}_te'] = te_te
    te_store[col] = (tr_te, te_te)
    te_stats_all[col] = {str(k): float(v) for k, v in stats.items()}

with open(os.path.join(OUTPUT_DIR, "te_stats.json"), "w") as f:
    json.dump(te_stats_all, f, indent=2)

d_tr, d_te = te_store['district']
f_tr, f_te = te_store['flood_occurrence_current_event']
s_tr, s_te = te_store['soil_type']

for feat, tr_v, te_v, col_name in [
    ('rainfall_7d_mm', d_tr, d_te, 'dist_te_x_rain'),
    ('extreme_weather_index', d_tr, d_te, 'dist_te_x_extreme'),
    ('historical_flood_count', d_tr, d_te, 'dist_te_x_flood'),
    ('rainfall_7d_mm', f_tr, f_te, 'fl_te_x_rain'),
    ('extreme_weather_index', f_tr, f_te, 'fl_te_x_extreme'),
    ('drainage_index', d_tr, d_te, 'dist_te_x_drainage'),
    ('river_clip', d_tr, d_te, 'dist_te_x_river'),
    ('elevation_m', d_tr, d_te, 'dist_te_x_elev'),
    ('historical_flood_count', f_tr, f_te, 'fl_te_x_flood'),
    ('drainage_index', f_tr, f_te, 'fl_te_x_drainage'),
    ('rainfall_7d_mm', s_tr, s_te, 'soil_te_x_rain'),
    ('extreme_weather_index', s_tr, s_te, 'soil_te_x_extreme'),
]:
    X[col_name] = tr_v * X[feat]
    X_test[col_name] = te_v * X_test[feat]

feature_cols = list(X.columns)
X_arr = X.values.astype(np.float32)
X_test_arr = X_test.values.astype(np.float32)
y_arr = y.values.astype(np.float32)

with open(os.path.join(OUTPUT_DIR, "feature_columns.json"), "w") as f:
    json.dump(feature_cols, f, indent=2)
print(f"Final features: {len(feature_cols)}")

# ─── Step 3: Fit Meta-Model on OOF Checkpoints ────────────────

print("\nLoading OOF checkpoints for meta-model fitting...")
oof_stack = []
for m in model_names:
    oof_stack.append(np.load(os.path.join(CHECKPOINT_DIR, f"{m}_oof.npy")))
oof_stack = np.column_stack(oof_stack)
print(f"OOF stacking matrix: {oof_stack.shape[0]} samples × {oof_stack.shape[1]} models")

huber = HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=2000)
huber.fit(oof_stack, y_arr)

huber_oof = huber.predict(oof_stack)
stack_rmse = np.sqrt(mean_squared_error(y_arr, huber_oof))
stack_r2 = r2_score(y_arr, huber_oof)
print(f"[COLOSSUS STACK] OOF RMSE = {stack_rmse:.5f} | R² = {stack_r2:.4f}")

weights = list(zip(model_names, huber.coef_))
weights.sort(key=lambda x: abs(x[1]), reverse=True)
print("\nTop 15 model weights:")
for name, w in weights[:15]:
    print(f"  {name:30s} : {w:+.6f}")

# ─── Step 4: Retrain Base Models on Full Data ──────────────────
# V20 has ~270 models. Retraining ALL would take many hours.
# Strategy: Parse model name → reconstruct config → train on full data.
# For very large ensembles, this step can be batched or parallelized.

print("\n" + "=" * 60)
print("  RETRAINING BASE MODELS ON FULL DATA FOR SERVING")
print(f"  ({len(model_names)} models — this will take a while)")
print("=" * 60)

full_models = {}
total_start = time.time()
skipped = 0

for i, m_name in enumerate(model_names):
    parts = m_name.split('_')
    family = parts[0]
    start = time.time()

    try:
        if family == 'hgb':
            # hgb_d{depth}_s{seed}
            depth = int(parts[1].replace('d', ''))
            seed = int(parts[2].replace('s', ''))
            model = HistGradientBoostingRegressor(
                max_iter=500, learning_rate=0.03, max_depth=depth,
                random_state=seed, early_stopping=False
            )
            model.fit(X_arr, y_arr)
            full_models[m_name] = model

        elif family == 'lgb':
            # lgb_{variant}_s{seed}  e.g. lgb_mae_s42, lgb_deep_s142, lgb_wide_s42, lgb_huber_s42
            variant = parts[1]
            seed = int(parts[2].replace('s', ''))

            if variant == 'mae':
                p = {'objective': 'regression_l1', 'metric': 'rmse', 'learning_rate': 0.03,
                     'num_leaves': 63, 'min_child_samples': 20, 'feature_fraction': 0.7,
                     'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.1,
                     'reg_lambda': 1.0, 'n_jobs': -1, 'verbose': -1, 'seed': seed}
            elif variant == 'rmse':
                p = {'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.03,
                     'num_leaves': 63, 'min_child_samples': 20, 'feature_fraction': 0.7,
                     'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.1,
                     'reg_lambda': 1.0, 'n_jobs': -1, 'verbose': -1, 'seed': seed}
            elif variant == 'deep':
                p = {'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.02,
                     'num_leaves': 255, 'min_child_samples': 30, 'feature_fraction': 0.6,
                     'bagging_fraction': 0.75, 'bagging_freq': 5, 'reg_alpha': 0.2,
                     'reg_lambda': 2.0, 'n_jobs': -1, 'verbose': -1, 'seed': seed}
            elif variant == 'dart':
                p = {'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.05,
                     'num_leaves': 63, 'min_child_samples': 20, 'feature_fraction': 0.7,
                     'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.1,
                     'reg_lambda': 1.0, 'boosting_type': 'dart',
                     'n_jobs': -1, 'verbose': -1, 'seed': seed}
            elif variant == 'wide':
                p = {'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.03,
                     'num_leaves': 127, 'min_child_samples': 15, 'feature_fraction': 0.8,
                     'bagging_fraction': 0.85, 'bagging_freq': 5, 'reg_alpha': 0.05,
                     'reg_lambda': 0.5, 'n_jobs': -1, 'verbose': -1, 'seed': seed}
            elif variant == 'huber':
                p = {'objective': 'huber', 'metric': 'rmse', 'learning_rate': 0.03,
                     'num_leaves': 63, 'min_child_samples': 20, 'feature_fraction': 0.7,
                     'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.1,
                     'reg_lambda': 1.0, 'n_jobs': -1, 'verbose': -1, 'seed': seed}
            else:
                print(f"  [{i+1}/{len(model_names)}] [SKIP] Unknown LGB: {m_name}")
                skipped += 1
                continue

            model = lgb.train(p, lgb.Dataset(X_arr, y_arr), 3000)
            full_models[m_name] = model

        elif family == 'xgb' and HAS_XGB:
            # xgb_sq_d{depth}_s{seed} or xgb_hub_d{depth}_s{seed}
            variant = parts[1]
            depth = int(parts[2].replace('d', ''))
            seed = int(parts[3].replace('s', ''))

            if variant == 'sq':
                p = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
                     'learning_rate': 0.03, 'max_depth': depth, 'subsample': 0.8,
                     'colsample_bytree': 0.7, 'alpha': 0.1, 'lambda': 1.0,
                     'seed': seed, 'tree_method': 'hist'}
            elif variant == 'hub':
                p = {'objective': 'reg:pseudohubererror', 'eval_metric': 'rmse',
                     'learning_rate': 0.03, 'max_depth': depth, 'subsample': 0.8,
                     'colsample_bytree': 0.7, 'alpha': 0.1, 'lambda': 1.0,
                     'seed': seed, 'tree_method': 'hist'}
            else:
                print(f"  [{i+1}/{len(model_names)}] [SKIP] Unknown XGB: {m_name}")
                skipped += 1
                continue

            dtrain = xgb.DMatrix(X_arr, label=y_arr)
            model = xgb.train(p, dtrain, num_boost_round=3000)
            full_models[m_name] = model

        elif family == 'cat':
            # cat_d{depth}_s{seed} or cat_d{depth}_slow_s{seed} or cat_d{depth}_reg_s{seed}
            depth = int(parts[1].replace('d', ''))
            seed = int(parts[-1].replace('s', ''))

            # Check for variant modifiers
            variant = None
            if 'slow' in m_name:
                variant = 'slow'
            elif 'reg' in m_name:
                variant = 'reg'

            p = dict(iterations=3000, learning_rate=0.03, depth=depth,
                     l2_leaf_reg=3.0, min_data_in_leaf=15,
                     subsample=0.8, bootstrap_type='Poisson',
                     random_seed=seed, task_type='GPU', verbose=0)

            if variant == 'slow':
                p['learning_rate'] = 0.01
                p['iterations'] = 5000
            elif variant == 'reg':
                p['l2_leaf_reg'] = 10.0
                p['min_data_in_leaf'] = 30

            model = CatBoostRegressor(**p)
            model.fit(X_arr, y_arr, verbose=0)
            full_models[m_name] = model

        elif family == 'xgb' and not HAS_XGB:
            print(f"  [{i+1}/{len(model_names)}] [SKIP] {m_name} (xgboost not installed)")
            skipped += 1
            continue

        else:
            print(f"  [{i+1}/{len(model_names)}] [SKIP] {m_name} (unknown family: {family})")
            skipped += 1
            continue

        elapsed = time.time() - start
        if (i + 1) % 10 == 0 or (i + 1) == len(model_names):
            total_elapsed = (time.time() - total_start) / 60
            print(f"  [{i+1}/{len(model_names)}] {m_name} ({elapsed:.1f}s) — total: {total_elapsed:.1f} min")

    except Exception as e:
        print(f"  [{i+1}/{len(model_names)}] [FAIL] {m_name}: {e}")
        skipped += 1

total_mins = (time.time() - total_start) / 60
print(f"\nRetrained {len(full_models)} / {len(model_names)} models in {total_mins:.1f} min")
if skipped > 0:
    print(f"  (Skipped {skipped} models)")

# ─── Step 5: Build & Save Inference Pipeline ──────────────────

class InferencePipeline:
    """V20 Colossus inference pipeline: base models + HuberRegressor meta."""

    def __init__(self, base_models, meta_model, model_names):
        self.base_models = base_models
        self.meta_model = meta_model
        self.model_names = model_names

    def predict(self, X):
        base_preds = []
        for name in self.model_names:
            model = self.base_models[name]
            if hasattr(model, 'predict'):
                if hasattr(model, 'best_iteration'):
                    import xgboost as xgb
                    base_preds.append(model.predict(xgb.DMatrix(X)))
                else:
                    base_preds.append(model.predict(X))
            else:
                # LightGBM Booster
                base_preds.append(model.predict(X))
        stack_input = np.column_stack(base_preds)
        return self.meta_model.predict(stack_input)


valid_names = [n for n in model_names if n in full_models]
pipeline = InferencePipeline(full_models, huber, valid_names)
pipeline_path = os.path.join(OUTPUT_DIR, "pipeline.joblib")
joblib.dump(pipeline, pipeline_path)

file_size_mb = os.path.getsize(pipeline_path) / 1024 / 1024
print(f"\n{'=' * 60}")
print(f"V20 COLOSSUS MODEL EXPORT COMPLETE")
print(f"{'=' * 60}")
print(f"Artifacts saved to: {OUTPUT_DIR}/")
print(f"  - pipeline.joblib ({file_size_mb:.1f} MB)")
print(f"  - label_encoders.json")
print(f"  - te_stats.json")
print(f"  - feature_columns.json ({len(feature_cols)} features)")
print(f"  - medians.json")
print(f"  - global_mean.json")
print(f"Models exported: {len(valid_names)} / {len(model_names)}")
print(f"Stack RMSE: {stack_rmse:.5f} | R²: {stack_r2:.4f}")
print(f"Total retraining time: {total_mins:.1f} min")
print(f"{'=' * 60}")
