"""
V20 Colossus Ensemble

The absolute maximum-scale training script. Designed for multi-day training.

Architecture based on lessons from V18 (champion) and V19 (failure):
- RAW target prediction (NO logit transform — V19 proved this hurts)
- NO pseudo-labeling (V19 proved this creates confirmation bias)
- 5 Repeats × 20 Folds = 100-Fold CV per model (extreme stability)
- 10 random seeds per algorithm config (heavy seed averaging)
- 50+ unique model configurations across 4 algorithm families
- 2-Level stacking: Level-1 base models → Level-2 Huber meta-model
- Checkpointing: fully crash-safe, resumes from where it left off

Estimated training time: 2-4 days depending on hardware.
"""

import numpy as np
import pandas as pd
import os
import warnings
import time
import gc
warnings.filterwarnings('ignore')

from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

import lightgbm as lgb
from catboost import CatBoostRegressor

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed. XGBoost models will be skipped.")
    print("Install with: pip install xgboost")

# ─── Configuration ─────────────────────────────────────
SEED           = 42
N_FOLDS        = 20
N_REPEATS      = 5          # 5×20 = 100 folds per model
DATA_DIR       = "data"
CHECKPOINT_DIR = "colossus_checkpoints"
N_SEEDS        = 10         # 10 seeds per config

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
np.random.seed(SEED)

ALL_SEEDS = [42, 142, 242, 342, 442, 542, 642, 742, 842, 942]

print("=" * 70)
print("  V20 COLOSSUS ENSEMBLE")
print(f"  {N_FOLDS}-Fold × {N_REPEATS} Repeats = {N_FOLDS * N_REPEATS} folds per model")
print(f"  {N_SEEDS} random seeds per configuration")
print("  Estimated training time: 2-4 days")
print("=" * 70)

# ─── Data Loading ──────────────────────────────────────
print("\nLoading data...")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
print(f"Train: {train.shape}, Test: {test.shape}")

TARGET = 'flood_risk_score'
test_record_ids = test['record_id'].copy()

# ─── Date / Meta Features ─────────────────────────────
for df in [train, test]:
    df['gen_date']        = pd.to_datetime(df['generation_date'])
    df['gen_month']       = df['gen_date'].dt.month
    df['gen_year']        = df['gen_date'].dt.year
    df['gen_day_of_year'] = df['gen_date'].dt.dayofyear
    df['gen_quarter']     = df['gen_date'].dt.quarter
    df['is_ne_monsoon']   = df['gen_month'].isin([12, 1, 2]).astype(int)
    df['is_sw_monsoon']   = df['gen_month'].isin([5, 6, 7, 8, 9]).astype(int)
    df['gen_month_sin']   = np.sin(2 * np.pi * df['gen_month'] / 12)
    df['gen_month_cos']   = np.cos(2 * np.pi * df['gen_month'] / 12)
    reason = df['reason_not_good_to_live'].fillna('Other')
    df['reason_flood_flag'] = reason.str.contains('flood', case=False).astype(int)
    df['reason_infra_flag'] = reason.str.contains('infrastructure', case=False).astype(int)
    df['reason_road_flag']  = reason.str.contains('road', case=False).astype(int)
    df['reason_other_flag'] = (reason == 'Other').astype(int)
    df['is_good_binary']    = (df['is_good_to_live'] == 'Yes').astype(int)
    df['log_inundation']    = np.log1p(df['inundation_area_sqm'])
    df['sqrt_inundation']   = np.sqrt(df['inundation_area_sqm'])
    df['inundation_per_pop']= df['inundation_area_sqm'] / (df['population_density_per_km2'] + 1)
    df['record_id_num']     = df['record_id'].str.replace('F', '', regex=False).astype(int)

def engineer_features(df):
    df = df.copy(); eps = 1e-6
    df['rainfall_x_flood']       = df['rainfall_7d_mm'] * df['historical_flood_count']
    df['monthly_x_flood']        = df['monthly_rainfall_mm'] * df['historical_flood_count']
    df['rain_ratio']              = df['rainfall_7d_mm'] / (df['monthly_rainfall_mm'] + eps)
    df['rain_cum']                = df['rainfall_7d_mm'] + df['monthly_rainfall_mm']
    df['river_clip']              = df['distance_to_river_m'].clip(lower=0)
    df['river_rain_risk']         = df['rainfall_7d_mm'] / (df['river_clip'] + 1)
    df['river_monthly_risk']      = df['monthly_rainfall_mm'] / (df['river_clip'] + 1)
    df['elev_clip']               = df['elevation_m'].clip(lower=0)
    df['elev_rain_ratio']         = df['rainfall_7d_mm'] / (df['elev_clip'] + 1)
    df['low_elev_flag']           = (df['elevation_m'] < 30).astype(int)
    df['infra_socio']             = df['infrastructure_score'] * df['socioeconomic_status_index']
    df['water_veg_balance']       = df['ndwi'] - df['ndvi']
    df['ndvi_ndwi_product']       = df['ndvi'] * df['ndwi']
    df['ndwi_sq']                 = df['ndwi'] ** 2
    df['drainage_x_rain']         = df['drainage_index'] * df['rainfall_7d_mm']
    df['bad_drainage_rain']       = (df['drainage_index'] < 0.35).astype(int) * df['rainfall_7d_mm']
    df['urban_runoff']            = df['built_up_percent'] * df['rainfall_7d_mm'] / 100
    df['evac_hosp_sum']           = df['nearest_hospital_km'] + df['nearest_evac_km']
    df['max_dist_help']           = df[['nearest_hospital_km', 'nearest_evac_km']].max(axis=1)
    df['pop_x_rain']              = df['population_density_per_km2'] * df['rainfall_7d_mm']
    df['pop_x_flood']             = df['population_density_per_km2'] * df['historical_flood_count']
    df['extreme_x_rain']          = df['extreme_weather_index'] * df['rainfall_7d_mm']
    df['extreme_x_flood']         = df['extreme_weather_index'] * df['historical_flood_count']
    df['extreme_x_monthly']       = df['extreme_weather_index'] * df['monthly_rainfall_mm']
    df['seasonal_rain']           = df['seasonal_index'] * df['rainfall_7d_mm']
    df['seasonal_extreme']        = df['seasonal_index'] * df['extreme_weather_index']
    df['terrain_rain']            = df['terrain_roughness_index'] * df['rainfall_7d_mm']
    df['inundation_x_rain']       = df['inundation_area_sqm'] * df['rainfall_7d_mm']
    df['log_inundation_x_extreme']= df['log_inundation'] * df['extreme_weather_index']
    df['composite_vuln']          = (
        df['rainfall_7d_mm'] * 0.3 + df['historical_flood_count'] * 15.0 +
        df['extreme_weather_index'] * 50.0 + (1-df['drainage_index']) * 30.0 +
        df['built_up_percent'] * 0.10)
    return df

print("Engineering features...")
train = engineer_features(train)
test  = engineer_features(test)

CAT_COLS = ['district','landcover','soil_type','water_supply','electricity',
            'road_quality','urban_rural','water_presence_flag',
            'flood_occurrence_current_event','is_good_to_live',
            'reason_not_good_to_live','place_name']
DROP_COLS = ['record_id','gen_date','generation_date','is_synthetic',TARGET]

all_data = pd.concat([train, test], axis=0, ignore_index=True)
for col in CAT_COLS:
    if col in all_data.columns:
        le = LabelEncoder()
        all_data[col] = le.fit_transform(all_data[col].astype(str).fillna('missing'))

n_train   = len(train)
train_enc = all_data.iloc[:n_train].copy()
test_enc  = all_data.iloc[n_train:].copy()
train_enc[TARGET] = train[TARGET].values

EXCLUDE      = set(DROP_COLS + [TARGET])
feature_cols = [c for c in train_enc.columns if c not in EXCLUDE]
X      = train_enc[feature_cols].copy()
y      = train_enc[TARGET].copy()
X_test = test_enc[feature_cols].copy()
medians = X.median(numeric_only=True)
X       = X.fillna(medians)
X_test  = X_test.fillna(medians)

# ─── Feature Caching (Target Encoding) ─────────────────
TE_CACHE_FILE = os.path.join(CHECKPOINT_DIR, "features_cached.npz")
if os.path.exists(TE_CACHE_FILE):
    print("Loading cached features...")
    data = np.load(TE_CACHE_FILE)
    X_arr = data['X_arr']
    X_test_arr = data['X_test_arr']
    y_arr = data['y_arr']
else:
    from sklearn.model_selection import KFold as KFold_TE
    def fold_safe_te(X_df, y_s, X_te_df, col, smooth, gm, n_folds=20, seed=SEED):
        tr_te = np.zeros(len(X_df))
        kf = KFold_TE(n_splits=n_folds, shuffle=True, random_state=seed)
        for tr_idx, val_idx in kf.split(X_df):
            s = pd.DataFrame({'k': X_df.iloc[tr_idx][col].values, 'y': y_s.iloc[tr_idx].values}).groupby('k')['y'].agg(['mean','count'])
            s['enc'] = (s['mean']*s['count'] + gm*smooth) / (s['count']+smooth)
            tr_te[val_idx] = X_df.iloc[val_idx][col].map(s['enc']).fillna(gm).values
        s_all = pd.DataFrame({'k': X_df[col].values, 'y': y_s.values}).groupby('k')['y'].agg(['mean','count'])
        s_all['enc'] = (s_all['mean']*s_all['count'] + gm*smooth) / (s_all['count']+smooth)
        return tr_te, X_te_df[col].map(s_all['enc']).fillna(gm).values

    gm = y.mean()
    print(f"Computing fold-safe TEs + interactions (20 folds)...")
    te_store = {}
    for col, smooth in [('place_name',5),('district',10),('soil_type',10),
                        ('landcover',10),('road_quality',10),('flood_occurrence_current_event',10)]:
        tr_te, te_te = fold_safe_te(X, y, X_test, col, smooth, gm, 20, SEED)
        X[f'{col}_te'] = tr_te; X_test[f'{col}_te'] = te_te
        te_store[col]  = (tr_te, te_te)

    d_tr, d_te = te_store['district']
    f_tr, f_te = te_store['flood_occurrence_current_event']
    s_tr, s_te = te_store['soil_type']

    for feat, tr_v, te_v, col_name in [
        ('rainfall_7d_mm',       d_tr, d_te, 'dist_te_x_rain'),
        ('extreme_weather_index',d_tr, d_te, 'dist_te_x_extreme'),
        ('historical_flood_count',d_tr,d_te,'dist_te_x_flood'),
        ('rainfall_7d_mm',       f_tr, f_te, 'fl_te_x_rain'),
        ('extreme_weather_index',f_tr, f_te, 'fl_te_x_extreme'),
        ('drainage_index',       d_tr, d_te, 'dist_te_x_drainage'),
        ('river_clip',           d_tr, d_te, 'dist_te_x_river'),
        ('elevation_m',          d_tr, d_te, 'dist_te_x_elev'),
        ('historical_flood_count',f_tr,f_te,'fl_te_x_flood'),
        ('drainage_index',       f_tr, f_te, 'fl_te_x_drainage'),
        ('rainfall_7d_mm',       s_tr, s_te, 'soil_te_x_rain'),
        ('extreme_weather_index',s_tr, s_te, 'soil_te_x_extreme'),
    ]:
        X[col_name]      = tr_v * X[feat]
        X_test[col_name] = te_v * X_test[feat]

    X_arr      = X.values.astype(np.float32)
    X_test_arr = X_test.values.astype(np.float32)
    y_arr      = y.values.astype(np.float32)
    np.savez(TE_CACHE_FILE, X_arr=X_arr, X_test_arr=X_test_arr, y_arr=y_arr)
    print(f"Features engineered and cached! Count: {X_arr.shape[1]}")

print(f"Feature count: {X_arr.shape[1]}")
print(f"Train samples: {X_arr.shape[0]}, Test samples: {X_test_arr.shape[0]}")

# ─── 100-Fold Training Engine ─────────────────────────
TOTAL_FOLDS = N_FOLDS * N_REPEATS

def train_and_cache(name, train_fn):
    """Train a single model config across 100 folds with crash-safe checkpointing."""
    oof_path  = os.path.join(CHECKPOINT_DIR, f"{name}_oof.npy")
    test_path = os.path.join(CHECKPOINT_DIR, f"{name}_test.npy")
    
    if os.path.exists(oof_path) and os.path.exists(test_path):
        print(f"[SKIP] '{name}' already complete.")
        return
    
    print(f"\n{'='*60}")
    print(f">>> Model: {name} | {TOTAL_FOLDS} folds")
    print(f"{'='*60}")
    start_time = time.time()
    
    oof_preds_sum = np.zeros(len(X_arr))
    oof_counts    = np.zeros(len(X_arr))
    test_preds    = np.zeros(len(X_test_arr))
    
    rkf = RepeatedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)
    
    for fold, (tr_i, val_i) in enumerate(rkf.split(X_arr)):
        X_tr, y_tr = X_arr[tr_i], y_arr[tr_i]
        X_va, y_va = X_arr[val_i], y_arr[val_i]
        
        preds_va, preds_te = train_fn(X_tr, y_tr, X_va, y_va, X_test_arr)
        oof_preds_sum[val_i] += preds_va
        oof_counts[val_i]    += 1
        test_preds           += preds_te / TOTAL_FOLDS
        
        if (fold + 1) % 20 == 0:
            elapsed = (time.time() - start_time) / 60
            print(f"  Repeat {(fold+1)//20}/{N_REPEATS} done | {elapsed:.1f} min elapsed")
        
        gc.collect()
    
    final_oof = oof_preds_sum / oof_counts
    rmse_total = np.sqrt(mean_squared_error(y_arr, final_oof))
    mins = (time.time() - start_time) / 60
    print(f"<<< {name} | RMSE = {rmse_total:.5f} | Time = {mins:.1f} min")
    
    np.save(oof_path, final_oof)
    np.save(test_path, test_preds)


# ─── Model Trainer Factories ──────────────────────────

def get_lgb_trainer(params):
    def train_fn(X_tr, y_tr, X_va, y_va, X_te):
        m = lgb.train(params, lgb.Dataset(X_tr, y_tr),
                      5000, valid_sets=[lgb.Dataset(X_va, y_va)],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        return m.predict(X_va), m.predict(X_te)
    return train_fn

def get_cat_trainer(params):
    def train_fn(X_tr, y_tr, X_va, y_va, X_te):
        m = CatBoostRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True, verbose=0)
        return m.predict(X_va), m.predict(X_te)
    return train_fn

def get_xgb_trainer(params):
    def train_fn(X_tr, y_tr, X_va, y_va, X_te):
        dtr = xgb.DMatrix(X_tr, label=y_tr)
        dva = xgb.DMatrix(X_va, label=y_va)
        dte = xgb.DMatrix(X_te)
        m = xgb.train(params, dtr, num_boost_round=5000,
                      evals=[(dva, 'val')], early_stopping_rounds=150, verbose_eval=False)
        return m.predict(dva), m.predict(dte)
    return train_fn

def get_hgb_trainer(params):
    def train_fn(X_tr, y_tr, X_va, y_va, X_te):
        m = HistGradientBoostingRegressor(**params)
        m.fit(X_tr, y_tr)
        return m.predict(X_va), m.predict(X_te)
    return train_fn


# ══════════════════════════════════════════════════════════
#  THE COLOSSUS MODEL GRID
# ══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  COMMENCING V20 COLOSSUS TRAINING QUEUE")
print(f"  Total seeds: {N_SEEDS} | Folds per model: {TOTAL_FOLDS}")
print(f"{'='*70}")

model_count = 0

# ──────────────────────────────────────────────────────────
#  PHASE 1: HistGradientBoosting (fastest, warms up the checkpoints)
# ──────────────────────────────────────────────────────────
print("\n[PHASE 1] HistGradientBoosting Models")

hgb_configs = [
    ("hgb_d4",  dict(max_iter=800, learning_rate=0.03, max_depth=4,  early_stopping=True, validation_fraction=0.1)),
    ("hgb_d6",  dict(max_iter=800, learning_rate=0.03, max_depth=6,  early_stopping=True, validation_fraction=0.1)),
    ("hgb_d8",  dict(max_iter=800, learning_rate=0.03, max_depth=8,  early_stopping=True, validation_fraction=0.1)),
    ("hgb_d10", dict(max_iter=800, learning_rate=0.02, max_depth=10, early_stopping=True, validation_fraction=0.1)),
]

for cfg_name, cfg_params in hgb_configs:
    for s in ALL_SEEDS:
        p = cfg_params.copy()
        p['random_state'] = s
        train_and_cache(f"{cfg_name}_s{s}", get_hgb_trainer(p))
        model_count += 1

# ──────────────────────────────────────────────────────────
#  PHASE 2: LightGBM (moderate speed, high diversity)
# ──────────────────────────────────────────────────────────
print("\n[PHASE 2] LightGBM Models")

lgb_base = {'metric':'rmse','n_jobs':-1,'verbose':-1}

lgb_configs = [
    ("lgb_mae",  {**lgb_base, 'objective':'regression_l1','learning_rate':0.03,
                  'num_leaves':63,'min_child_samples':20,'feature_fraction':0.7,
                  'bagging_fraction':0.8,'bagging_freq':5,'reg_alpha':0.1,'reg_lambda':1.0}),
    ("lgb_rmse", {**lgb_base, 'objective':'regression','learning_rate':0.03,
                  'num_leaves':63,'min_child_samples':20,'feature_fraction':0.7,
                  'bagging_fraction':0.8,'bagging_freq':5,'reg_alpha':0.1,'reg_lambda':1.0}),
    ("lgb_deep", {**lgb_base, 'objective':'regression','learning_rate':0.02,
                  'num_leaves':255,'min_child_samples':30,'feature_fraction':0.6,
                  'bagging_fraction':0.75,'bagging_freq':5,'reg_alpha':0.2,'reg_lambda':2.0}),
    ("lgb_wide", {**lgb_base, 'objective':'regression','learning_rate':0.025,
                  'num_leaves':127,'min_child_samples':25,'feature_fraction':0.65,
                  'bagging_fraction':0.8,'bagging_freq':5,'reg_alpha':0.15,'reg_lambda':1.5}),
    ("lgb_huber",{**lgb_base, 'objective':'huber','learning_rate':0.03,
                  'num_leaves':63,'min_child_samples':20,'feature_fraction':0.7,
                  'bagging_fraction':0.8,'bagging_freq':5,'reg_alpha':0.1,'reg_lambda':1.0}),
    ("lgb_dart", {**lgb_base, 'objective':'regression','boosting_type':'dart','learning_rate':0.05,
                  'num_leaves':63,'min_child_samples':20,'feature_fraction':0.7,
                  'bagging_fraction':0.8,'bagging_freq':5,'reg_alpha':0.1,'reg_lambda':1.0}),
]

for cfg_name, cfg_params in lgb_configs:
    for s in ALL_SEEDS:
        p = cfg_params.copy()
        p['seed'] = s
        train_and_cache(f"{cfg_name}_s{s}", get_lgb_trainer(p))
        model_count += 1

# ──────────────────────────────────────────────────────────
#  PHASE 3: XGBoost (moderate speed, algorithmic diversity)
# ──────────────────────────────────────────────────────────
if HAS_XGB:
    print("\n[PHASE 3] XGBoost Models")
    
    xgb_configs = [
        ("xgb_sq_d6",  {'objective':'reg:squarederror','eval_metric':'rmse','learning_rate':0.03,
                         'max_depth':6,'subsample':0.8,'colsample_bytree':0.7,
                         'alpha':0.1,'lambda':1.0,'tree_method':'hist'}),
        ("xgb_sq_d8",  {'objective':'reg:squarederror','eval_metric':'rmse','learning_rate':0.03,
                         'max_depth':8,'subsample':0.8,'colsample_bytree':0.7,
                         'alpha':0.1,'lambda':1.0,'tree_method':'hist'}),
        ("xgb_hub_d6", {'objective':'reg:pseudohubererror','eval_metric':'rmse','learning_rate':0.03,
                         'max_depth':6,'subsample':0.8,'colsample_bytree':0.7,
                         'alpha':0.1,'lambda':1.0,'tree_method':'hist'}),
        ("xgb_hub_d8", {'objective':'reg:pseudohubererror','eval_metric':'rmse','learning_rate':0.03,
                         'max_depth':8,'subsample':0.8,'colsample_bytree':0.7,
                         'alpha':0.1,'lambda':1.0,'tree_method':'hist'}),
    ]
    
    for cfg_name, cfg_params in xgb_configs:
        for s in ALL_SEEDS:
            p = cfg_params.copy()
            p['seed'] = s
            train_and_cache(f"{cfg_name}_s{s}", get_xgb_trainer(p))
            model_count += 1

# ──────────────────────────────────────────────────────────
#  PHASE 4: CatBoost (slowest, most powerful — the heart of Colossus)
# ──────────────────────────────────────────────────────────
print("\n[PHASE 4] CatBoost Models (Heavy Compute)")

cat_base = dict(iterations=5000, l2_leaf_reg=3.0, min_data_in_leaf=15,
                subsample=0.8, colsample_bylevel=0.7,
                task_type='CPU', verbose=0,
                eval_metric='RMSE', early_stopping_rounds=150)

cat_configs = [
    ("cat_d3",  {**cat_base, 'depth':3,  'learning_rate':0.03}),
    ("cat_d4",  {**cat_base, 'depth':4,  'learning_rate':0.03}),
    ("cat_d5",  {**cat_base, 'depth':5,  'learning_rate':0.03}),
    ("cat_d6",  {**cat_base, 'depth':6,  'learning_rate':0.03}),
    ("cat_d7",  {**cat_base, 'depth':7,  'learning_rate':0.03}),
    ("cat_d8",  {**cat_base, 'depth':8,  'learning_rate':0.03}),
    ("cat_d10", {**cat_base, 'depth':10, 'learning_rate':0.03}),
    ("cat_d12", {**cat_base, 'depth':12, 'learning_rate':0.03}),
    # Slower learning rate variants for stability
    ("cat_d6_slow",  {**cat_base, 'depth':6,  'learning_rate':0.01, 'iterations':10000}),
    ("cat_d8_slow",  {**cat_base, 'depth':8,  'learning_rate':0.01, 'iterations':10000}),
    ("cat_d10_slow", {**cat_base, 'depth':10, 'learning_rate':0.01, 'iterations':10000}),
    # Higher regularization variants
    ("cat_d6_reg",   {**cat_base, 'depth':6,  'learning_rate':0.03, 'l2_leaf_reg':10.0}),
    ("cat_d8_reg",   {**cat_base, 'depth':8,  'learning_rate':0.03, 'l2_leaf_reg':10.0}),
]

for cfg_name, cfg_params in cat_configs:
    for s in ALL_SEEDS:
        p = cfg_params.copy()
        p['random_seed'] = s
        train_and_cache(f"{cfg_name}_s{s}", get_cat_trainer(p))
        model_count += 1

# ─── Summary ──────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  ALL {model_count} V20 COLOSSUS MODELS COMPLETED!")
print(f"  Run `python v20_stacker.py` to compile the final submission.")
print(f"{'='*70}")
