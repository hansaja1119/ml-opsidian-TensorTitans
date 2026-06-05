"""
V20 Colossus — Parallel Worker Script

Run this on any machine to train a specific subset of V20 models.
Checkpoints are saved to a local folder, which you then merge into 
the main machine's colossus_checkpoints/ folder using v20_merge.py.

Usage examples:

  # Train only deep CatBoost (recommended for VM2/Colab):
  python v20_worker.py --models cat_d8 cat_d10 cat_d12 cat_d8_slow cat_d10_slow

  # Train all CatBoost regularization variants:
  python v20_worker.py --models cat_d6_slow cat_d6_reg cat_d8_reg cat_d10_reg

  # Train with a specific checkpoint directory (e.g. on Colab):
  python v20_worker.py --models cat_d8 cat_d10 cat_d12 --checkpoint-dir /content/worker_checkpoints

Available model names:
  hgb_d4, hgb_d6, hgb_d8, hgb_d10
  lgb_mae, lgb_rmse, lgb_deep, lgb_wide, lgb_huber, lgb_dart
  xgb_sq_d6, xgb_sq_d8, xgb_hub_d6, xgb_hub_d8
  cat_d3, cat_d4, cat_d5, cat_d6, cat_d7, cat_d8, cat_d10, cat_d12
  cat_d6_slow, cat_d8_slow, cat_d10_slow
  cat_d6_reg, cat_d8_reg
"""

import argparse
import numpy as np
import pandas as pd
import os
import warnings
import time
import gc
warnings.filterwarnings('ignore')

from sklearn.model_selection import RepeatedKFold, KFold
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

# ─── Argument Parsing ─────────────────────────────────
parser = argparse.ArgumentParser(description='V20 Colossus Parallel Worker')
parser.add_argument('--models', nargs='+', required=True,
                    help='List of model config names to train (e.g. cat_d8 cat_d10 cat_d12)')
parser.add_argument('--checkpoint-dir', default='worker_checkpoints',
                    help='Directory to save checkpoints (default: worker_checkpoints)')
parser.add_argument('--data-dir', default='data',
                    help='Directory containing train.csv and test.csv')
parser.add_argument('--seeds', nargs='+', type=int,
                    default=[42, 142, 242, 342, 442, 542, 642, 742, 842, 942],
                    help='Seeds to train (default: all 10)')
args = parser.parse_args()

CHECKPOINT_DIR = args.checkpoint_dir
DATA_DIR       = args.data_dir
ALL_SEEDS      = args.seeds
MODELS_TO_RUN  = set(args.models)
SEED           = 42
N_FOLDS        = 20
N_REPEATS      = 5
TOTAL_FOLDS    = N_FOLDS * N_REPEATS

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
np.random.seed(SEED)

print("=" * 60)
print("  V20 COLOSSUS PARALLEL WORKER")
print(f"  Models     : {sorted(MODELS_TO_RUN)}")
print(f"  Seeds      : {ALL_SEEDS}")
print(f"  Checkpoint : {CHECKPOINT_DIR}/")
print(f"  CV         : {N_FOLDS} folds × {N_REPEATS} repeats = {TOTAL_FOLDS} total")
print("=" * 60)

# ─── Data Loading ─────────────────────────────────────
print("\nLoading data...")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
TARGET = 'flood_risk_score'
test_record_ids = test['record_id'].copy()
print(f"Train: {train.shape}, Test: {test.shape}")

# ─── Feature Engineering ──────────────────────────────
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
        df['extreme_weather_index'] * 50.0 + (1 - df['drainage_index']) * 30.0 +
        df['built_up_percent'] * 0.10)
    return df

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

# ─── Target Encoding ──────────────────────────────────
TE_CACHE = os.path.join(CHECKPOINT_DIR, "features_cached.npz")
if os.path.exists(TE_CACHE):
    print("Loading cached features...")
    d = np.load(TE_CACHE)
    X_arr = d['X_arr']; X_test_arr = d['X_test_arr']; y_arr = d['y_arr']
else:
    def fold_safe_te(X_df, y_s, X_te_df, col, smooth, gm):
        tr_te = np.zeros(len(X_df))
        kf = KFold(n_splits=20, shuffle=True, random_state=SEED)
        for tr_idx, val_idx in kf.split(X_df):
            s = pd.DataFrame({'k': X_df.iloc[tr_idx][col].values, 'y': y_s.iloc[tr_idx].values}).groupby('k')['y'].agg(['mean','count'])
            s['enc'] = (s['mean']*s['count'] + gm*smooth) / (s['count']+smooth)
            tr_te[val_idx] = X_df.iloc[val_idx][col].map(s['enc']).fillna(gm).values
        s_all = pd.DataFrame({'k': X_df[col].values, 'y': y_s.values}).groupby('k')['y'].agg(['mean','count'])
        s_all['enc'] = (s_all['mean']*s_all['count'] + gm*smooth) / (s_all['count']+smooth)
        return tr_te, X_te_df[col].map(s_all['enc']).fillna(gm).values

    gm = y.mean()
    print("Computing Target Encodings...")
    te_store = {}
    for col, smooth in [('place_name',5),('district',10),('soil_type',10),
                        ('landcover',10),('road_quality',10),('flood_occurrence_current_event',10)]:
        tr_te, te_te = fold_safe_te(X, y, X_test, col, smooth, gm)
        X[f'{col}_te'] = tr_te; X_test[f'{col}_te'] = te_te
        te_store[col] = (tr_te, te_te)

    d_tr,d_te = te_store['district']
    f_tr,f_te = te_store['flood_occurrence_current_event']
    s_tr,s_te = te_store['soil_type']
    for feat, tr_v, te_v, col_name in [
        ('rainfall_7d_mm',d_tr,d_te,'dist_te_x_rain'),('extreme_weather_index',d_tr,d_te,'dist_te_x_extreme'),
        ('historical_flood_count',d_tr,d_te,'dist_te_x_flood'),('rainfall_7d_mm',f_tr,f_te,'fl_te_x_rain'),
        ('extreme_weather_index',f_tr,f_te,'fl_te_x_extreme'),('drainage_index',d_tr,d_te,'dist_te_x_drainage'),
        ('river_clip',d_tr,d_te,'dist_te_x_river'),('elevation_m',d_tr,d_te,'dist_te_x_elev'),
        ('historical_flood_count',f_tr,f_te,'fl_te_x_flood'),('drainage_index',f_tr,f_te,'fl_te_x_drainage'),
        ('rainfall_7d_mm',s_tr,s_te,'soil_te_x_rain'),('extreme_weather_index',s_tr,s_te,'soil_te_x_extreme'),
    ]:
        X[col_name] = tr_v * X[feat]; X_test[col_name] = te_v * X_test[feat]

    X_arr = X.values.astype(np.float32)
    X_test_arr = X_test.values.astype(np.float32)
    y_arr = y.values.astype(np.float32)
    np.savez(TE_CACHE, X_arr=X_arr, X_test_arr=X_test_arr, y_arr=y_arr)

print(f"Features: {X_arr.shape[1]}")

# ─── Training Engine ──────────────────────────────────
rkf = RepeatedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)

def train_and_cache(name, train_fn):
    oof_path  = os.path.join(CHECKPOINT_DIR, f"{name}_oof.npy")
    test_path = os.path.join(CHECKPOINT_DIR, f"{name}_test.npy")
    if os.path.exists(oof_path) and os.path.exists(test_path):
        print(f"[SKIP] {name}")
        return
    print(f"\n>>> {name} | {TOTAL_FOLDS} folds")
    start = time.time()
    oof_sum = np.zeros(len(X_arr)); counts = np.zeros(len(X_arr)); test_p = np.zeros(len(X_test_arr))
    for fold, (tr_i, val_i) in enumerate(rkf.split(X_arr)):
        pv, pt = train_fn(X_arr[tr_i], y_arr[tr_i], X_arr[val_i], y_arr[val_i], X_test_arr)
        oof_sum[val_i] += pv; counts[val_i] += 1; test_p += pt / TOTAL_FOLDS
        if (fold + 1) % N_FOLDS == 0:
            cur = oof_sum / np.maximum(counts, 1)
            rmse = np.sqrt(mean_squared_error(y_arr, cur))
            print(f"  Repeat {(fold+1)//N_FOLDS}/{N_REPEATS} | RMSE={rmse:.5f} | {(time.time()-start)/60:.1f}min")
        gc.collect()
    final = oof_sum / counts
    print(f"<<< {name} | RMSE={np.sqrt(mean_squared_error(y_arr,final)):.5f} | {(time.time()-start)/60:.1f}min")
    np.save(oof_path, final); np.save(test_path, test_p)

# ─── Model Factories ──────────────────────────────────
def get_lgb_trainer(p):
    def fn(Xtr,ytr,Xva,yva,Xte):
        m=lgb.train(p,lgb.Dataset(Xtr,ytr),5000,valid_sets=[lgb.Dataset(Xva,yva)],callbacks=[lgb.early_stopping(150,verbose=False)])
        return m.predict(Xva),m.predict(Xte)
    return fn

def get_cat_trainer(p):
    def fn(Xtr,ytr,Xva,yva,Xte):
        m=CatBoostRegressor(**p); m.fit(Xtr,ytr,eval_set=(Xva,yva),use_best_model=True,verbose=0)
        return m.predict(Xva),m.predict(Xte)
    return fn

def get_xgb_trainer(p):
    def fn(Xtr,ytr,Xva,yva,Xte):
        m=xgb.train(p,xgb.DMatrix(Xtr,label=ytr),5000,evals=[(xgb.DMatrix(Xva,label=yva),'v')],early_stopping_rounds=150,verbose_eval=False)
        return m.predict(xgb.DMatrix(Xva)),m.predict(xgb.DMatrix(Xte))
    return fn

def get_hgb_trainer(p):
    def fn(Xtr,ytr,Xva,yva,Xte):
        m=HistGradientBoostingRegressor(**p); m.fit(Xtr,ytr)
        return m.predict(Xva),m.predict(Xte)
    return fn

# ─── Full Model Registry ──────────────────────────────
lgb_base = {'metric':'rmse','n_jobs':-1,'verbose':-1}
cat_base = dict(iterations=5000,l2_leaf_reg=3.0,min_data_in_leaf=15,subsample=0.8,
                colsample_bylevel=0.7,task_type='CPU',verbose=0,eval_metric='RMSE',early_stopping_rounds=150)

ALL_MODEL_CONFIGS = {
    # HGB
    **{f"hgb_d{d}": ('hgb', dict(max_iter=800,learning_rate=0.03 if d<=8 else 0.02,max_depth=d,early_stopping=True,validation_fraction=0.1))
       for d in [4,6,8,10]},
    # LGB
    "lgb_mae":   ('lgb', {**lgb_base,'objective':'regression_l1','learning_rate':0.03,'num_leaves':63,'min_child_samples':20,'feature_fraction':0.7,'bagging_fraction':0.8,'bagging_freq':5,'reg_alpha':0.1,'reg_lambda':1.0}),
    "lgb_rmse":  ('lgb', {**lgb_base,'objective':'regression',   'learning_rate':0.03,'num_leaves':63,'min_child_samples':20,'feature_fraction':0.7,'bagging_fraction':0.8,'bagging_freq':5,'reg_alpha':0.1,'reg_lambda':1.0}),
    "lgb_deep":  ('lgb', {**lgb_base,'objective':'regression',   'learning_rate':0.02,'num_leaves':255,'min_child_samples':30,'feature_fraction':0.6,'bagging_fraction':0.75,'bagging_freq':5,'reg_alpha':0.2,'reg_lambda':2.0}),
    "lgb_wide":  ('lgb', {**lgb_base,'objective':'regression',   'learning_rate':0.025,'num_leaves':127,'min_child_samples':25,'feature_fraction':0.65,'bagging_fraction':0.8,'bagging_freq':5,'reg_alpha':0.15,'reg_lambda':1.5}),
    "lgb_huber": ('lgb', {**lgb_base,'objective':'huber',        'learning_rate':0.03,'num_leaves':63,'min_child_samples':20,'feature_fraction':0.7,'bagging_fraction':0.8,'bagging_freq':5,'reg_alpha':0.1,'reg_lambda':1.0}),
    "lgb_dart":  ('lgb', {**lgb_base,'objective':'regression','boosting_type':'dart','learning_rate':0.05,'num_leaves':63,'min_child_samples':20,'feature_fraction':0.7,'bagging_fraction':0.8,'bagging_freq':5,'reg_alpha':0.1,'reg_lambda':1.0}),
    # XGB
    "xgb_sq_d6":  ('xgb', {'objective':'reg:squarederror',   'eval_metric':'rmse','learning_rate':0.03,'max_depth':6,'subsample':0.8,'colsample_bytree':0.7,'alpha':0.1,'lambda':1.0,'tree_method':'hist'}),
    "xgb_sq_d8":  ('xgb', {'objective':'reg:squarederror',   'eval_metric':'rmse','learning_rate':0.03,'max_depth':8,'subsample':0.8,'colsample_bytree':0.7,'alpha':0.1,'lambda':1.0,'tree_method':'hist'}),
    "xgb_hub_d6": ('xgb', {'objective':'reg:pseudohubererror','eval_metric':'rmse','learning_rate':0.03,'max_depth':6,'subsample':0.8,'colsample_bytree':0.7,'alpha':0.1,'lambda':1.0,'tree_method':'hist'}),
    "xgb_hub_d8": ('xgb', {'objective':'reg:pseudohubererror','eval_metric':'rmse','learning_rate':0.03,'max_depth':8,'subsample':0.8,'colsample_bytree':0.7,'alpha':0.1,'lambda':1.0,'tree_method':'hist'}),
    # CatBoost
    "cat_d3":       ('cat', {**cat_base,'depth':3, 'learning_rate':0.03}),
    "cat_d4":       ('cat', {**cat_base,'depth':4, 'learning_rate':0.03}),
    "cat_d5":       ('cat', {**cat_base,'depth':5, 'learning_rate':0.03}),
    "cat_d6":       ('cat', {**cat_base,'depth':6, 'learning_rate':0.03}),
    "cat_d7":       ('cat', {**cat_base,'depth':7, 'learning_rate':0.03}),
    "cat_d8":       ('cat', {**cat_base,'depth':8, 'learning_rate':0.03}),
    "cat_d10":      ('cat', {**cat_base,'depth':10,'learning_rate':0.03}),
    "cat_d12":      ('cat', {**cat_base,'depth':12,'learning_rate':0.03}),
    "cat_d6_slow":  ('cat', {**cat_base,'depth':6, 'learning_rate':0.01,'iterations':10000}),
    "cat_d8_slow":  ('cat', {**cat_base,'depth':8, 'learning_rate':0.01,'iterations':10000}),
    "cat_d10_slow": ('cat', {**cat_base,'depth':10,'learning_rate':0.01,'iterations':10000}),
    "cat_d6_reg":   ('cat', {**cat_base,'depth':6, 'learning_rate':0.03,'l2_leaf_reg':10.0}),
    "cat_d8_reg":   ('cat', {**cat_base,'depth':8, 'learning_rate':0.03,'l2_leaf_reg':10.0}),
}

# ─── Run only requested models ────────────────────────
print(f"\n{'='*60}")
print(f"  Training {len(MODELS_TO_RUN)} model configs × {len(ALL_SEEDS)} seeds")
print(f"{'='*60}")

skipped_names = []
for cfg_name in sorted(MODELS_TO_RUN):
    if cfg_name not in ALL_MODEL_CONFIGS:
        print(f"[WARN] Unknown model name: '{cfg_name}' — skipping")
        skipped_names.append(cfg_name)
        continue
    family, base_params = ALL_MODEL_CONFIGS[cfg_name]
    if family == 'xgb' and not HAS_XGB:
        print(f"[SKIP] {cfg_name} — XGBoost not installed")
        continue
    for s in ALL_SEEDS:
        p = base_params.copy()
        if   family == 'lgb': p['seed'] = s
        elif family == 'cat': p['random_seed'] = s
        elif family == 'xgb': p['seed'] = s
        elif family == 'hgb': p['random_state'] = s
        full_name = f"{cfg_name}_s{s}"
        if   family == 'lgb': train_and_cache(full_name, get_lgb_trainer(p))
        elif family == 'cat': train_and_cache(full_name, get_cat_trainer(p))
        elif family == 'xgb': train_and_cache(full_name, get_xgb_trainer(p))
        elif family == 'hgb': train_and_cache(full_name, get_hgb_trainer(p))

if skipped_names:
    print(f"\n[WARN] Unknown model names skipped: {skipped_names}")
print(f"\n{'='*60}")
print("  WORKER COMPLETE")
print(f"  Checkpoints saved to: {CHECKPOINT_DIR}/")
print(f"  Copy all *.npy files to the main machine's colossus_checkpoints/")
print(f"  Then run: python v20_stacker.py")
print(f"{'='*60}")
