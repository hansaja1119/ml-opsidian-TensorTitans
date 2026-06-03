"""
V20 Colossus Stacker

Dynamic auto-stacker. Run this at ANY time during training to compile
whatever checkpoints have finished into a valid submission.

Features:
- Loads all completed model checkpoints from colossus_checkpoints/
- Trains HuberRegressor (ε=1.35) meta-model
- Prints per-model weights and overall OOF RMSE
- Also generates a V18+V20 blend for maximum safety
"""

import numpy as np
import pandas as pd
import os
import glob
from sklearn.linear_model import HuberRegressor, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score

CHECKPOINT_DIR = "colossus_checkpoints"
DATA_DIR = "data"
SUBMISSIONS_DIR = "submissions"

os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

print("=" * 60)
print("  V20 COLOSSUS AUTO-STACKER")
print("=" * 60)

# ─── Load Checkpoints ─────────────────────────────────
oof_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*_oof.npy")))
if not oof_files:
    print("\nNo checkpoints found! Run `v20_colossus_solution.py` first.")
    exit(1)

# Filter out the feature cache file
model_names = [os.path.basename(f).replace("_oof.npy", "") for f in oof_files
               if "features_cached" not in f]
print(f"\nFound {len(model_names)} completed models:")

# Group by algorithm family for display
families = {}
for m in model_names:
    family = m.split('_')[0]
    families.setdefault(family, []).append(m)
for fam, members in sorted(families.items()):
    print(f"  {fam}: {len(members)} models")

# ─── Load Data ─────────────────────────────────────────
print("\nLoading data...")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
y_arr = train['flood_risk_score'].values
test_record_ids = test['record_id'].values

oof_stack = []
test_stack = []

for m in model_names:
    oof_stack.append(np.load(os.path.join(CHECKPOINT_DIR, f"{m}_oof.npy")))
    test_stack.append(np.load(os.path.join(CHECKPOINT_DIR, f"{m}_test.npy")))

oof_stack = np.column_stack(oof_stack)
test_stack = np.column_stack(test_stack)

print(f"Stacking matrix: {oof_stack.shape[0]} samples × {oof_stack.shape[1]} models")

# ─── Huber Stacker ─────────────────────────────────────
print("\n--- HuberRegressor (ε=1.35) ---")
huber = HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=2000)
huber.fit(oof_stack, y_arr)

huber_oof  = huber.predict(oof_stack)
huber_test = huber.predict(test_stack)

rmse_huber = np.sqrt(mean_squared_error(y_arr, huber_oof))
r2_huber   = r2_score(y_arr, huber_oof)

print(f"\n[COLOSSUS STACK] OOF RMSE = {rmse_huber:.5f} | R2 = {r2_huber:.4f}")

# Print top weighted models
weights = list(zip(model_names, huber.coef_))
weights.sort(key=lambda x: abs(x[1]), reverse=True)
print("\nTop 15 model weights:")
for name, w in weights[:15]:
    print(f"  {name:25s} : {w:+.4f}")

# ─── Save Standalone Submission ─────────────────────────
preds_clipped = np.clip(huber_test, 0, 1)
sub = pd.DataFrame({'record_id': test_record_ids, 'flood_risk_score': preds_clipped})
sub.to_csv(os.path.join(SUBMISSIONS_DIR, "submission_v20_colossus.csv"), index=False)
print(f"\n[OK] Saved: submissions/submission_v20_colossus.csv")

# ─── Blend with V18 Titan (if available) ────────────────
v18_path = os.path.join(SUBMISSIONS_DIR, "submission_v18_titan.csv")
if os.path.exists(v18_path):
    print("\n--- Blending with V18 Titan ---")
    v18 = pd.read_csv(v18_path)
    
    for w20 in [0.3, 0.4, 0.5, 0.6, 0.7]:
        w18 = 1.0 - w20
        blended = w18 * v18['flood_risk_score'].values + w20 * preds_clipped
        blend_name = f"submission_v20_blend_{int(w20*100)}v20_{int(w18*100)}v18.csv"
        sub_b = pd.DataFrame({'record_id': test_record_ids, 'flood_risk_score': np.clip(blended, 0, 1)})
        sub_b.to_csv(os.path.join(SUBMISSIONS_DIR, blend_name), index=False)
        print(f"  [OK] {blend_name}")
    
    print("\n  Recommended: Start with 50/50 blend (submission_v20_blend_50v20_50v18.csv)")
else:
    print(f"\n  V18 submission not found at {v18_path}. Skipping blend.")

print(f"\n{'='*60}")
print("  STACKING COMPLETE")
print(f"{'='*60}")
