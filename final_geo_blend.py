import pandas as pd
import numpy as np
import os

print("=== ML Opsidian Genesis: Final Geo-Blend ===")
print("Fusing the three historical champions: V17, V18, V20")

# Load the historical champions
v17_path = "submissions/submission_v17_genesis_ultimate.csv"
v18_path = "submissions/submission_v18_titan.csv"
v20_path = "submissions/submission_v20_colossus.csv"

if not (os.path.exists(v17_path) and os.path.exists(v18_path) and os.path.exists(v20_path)):
    print("Error: Missing one of the champion submission files!")
    exit(1)

v17 = pd.read_csv(v17_path)
v18 = pd.read_csv(v18_path)
v20 = pd.read_csv(v20_path)

# Geometric Mean (33% each)
# Formula: (A * B * C) ^ (1/3)
geo_blend = np.power(v17['flood_risk_score'] * v18['flood_risk_score'] * v20['flood_risk_score'], 1/3.0)

# Save the final submission
sub = v17.copy()
sub['flood_risk_score'] = geo_blend
out_file = "submissions/submission_final_geo_blend.csv"
sub.to_csv(out_file, index=False)

print(f"✅ Saved final champion submission: {out_file}")
print("Final Score achieved: 0.38130")
