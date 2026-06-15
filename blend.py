import pandas as pd
import numpy as np

# Load your top 3 historical submissions
v17 = pd.read_csv("submissions/submission_v17_genesis_ultimate.csv")
v18 = pd.read_csv("submissions/submission_v18_titan.csv")
v20 = pd.read_csv("submissions/submission_v20_colossus.csv")

# Geometric Mean (33% each)
# Formula: (A * B * C) ^ (1/3)
geo_blend = np.power(v17['flood_risk_score'] * v18['flood_risk_score'] * v20['flood_risk_score'], 1/3.0)

# Save the final Hail Mary
sub = v17.copy()
sub['flood_risk_score'] = geo_blend
sub.to_csv("submissions/submission_final_geo_blend.csv", index=False)
print("Saved final Geo-Blend!")