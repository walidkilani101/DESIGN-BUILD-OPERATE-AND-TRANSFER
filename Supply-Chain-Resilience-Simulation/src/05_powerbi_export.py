"""
DBOT Phase 5: Power BI Master Export Pipeline
=============================================
This script bridges the gap between Data Science and Business Intelligence.
It scores the original DataCo dataset with our trained ML and OR models
and exports a unified `PowerBI_Master_Dataset.csv`.

This single file allows Power BI to dynamically slice and dice:
- XGBoost Risk Probabilities (by Country, Category, etc.)
- TOPSIS Ranks
- Optimal Discount Curves
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

DATA_PATH = r"c:\Users\samsung\Desktop\DESIGN-BUILD-OPERATE-AND-TRANSFER\archive (1)\DataCoSupplyChainDataset.csv"
OUT_DIR = r"c:\Users\samsung\Desktop\DESIGN-BUILD-OPERATE-AND-TRANSFER\Supply-Chain-Resilience-Simulation\outputs"

print("=" * 80)
print("DBOT PHASE 5: POWER BI MASTER EXPORT")
print("=" * 80)

# 1. Load Original Data
print("Loading original DataCo dataset...")
df = pd.read_csv(DATA_PATH, encoding='latin1')

# 2. Re-train the fast, optimal XGBoost Model to get Predictions
print("Generating live ML Risk Probabilities...")
target = 'Late_delivery_risk'
num_cols = ['Order Item Quantity', 'Order Item Discount Rate', 'Sales per customer', 'Product Price', 'Days for shipment (scheduled)']
cat_cols = ['Shipping Mode', 'Order Region', 'Customer Segment', 'Category Name', 'Order Status']

data = df[num_cols + cat_cols + [target]].dropna().copy()
y = data[target].values

X_cat = pd.get_dummies(data[cat_cols], drop_first=True)
X_num = data[num_cols]
X_df = pd.concat([X_num, X_cat], axis=1)
X = X_df.values

# Apply Gaussian Transform
pt = PowerTransformer(method='yeo-johnson')
num_indices = [X_df.columns.get_loc(c) for c in num_cols]
X_scaled = X.copy()
X_scaled[:, num_indices] = pt.fit_transform(X[:, num_indices])

# Train XGBoost with the known optimal hyperparameters we found earlier
ratio = float(np.sum(y == 0)) / np.sum(y == 1)
clf = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, scale_pos_weight=ratio, random_state=42, n_jobs=-1, eval_metric='logloss')
clf.fit(X_scaled, y)

# Score the entire dataset to feed Power BI
data['ML_Risk_Probability'] = clf.predict_proba(X_scaled)[:, 1]
data['ML_Risk_Flag'] = (data['ML_Risk_Probability'] > 0.5).astype(int)

# 3. Merge Phase 1 TOPSIS Ranks
print("Merging Operations Research (TOPSIS) ranks...")
topsis_df = pd.read_csv(os.path.join(OUT_DIR, "topsis_category_ranking.csv"))
# Keep only Rank for simplicity in Power BI
topsis_df = topsis_df[['Category Name', 'Rank']].rename(columns={'Rank': 'TOPSIS_Rank'})
data = data.merge(topsis_df, on='Category Name', how='left')

# 4. Merge Phase 3 Discount Optimizations
print("Merging Prescriptive Discount recommendations...")
discount_df = pd.read_csv(os.path.join(OUT_DIR, "discount_optimization_results.csv"))
discount_df = discount_df[['Category', 'Optimal_Discount', 'Profit_Increase_%']].rename(columns={'Category': 'Category Name'})
data = data.merge(discount_df, on='Category Name', how='left')

# 5. Export the Final Master Power BI Dataset
output_file = os.path.join(OUT_DIR, "PowerBI_Master_Dataset.csv")
print(f"Exporting {len(data)} rows to {output_file}...")
data.to_csv(output_file, index=False)

print("=" * 80)
print("SUCCESS: PowerBI_Master_Dataset.csv is ready for import!")
print("=" * 80)
