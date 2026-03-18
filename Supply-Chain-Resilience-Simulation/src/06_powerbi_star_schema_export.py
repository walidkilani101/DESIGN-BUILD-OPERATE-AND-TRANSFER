"""
DBOT Phase 6: Enterprise BI Star Schema Export
==============================================
A single flat table is inefficient for tools like Power BI.
This script normalizes the dataset into an industry-standard Star Schema:

1. Fact_Orders (Central transactional data + ML Risk + Optimal Discounts)
2. Dim_Customers
3. Dim_Products (Includes TOPSIS Ranks)
4. Dim_Geography
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
OUT_DIR = r"c:\Users\samsung\Desktop\DESIGN-BUILD-OPERATE-AND-TRANSFER\Supply-Chain-Resilience-Simulation\outputs\star_schema"

# Create output directory for star schema
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 80)
print("DBOT PHASE 6: POWER BI STAR SCHEMA NORMALIZATION")
print("=" * 80)

# 1. Load Original Data
print("Loading original DataCo dataset...")
data = pd.read_csv(DATA_PATH, encoding='latin1')

# 2. Re-train ML Risk probabilities (Fast XGBoost)
print("Generating live ML Risk Probabilities...")
target = 'Late_delivery_risk'
num_cols = ['Order Item Quantity', 'Order Item Discount Rate', 'Sales per customer', 'Product Price', 'Days for shipment (scheduled)']
cat_cols = ['Shipping Mode', 'Order Region', 'Customer Segment', 'Category Name', 'Order Status']

# Dropping NaNs only for the ML features to score properly
ml_data = data[num_cols + cat_cols + [target]].dropna().copy()
y = ml_data[target].values
X_cat = pd.get_dummies(ml_data[cat_cols], drop_first=True)
X_num = ml_data[num_cols]
X = pd.concat([X_num, X_cat], axis=1).values

pt = PowerTransformer(method='yeo-johnson')
num_indices = [pd.concat([X_num, X_cat], axis=1).columns.get_loc(c) for c in num_cols]
X_scaled = X.copy()
X_scaled[:, num_indices] = pt.fit_transform(X[:, num_indices])

ratio = float(np.sum(y == 0)) / np.sum(y == 1)
clf = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, scale_pos_weight=ratio, random_state=42, n_jobs=-1, eval_metric='logloss')
clf.fit(X_scaled, y)

# Add ML probabilities back to original dataframe (aligning indices)
data['ML_Risk_Probability'] = np.nan
data.loc[ml_data.index, 'ML_Risk_Probability'] = clf.predict_proba(X_scaled)[:, 1]

# 3. Incorporate TOPSIS & Discount Ranks
topsis_df = pd.read_csv(r"c:\Users\samsung\Desktop\DESIGN-BUILD-OPERATE-AND-TRANSFER\Supply-Chain-Resilience-Simulation\outputs\topsis_category_ranking.csv")
topsis_df = topsis_df[['Category Name', 'Rank']].rename(columns={'Rank': 'TOPSIS_Rank'})

discount_df = pd.read_csv(r"c:\Users\samsung\Desktop\DESIGN-BUILD-OPERATE-AND-TRANSFER\Supply-Chain-Resilience-Simulation\outputs\discount_optimization_results.csv")
discount_df = discount_df[['Category', 'Optimal_Discount', 'Profit_Increase_%']].rename(columns={'Category': 'Category Name'})

# Merge into main dataset before splitting
data = data.merge(topsis_df, on='Category Name', how='left')
data = data.merge(discount_df, on='Category Name', how='left')

# ==========================================
# 4. SPLIT INTO STAR SCHEMA
# ==========================================
print("Splitting dataset into Dimension and Fact tables...")

# A. Dim_Customers
# Unique customers by ID
dim_customers = data[['Customer Id', 'Customer Fname', 'Customer Lname', 'Customer Segment', 'Customer Country', 'Customer State', 'Customer Zipcode']].drop_duplicates(subset=['Customer Id'])
dim_customers.to_csv(os.path.join(OUT_DIR, "Dim_Customers.csv"), index=False)
print(f" -> Exported Dim_Customers ({len(dim_customers)} rows)")

# B. Dim_Products
# Unique products by Product Card Id, including Category and TOPSIS Ranks
dim_products = data[['Product Card Id', 'Product Name', 'Category Name', 'Department Name', 'Product Price', 'TOPSIS_Rank', 'Optimal_Discount']].drop_duplicates(subset=['Product Card Id'])
dim_products.to_csv(os.path.join(OUT_DIR, "Dim_Products.csv"), index=False)
print(f" -> Exported Dim_Products ({len(dim_products)} rows)")

# C. Dim_Geography
# Unique order locations
data['Geo_Key'] = data['Order City'] + "_" + data['Order State'] + "_" + data['Order Country']
dim_geography = data[['Geo_Key', 'Order City', 'Order State', 'Order Country', 'Order Region', 'Market', 'Latitude', 'Longitude']].drop_duplicates(subset=['Geo_Key'])
dim_geography.to_csv(os.path.join(OUT_DIR, "Dim_Geography.csv"), index=False)
print(f" -> Exported Dim_Geography ({len(dim_geography)} rows)")

# D. Fact_Orders
# Central transactional data
fact_cols = [
    'Order Id', 'Order Item Id', 'Customer Id', 'Product Card Id', 'Geo_Key', 
    'order date (DateOrders)', 'shipping date (DateOrders)', 'Shipping Mode',
    'Order Status', 'Sales', 'Order Item Quantity', 'Order Item Discount Rate', 
    'Order Item Total', 'Order Profit Per Order', 'Late_delivery_risk', 'ML_Risk_Probability'
]
fact_orders = data[fact_cols]
fact_orders.to_csv(os.path.join(OUT_DIR, "Fact_Orders.csv"), index=False)
print(f" -> Exported Fact_Orders ({len(fact_orders)} rows)")

print("=" * 80)
print("SUCCESS: OCP Solutions Enterprise Star Schema Ready!")
print("=" * 80)
