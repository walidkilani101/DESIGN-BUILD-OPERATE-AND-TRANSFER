"""
DBOT Phase 1: TOPSIS Category Performance Ranking
=================================================
Using the Real-World DataCo Smart Supply Chain Dataset.
Evaluates 50+ Product Categories across 5 distinct supply chain criteria.
Applies mathematical vector normalization to score and rank them.
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH = r"c:\Users\samsung\Desktop\DESIGN-BUILD-OPERATE-AND-TRANSFER\archive (1)\DataCoSupplyChainDataset.csv"
OUT_DIR = r"c:\Users\samsung\Desktop\DESIGN-BUILD-OPERATE-AND-TRANSFER\Supply-Chain-Resilience-Simulation\outputs"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("DBOT PHASE 1: TOPSIS CATEGORY RANKING (DataCo Dataset)")
print("=" * 70)

# ─── 1. Load Data ─────────────────────────────────────────────────────────────
print("Loading DataCo dataset (180k+ rows)...")
df = pd.read_csv(DATA_PATH, encoding='latin1')
print(f"Loaded {len(df)} orders.")

# ─── 2. Calculate Criteria per Category ───────────────────────────────────────
print("\nCalculating operational KPIs per Category...")
# 1. average profit ratio
# 2. average delivery delay = real shipping days - scheduled shipping days
# 3. total sales
# 4. average discount rate
# 5. on-time rate = 1 - mean(Late_delivery_risk) (since 1 means late)

df['Delivery_Delay'] = df['Days for shipping (real)'] - df['Days for shipment (scheduled)']

kpis = df.groupby('Category Name').agg(
    Profit_Ratio=('Order Item Profit Ratio', 'mean'),
    Delivery_Delay=('Delivery_Delay', 'mean'),
    Total_Sales=('Sales', 'sum'),
    Discount_Rate=('Order Item Discount Rate', 'mean'),
    Late_Risk=('Late_delivery_risk', 'mean'),
    Order_Count=('Order Id', 'count')
).reset_index()

# Only keep categories with a statistically meaningful volume (e.g., > 100 orders)
kpis = kpis[kpis['Order_Count'] >= 100].copy()
kpis['On_Time_Rate'] = 1.0 - kpis['Late_Risk']
kpis = kpis.drop(columns=['Late_Risk', 'Order_Count'])

# Ensure no NaNs
kpis = kpis.fillna(0)
print(f"Aggregated {len(kpis)} distinct product categories.")

# ─── 3. TOPSIS Implementation ─────────────────────────────────────────────────
print("\nRunning TOPSIS Algorithm...")

# CRITERIA DEFINITION
# [Profit_Ratio, Delivery_Delay, Total_Sales, Discount_Rate, On_Time_Rate]
criteria_cols = ['Profit_Ratio', 'Delivery_Delay', 'Total_Sales', 'Discount_Rate', 'On_Time_Rate']
signs = np.array([1, -1, 1, -1, 1])  # 1 for maximize, -1 for minimize

# Base Weights (Equal priority)
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

def run_topsis(data, cols, w, s):
    # Step 1: Extract matrix
    X = data[cols].values
    
    # Step 2: Vector Normalization
    norms = np.sqrt((X**2).sum(axis=0))
    # Prevent div-by-zero
    norms[norms == 0] = 1.0
    R = X / norms
    
    # Step 3: Weighted Normalized Matrix
    V = R * w
    
    # Step 4: Ideal Best (V+) and Ideal Worst (V-)
    V_plus = np.zeros(len(cols))
    V_minus = np.zeros(len(cols))
    
    for i in range(len(cols)):
        if s[i] == 1:  # Maximize
            V_plus[i] = V[:, i].max()
            V_minus[i] = V[:, i].min()
        else:          # Minimize
            V_plus[i] = V[:, i].min()
            V_minus[i] = V[:, i].max()
            
    # Step 5: Separation measures (Euclidean distance)
    S_plus = np.sqrt(((V - V_plus)**2).sum(axis=1))
    S_minus = np.sqrt(((V - V_minus)**2).sum(axis=1))
    
    # Step 6: Relative Closeness to Ideal (TOPSIS Score)
    # Prevent division by zero if both are 0 (unlikely)
    denom = S_plus + S_minus
    denom[denom == 0] = 1.0
    C_star = S_minus / denom
    
    return C_star

kpis['TOPSIS_Score'] = run_topsis(kpis, criteria_cols, weights, signs)
kpis['Rank'] = kpis['TOPSIS_Score'].rank(ascending=False).astype(int)
kpis = kpis.sort_values('Rank')

# ─── 4. Sensitivity Analysis ──────────────────────────────────────────────────
print("Running Sensitivity Analysis (4 Weighting Scenarios)...")
scenarios = {
    'Equal_Weights':      [0.20, 0.20, 0.20, 0.20, 0.20],
    'Profit_Focused':     [0.50, 0.10, 0.20, 0.10, 0.10],
    'Speed_Focused':      [0.10, 0.40, 0.10, 0.10, 0.30],
    'Volume_Focused':     [0.10, 0.10, 0.60, 0.10, 0.10]
}

sensitivity_df = kpis[['Category Name']].copy()
for sc_name, sc_weights in scenarios.items():
    sc_scores = run_topsis(kpis, criteria_cols, np.array(sc_weights), signs)
    sensitivity_df[f'{sc_name}_Rank'] = pd.Series(sc_scores).rank(ascending=False).values.astype(int)

sensitivity_df = sensitivity_df.sort_values('Equal_Weights_Rank')

# ─── 5. Export and Charting ───────────────────────────────────────────────────
print("\nExporting results...")
# Main Ranking Table
out_main = os.path.join(OUT_DIR, "topsis_category_ranking.csv")
kpis.to_csv(out_main, index=False)

# Sensitivity Analysis Table
out_sens = os.path.join(OUT_DIR, "topsis_sensitivity_analysis.csv")
sensitivity_df.to_csv(out_sens, index=False)

# Chart 1: Top 15 Categories by TOPSIS Score
top15 = kpis.head(15).sort_values('TOPSIS_Score', ascending=True) # Ascending for horizontal bar
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(top15['Category Name'], top15['TOPSIS_Score'], color='#2ecc71', edgecolor='white')
ax.set_xlabel('TOPSIS Score (Closeness to Ideal)')
ax.set_title('Top 15 Product Categories by Operational Performance (DataCo)', fontsize=14, fontweight='bold')
for i, v in enumerate(top15['TOPSIS_Score']):
    ax.text(v + 0.005, i, f"{v:.3f}", va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "chart_topsis_top15_categories.png"), dpi=200)

print(f"Saved: {out_main}")
print(f"Saved: {out_sens}")
print(f"Saved: chart_topsis_top15_categories.png")

print("\n" + "=" * 70)
print("PHASE 1 COMPLETE")
print("=" * 70)
