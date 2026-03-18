"""
DBOT Phase 4: State-of-the-Art Business Impact Visuals
======================================================
1. Business Strategy Quadrant (BCG-Matrix style):
   Maps 'Average Profit Margin' vs 'Delivery Risk' across Categories.
   Bubble size represents 'Total Sales Volume'.
   Provides explicit strategic actions (Scale, Optimize Logistics, Discontinue).
   
2. Financial Impact Waterfall:
   Translates the ML Confusion Matrix (89% Precision) into pure Dollar Savings.
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = r"c:\Users\samsung\Desktop\DESIGN-BUILD-OPERATE-AND-TRANSFER\archive (1)\DataCoSupplyChainDataset.csv"
OUT_DIR = r"c:\Users\samsung\Desktop\DESIGN-BUILD-OPERATE-AND-TRANSFER\Supply-Chain-Resilience-Simulation\outputs"

print("=" * 80)
print("DBOT PHASE 4: GENERATING BUSINESS IMPACT PLOTS")
print("=" * 80)

# ─── 1. Load Data ─────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, encoding='latin1')

# ─── 2. Business Strategy Quadrant (Risk vs Profit) ─────────────────────────
print("Generating Business Strategy Quadrant...")
# Calculate Category KPIs
cat_kpis = df.groupby('Category Name').agg({
    'Order Item Profit Ratio': 'mean',
    'Late_delivery_risk': 'mean',
    'Sales': 'sum'
}).reset_index()

# Filter out very small categories to keep plot clean
cat_kpis = cat_kpis[cat_kpis['Sales'] > 100000]

median_risk = cat_kpis['Late_delivery_risk'].median()
median_profit = cat_kpis['Order Item Profit Ratio'].median()

fig, ax = plt.subplots(figsize=(12, 8))
# Draw Quadrants
ax.axhline(median_profit, color='gray', linestyle='--', alpha=0.5)
ax.axvline(median_risk, color='gray', linestyle='--', alpha=0.5)

# Colored backgrounds for quadrants
ax.axvspan(xmin=0, xmax=median_risk, ymin=(median_profit - ax.get_ylim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0]) if ax.get_ylim()[1]!=ax.get_ylim()[0] else 0, ymax=1, color='#2ecc71', alpha=0.1) # Top Left: Scale
ax.axvspan(xmin=median_risk, xmax=1, ymin=(median_profit - ax.get_ylim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0]) if ax.get_ylim()[1]!=ax.get_ylim()[0] else 0, ymax=1, color='#f1c40f', alpha=0.1) # Top Right: Optimize
ax.axvspan(xmin=0, xmax=median_risk, ymin=0, ymax=(median_profit - ax.get_ylim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0]) if ax.get_ylim()[1]!=ax.get_ylim()[0] else 0, color='#3498db', alpha=0.1) # Bottom Left: Volume
ax.axvspan(xmin=median_risk, xmax=1, ymin=0, ymax=(median_profit - ax.get_ylim()[0])/(ax.get_ylim()[1]-ax.get_ylim()[0]) if ax.get_ylim()[1]!=ax.get_ylim()[0] else 0, color='#e74c3c', alpha=0.1) # Bottom Right: Kill

scatter = ax.scatter(
    cat_kpis['Late_delivery_risk'], 
    cat_kpis['Order Item Profit Ratio'], 
    s=cat_kpis['Sales'] / 2000, # Normalize bubble size
    alpha=0.6, 
    c='#2c3e50',
    edgecolors='white',
    linewidth=1.5
)

# Annotate Top Categories
for idx, row in cat_kpis.sort_values('Sales', ascending=False).head(15).iterrows():
    ax.text(row['Late_delivery_risk'], row['Order Item Profit Ratio'] + 0.005, 
            row['Category Name'], fontsize=9, ha='center', va='bottom', fontweight='bold')

# Quadrant Labels
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax.text(median_risk - 0.01, median_profit + 0.05, '1. SCALE & INVEST\n(Low Risk, High Margin)', ha='right', va='center', bbox=props, color='#27ae60', fontweight='bold')
ax.text(median_risk + 0.01, median_profit + 0.05, '2. OPTIMIZE LOGISTICS\n(High Risk, High Margin)', ha='left', va='center', bbox=props, color='#d35400', fontweight='bold')
ax.text(median_risk - 0.01, median_profit - 0.05, '3. VOLUME DRIVERS\n(Low Risk, Low Margin)', ha='right', va='center', bbox=props, color='#2980b9', fontweight='bold')
ax.text(median_risk + 0.01, median_profit - 0.05, '4. OVERHAUL / KILL\n(High Risk, Low Margin)', ha='left', va='center', bbox=props, color='#c0392b', fontweight='bold')

ax.set_title("Strategic Action Quadrant: Category Risk vs. Profitability", fontsize=15, fontweight='bold', pad=20)
ax.set_xlabel("Average Late Delivery Risk (Probability)", fontsize=12)
ax.set_ylabel("Average Profit Margin (Ratio)", fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "chart_business_strategy_quadrant.png"), dpi=200)
print("Saved Business Strategy Quadrant to outputs/")

# ─── 3. Financial Waterfall Chart (ML Value Translation) ────────────────────
print("Generating Financial ROI Waterfall Chart...")
# Assume average penalty for late delivery is $15 per order.
# Total dataset has ~98,000 late orders (55% of 180k).
total_late = df['Late_delivery_risk'].sum()
cost_per_late = 15.0  # Consulting assumption
baseline_cost = total_late * cost_per_late

# Our ML Model (LR Gaussian) captured 54.26% (Recall) of these with 88.82% Precision.
recall = 0.5426
precision = 0.8882

# Cost avoidance: We successfully flag and proactively save 54% of late orders
# Assume proactive intervention costs $3 (e.g. expedited shipping upgrade)
intervention_cost = 3.0
true_positives = total_late * recall
false_positives = true_positives / precision - true_positives

gross_savings = true_positives * cost_per_late
intervention_spend = (true_positives + false_positives) * intervention_cost
net_savings = gross_savings - intervention_spend
final_cost = baseline_cost - net_savings

fig, ax = plt.subplots(figsize=(10, 6))

labels = ['Baseline Penalty', 'Gross Savings (ML True Positives)', 'Intervention Spend (Including False Positives)', 'Optimized Final Cost']
values = [baseline_cost, -gross_savings, intervention_spend, final_cost]

# Calculate starting points for the waterfall bars
starts = [0, baseline_cost - gross_savings, baseline_cost - gross_savings, 0]

colors = ['#c0392b', '#27ae60', '#e67e22', '#2980b9']

for i in range(len(labels)):
    ax.bar(labels[i], values[i], bottom=starts[i], color=colors[i], edgecolor='black', linewidth=1)
    # Add text
    val_text = f"${abs(values[i])/1000000:.1f}M"
    y_pos = starts[i] + values[i]/2 if i != 3 and i != 0 else values[i]/2
    ax.text(i, y_pos, val_text, ha='center', va='center', color='white', fontweight='bold', fontsize=11)

ax.set_title("Financial ROI of Delivery Risk ML Classifier\n(Translating the 89% Precision Matrix to Dollars)", fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel("Total Logistics Penalty & Intervention Cost ($)", fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "chart_financial_ml_waterfall.png"), dpi=200)
print("Saved Financial ML Waterfall Chart to outputs/")

print("=" * 80)
print("PHASE 4 COMPLETE: SOTA BUSINESS VISUALS GENERATED")
print("=" * 80)
