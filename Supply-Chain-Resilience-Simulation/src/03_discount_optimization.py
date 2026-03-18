"""
DBOT Phase 3: Prescriptive Discount Optimization
================================================
Using the Real-World DataCo Smart Supply Chain Dataset.
Hybrid Machine Learning (Log-Log Elasticity Regression) + 
Operations Research (Constrained Scalar Optimization).
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import linregress
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = r"c:\Users\samsung\Desktop\DESIGN-BUILD-OPERATE-AND-TRANSFER\archive (1)\DataCoSupplyChainDataset.csv"
OUT_DIR = r"c:\Users\samsung\Desktop\DESIGN-BUILD-OPERATE-AND-TRANSFER\Supply-Chain-Resilience-Simulation\outputs"

print("=" * 70)
print("DBOT PHASE 3: LOG-LOG ELASTICITY & CONSTRAINED OPTIMIZATION (DataCo)")
print("=" * 70)

# ─── 1. Load Data ─────────────────────────────────────────────────────────────
print("Loading DataCo dataset...")
df = pd.read_csv(DATA_PATH, encoding='latin1')
print(f"Loaded {len(df)} orders.")

# ─── 2. Prepare Economic Variables ────────────────────────────────────────────
# Effective Price paid after discount rate.
# In DataCo, 'Order Item Product Price' is the unit base price, and 'Order Item Discount Rate' is the % off
df['Effective_Price'] = df['Order Item Product Price'] * (1 - df['Order Item Discount Rate'])

categories = df['Category Name'].dropna().unique()
print(f"Analyzing elasticity across {len(categories)} product categories.")

results = []
plot_data = {}

# ─── 3. Optimization Loop ─────────────────────────────────────────────────────
for cat in categories:
    cat_data = df[df['Category Name'] == cat]
    if len(cat_data) < 500: # Ensure large statistical sample
        continue
        
    avg_price = cat_data['Order Item Product Price'].mean()
    # Estimate Cost = Price - (Profit / Quantity)
    # Actually DataCo provides 'Order Item Total' (Revenue) and 'Order Profit Per Order' matches the item profit.
    # Cost = (Revenue - Profit) / Quantity
    cat_data['Estimated_Unit_Cost'] = (cat_data['Order Item Total'] - cat_data['Order Profit Per Order']) / cat_data['Order Item Quantity']
    avg_cost = cat_data['Estimated_Unit_Cost'].mean()
    current_avg_discount = cat_data['Order Item Discount Rate'].mean()
    
    # 2a. Data Science: Log-Log Regression for Elasticity
    valid_data = cat_data[(cat_data['Order Item Quantity'] > 0) & (cat_data['Effective_Price'] > 0)]
    
    ln_Q = np.log(valid_data['Order Item Quantity'])
    ln_P = np.log(valid_data['Effective_Price'])
    
    slope, intercept, r_value, p_value, std_err = linregress(ln_P, ln_Q)
    elasticity = slope  
    
    is_significant = p_value < 0.05
    
    alpha = np.exp(intercept)
    curr_eff_price = avg_price * (1 - current_avg_discount)
    curr_Q = valid_data['Order Item Quantity'].mean()
    profit_current = curr_Q * (curr_eff_price - avg_cost)
    
    # 2b. Operations Research: Maximize Profit
    def profit_function(d, a, elast, std_price, std_cost):
        eff_p = std_price * (1 - d)
        q = a * (eff_p ** elast)
        return q * (eff_p - std_cost)
        
    def objective(d):
        return -profit_function(d, alpha, elasticity, avg_price, avg_cost)

    if (not is_significant) or (elasticity >= 0) or (avg_price <= avg_cost):
        opt_discount = 0.0
        profit_at_opt = valid_data['Order Item Quantity'].mean() * (avg_price - avg_cost)
    else:
        # Bounded scalar optimization (0% to max current discount + 10%)
        # Cap at 40% discount max.
        res = minimize_scalar(objective, bounds=(0.0, 0.40), method='bounded')
        opt_discount = res.x
        profit_at_opt = -res.fun
    
    results.append({
        'Category': cat,
        'Significant': 'Yes' if is_significant else 'No',
        'Elasticity': elasticity,
        'P_Value': p_value,
        'Current_Discount': current_avg_discount,
        'Optimal_Discount': opt_discount,
        'Current_Profit': profit_current,
        'Optimal_Profit': profit_at_opt
    })
    
    if is_significant and elasticity < 0 and avg_price > avg_cost:
        d_range = np.linspace(0.0, 0.40, 100)
        p_curve = [profit_function(d, alpha, elasticity, avg_price, avg_cost) for d in d_range]
        q_curve = [alpha * ((avg_price * (1 - d)) ** elasticity) for d in d_range]
        plot_data[cat] = (d_range, q_curve, p_curve, current_avg_discount, profit_current, opt_discount, profit_at_opt)

# ─── 4. Summarize & Export ────────────────────────────────────────────────────
res_df = pd.DataFrame(results)
res_df['Profit_Increase_%'] = ((res_df['Optimal_Profit'] - res_df['Current_Profit']) / res_df['Current_Profit'].replace(0, 0.01)) * 100

out_csv = os.path.join(OUT_DIR, "discount_optimization_results.csv")
res_df.to_csv(out_csv, index=False)
print(f"Optimization table saved to {out_csv}")

# ─── 5. Microeconomics Charts ─────────────────────────────────────────────────
if plot_data:
    cats_to_plot = list(plot_data.keys())[:4]
    fig, axes = plt.subplots(len(cats_to_plot), 1, figsize=(10, 5 * len(cats_to_plot)))
    if len(cats_to_plot) == 1: axes = [axes]
    
    for ax, cat in zip(axes, cats_to_plot):
        d_range, q_curve, p_curve, curr_d, curr_p, opt_d, opt_p = plot_data[cat]
        
        ax.set_title(f"Microeconomic Curves: {cat}", fontsize=14, fontweight='bold', pad=15)
        
        color1 = '#2980b9'
        ax.set_xlabel('Discount Offered (%)', fontsize=12)
        ax.set_ylabel('Expected Demand', color=color1, fontsize=12)
        ax.plot(d_range * 100, q_curve, color=color1, linewidth=2.5)
        ax.tick_params(axis='y', labelcolor=color1)
        
        ax2 = ax.twinx()  
        color2 = '#27ae60'
        ax2.set_ylabel('Expected Total Profit', color=color2, fontsize=12)
        ax2.plot(d_range * 100, p_curve, color=color2, linewidth=3.0)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        ax2.axvline(x=opt_d * 100, color='#e74c3c', linestyle='--', alpha=0.8, linewidth=2)
        ax2.scatter(opt_d * 100, opt_p, color='#e74c3c', s=150, zorder=10)
        ax2.text((opt_d * 100) + 1, opt_p, f'Optimal:\n{opt_d:.1%} (${opt_p:.2f})', 
                 color='#e74c3c', fontweight='bold', fontsize=11)
        
        ax2.scatter(curr_d * 100, curr_p, color='black', s=100, zorder=9, marker='s')
        
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "chart_discount_microeconomics.png"), dpi=200)
    print("Saved Microeconomics Profit Curves chart.")
else:
    # Proof of Noise Chart if no significance
    print("Generating 'Proof of Noise' chart (No significant negative elasticity found).")
    demo_cat = res_df.iloc[0]['Category'] if len(res_df) > 0 else categories[0]
    cat_data = df[df['Category Name'] == demo_cat]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(cat_data['Order Item Discount Rate'] * 100, cat_data['Order Item Quantity'], alpha=0.1, color='#3498db')
    ax.set_title(f"Proof of Price Inelasticity: {demo_cat}", fontsize=14, fontweight='bold')
    ax.set_xlabel('Discount Offered (%)')
    ax.set_ylabel('Order Quantity')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "chart_discount_proof_of_noise.png"), dpi=200)

print("=" * 70)
print("PHASE 3 COMPLETE")
print("=" * 70)
