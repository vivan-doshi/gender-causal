import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
df_final = pd.read_csv('data/processed/processed_acs_data.csv')

# Define control features (exclude target, treatment, weights, and non-numeric)
# We'll select all numeric columns that are not the target or treatment
exclude_cols = ['LOG_WAGE', 'FEMALE', 'PERWT', 'INCWAGE', 'WAGE_HOURLY', 'ln_wage_hourly']
numeric_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()
control_features = [c for c in numeric_cols if c not in exclude_cols]

print(f"Control features selected: {len(control_features)}")

# Function to estimate gap for a subgroup
def estimate_gap_subgroup(subgroup_mask, group_name):
    """Estimate gender gap for a specific subgroup using simple regression"""
    subset = df_final[subgroup_mask].copy()
    
    # Drop NaNs in relevant columns
    # Ensure we only check columns that exist in subset
    valid_controls = [c for c in control_features if c in subset.columns]
    
    # CRITICAL FIX: Ensure only numeric columns are used
    subset_numeric = subset.select_dtypes(include=[np.number])
    valid_controls = [c for c in valid_controls if c in subset_numeric.columns]
    
    cols_to_check = ['FEMALE', 'LOG_WAGE', 'PERWT'] + valid_controls
    subset = subset.dropna(subset=cols_to_check)
    
    if len(subset) < 100 or subset['FEMALE'].sum() < 50 or (subset['FEMALE']==0).sum() < 50:
        print(f"  Skipping {group_name}: Insufficient data (n={len(subset)})")
        return None, None, None, len(subset)
    
    # Remove constant columns
    current_controls = [c for c in valid_controls if subset[c].nunique() > 1]
    
    try:
        X_sub = sm.add_constant(subset[['FEMALE'] + current_controls])
        y_sub = subset['LOG_WAGE']
        weights_sub = subset['PERWT']
    
        model = sm.WLS(y_sub, X_sub, weights=weights_sub).fit()
        coef = model.params['FEMALE']
        se = model.bse['FEMALE']
        pval = model.pvalues['FEMALE']
        return coef, se, pval, len(subset)
    except Exception as e:
        print(f"  Error for {group_name}: {str(e)}")
        return None, None, None, len(subset)

print("\nRunning Analysis...")

# Education
print("\n📚 Gender Gap by EDUCATION LEVEL:")
education_results = []
educ_groups = {
    'High School or Less': df_final['EDUC_NUM'] <= 6,
    'Some College': (df_final['EDUC_NUM'] > 6) & (df_final['EDUC_NUM'] < 9),
    "Bachelor's Degree": df_final['EDUC_NUM'] == 9,
    'Graduate Degree': df_final['EDUC_NUM'] >= 10
}
for group_name, mask in educ_groups.items():
    coef, se, pval, n = estimate_gap_subgroup(mask, group_name)
    if coef is not None:
        gap_pct = (np.exp(coef) - 1) * 100
        print(f"   {group_name:25s}: {gap_pct:6.2f}% (n={n:,}, p={pval:.4f})")
        education_results.append({'Group': group_name, 'Gap': gap_pct, 'N': n})

# Age
print("\n📅 Gender Gap by AGE GROUP:")
age_results = []
age_groups = {
    '18-25': (df_final['AGE'] >= 18) & (df_final['AGE'] <= 25),
    '26-35': (df_final['AGE'] >= 26) & (df_final['AGE'] <= 35),
    '36-45': (df_final['AGE'] >= 36) & (df_final['AGE'] <= 45),
    '46-55': (df_final['AGE'] >= 46) & (df_final['AGE'] <= 55),
    '56-65': (df_final['AGE'] >= 56) & (df_final['AGE'] <= 65)
}
for group_name, mask in age_groups.items():
    coef, se, pval, n = estimate_gap_subgroup(mask, group_name)
    if coef is not None:
        gap_pct = (np.exp(coef) - 1) * 100
        print(f"   {group_name:25s}: {gap_pct:6.2f}% (n={n:,}, p={pval:.4f})")
        age_results.append({'Group': group_name, 'Gap': gap_pct, 'N': n})

# Generate Plot
print("\nGenerating plot...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: By Education
ax1 = axes[0]
if education_results:
    ed_df = pd.DataFrame(education_results)
    colors = ['#d62728' if g < 0 else '#2ca02c' for g in ed_df['Gap']]
    ax1.barh(ed_df['Group'], ed_df['Gap'], color=colors, edgecolor='black')
    ax1.axvline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('Gender Gap (%)', fontsize=12)
    ax1.set_title('Gender Wage Gap by Education Level', fontsize=14, fontweight='bold')
    for i, (g, n) in enumerate(zip(ed_df['Gap'], ed_df['N'])):
        ax1.text(g - 1, i, f'{g:.1f}%', va='center', ha='right', fontsize=10, fontweight='bold')

# Plot 2: By Age
ax2 = axes[1]
if age_results:
    age_df = pd.DataFrame(age_results)
    ax2.plot(age_df['Group'], age_df['Gap'], 'o-', markersize=12, linewidth=2, color='steelblue')
    ax2.fill_between(age_df['Group'], age_df['Gap'], alpha=0.3, color='steelblue')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Age Group', fontsize=12)
    ax2.set_ylabel('Gender Gap (%)', fontsize=12)
    ax2.set_title('Gender Wage Gap by Age Group', fontsize=14, fontweight='bold')
    for i, (x, y) in enumerate(zip(age_df['Group'], age_df['Gap'])):
        ax2.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                     xytext=(0,10), ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('reports/figures/heterogeneous_effects.png', dpi=150, bbox_inches='tight')
print("✅ Figure saved to: reports/figures/heterogeneous_effects.png")
