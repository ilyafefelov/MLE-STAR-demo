import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data from the CSV
data = {
    'configuration': ['minimal', 'no_feature_engineering', 'no_ensemble', 'full', 'no_tuning', 'no_scaling'],
    'mean': [0.8440, 0.7817, 0.6830, 0.6366, 0.6366, -0.0007],
    'ci_lower': [0.8408, 0.7777, 0.6773, 0.6311, 0.6311, -0.0014],
    'ci_upper': [0.8472, 0.7856, 0.6886, 0.6420, 0.6420, -0.00005]
}

df = pd.DataFrame(data)

# Calculate error bars (symmetric or asymmetric, matplotlib takes relative values)
# yerr should be shape (2, N) for (lower, upper) relative to mean
lower_err = df['mean'] - df['ci_lower']
upper_err = df['ci_upper'] - df['mean']
yerr = [lower_err, upper_err]

# Sort by mean for better visualization
df = df.sort_values('mean', ascending=True)
yerr = [df['mean'] - df['ci_lower'], df['ci_upper'] - df['mean']]

plt.figure(figsize=(10, 6))

# Plot points with error bars
plt.errorbar(df['mean'], df['configuration'], xerr=yerr, fmt='o', capsize=5, color='black', ecolor='red', markersize=8)

plt.title('Statistical Validity of Ablation Study (California Housing)\nMean $R^2$ with 95% Confidence Intervals (N=20)', fontsize=14)
plt.xlabel('$R^2$ Score', fontsize=12)
plt.ylabel('Configuration', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Add annotation about N=20
plt.text(0.1, 0.9, 'N = 20 runs per config\nError Bars = 95% CI', transform=plt.gca().transAxes, 
         fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('reports/ci_validity_plot.png', dpi=300)
print("Plot generated: reports/ci_validity_plot.png")
