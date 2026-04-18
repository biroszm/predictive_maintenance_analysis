import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu

# Read the CSV file
df = pd.read_csv("predictive_maintenance.csv")

# Columns to analyze
columns_to_analyze = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# Keep only columns that exist
columns_to_analyze = [col for col in columns_to_analyze if col in df.columns]

# Split by Target
no_failure = df[df["Target"] == 0]
failure = df[df["Target"] == 1]

# Store results
results = []

for col in columns_to_analyze:
    x0 = no_failure[col].dropna()
    x1 = failure[col].dropna()

    # Basic statistics
    mean_0 = x0.mean()
    mean_1 = x1.mean()
    median_0 = x0.median()
    median_1 = x1.median()
    std_0 = x0.std()
    std_1 = x1.std()

    mean_diff = mean_1 - mean_0
    median_diff = median_1 - median_0

    # Welch's t-test
    t_stat, t_p = ttest_ind(x0, x1, equal_var=False, nan_policy="omit")

    # Mann-Whitney U test
    u_stat, u_p = mannwhitneyu(x0, x1, alternative="two-sided")

    results.append({
        "Variable": col,
        "Mean (No Failure)": mean_0,
        "Mean (Failure)": mean_1,
        "Mean Difference": mean_diff,
        "Median (No Failure)": median_0,
        "Median (Failure)": median_1,
        "Median Difference": median_diff,
        "Std (No Failure)": std_0,
        "Std (Failure)": std_1,
        "T-test p-value": t_p,
        "Mann-Whitney p-value": u_p
    })

# Create results table
results_df = pd.DataFrame(results)

# Optional: sort by smallest p-value
results_df = results_df.sort_values("Mann-Whitney p-value")

# Make numbers easier to read
pd.set_option("display.float_format", lambda x: f"{x:.6f}")

print("\nStatistical comparison between Target=0 and Target=1:\n")
print(results_df)

# Save results
results_df.to_csv("statistical_comparison_results.csv", index=False)

# Boxplots
for col in columns_to_analyze:
    plt.figure(figsize=(8, 5))
    plt.boxplot(
        [no_failure[col].dropna(), failure[col].dropna()],
        labels=["No Failure (0)", "Failure (1)"]
    )
    plt.title(f"Boxplot of {col} by Target")
    plt.ylabel(col)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()