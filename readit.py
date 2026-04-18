import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv("predictive_maintenance.csv")

# Optional: check exact column names
print("Columns in file:")
print(df.columns.tolist())

# Sort by Target
df = df.sort_values("Target")

# Split into categories
no_failure = df[df["Target"] == 0]
failure = df[df["Target"] == 1]

# Requested columns
requested_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# Keep only columns that actually exist
columns_to_plot = [col for col in requested_columns if col in df.columns]

for col in columns_to_plot:
    plt.figure(figsize=(10, 5))

    # Count values for each class
    no_failure_counts = no_failure[col].value_counts().sort_index()
    failure_counts = failure[col].value_counts().sort_index()

    # Align both to same x-values
    all_values = sorted(set(no_failure_counts.index).union(set(failure_counts.index)))
    no_failure_counts = no_failure_counts.reindex(all_values, fill_value=0)
    failure_counts = failure_counts.reindex(all_values, fill_value=0)

    x = np.arange(len(all_values))

    # Overlapping bar plots
    plt.bar(x, no_failure_counts.values, alpha=0.6, label="No Failure (Target=0)")
    plt.bar(x, failure_counts.values, alpha=0.6, label="Failure (Target=1)")

    plt.title(f"Bar Plot of {col} by Target Category")
    plt.xlabel(col)
    plt.ylabel("Count")

    # Show fewer x-axis tick labels
    step = max(1, len(all_values) // 10)
    tick_positions = np.arange(0, len(all_values), step)
    tick_labels = [round(all_values[i], 2) for i in tick_positions]

    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()