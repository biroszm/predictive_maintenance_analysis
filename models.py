import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    classification_report
)

# =========================
# 1. Load data
# =========================
df = pd.read_csv("predictive_maintenance.csv")

print("Columns in dataset:")
print(df.columns.tolist())
print()

# =========================
# 2. Define target and features
# =========================
target_col = "Target"

# Use the numeric variables you focused on
feature_cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# Keep only columns that exist
feature_cols = [col for col in feature_cols if col in df.columns]

if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset.")

if len(feature_cols) == 0:
    raise ValueError("None of the requested feature columns were found in the dataset.")

print("Features used:")
print(feature_cols)
print()

X = df[feature_cols]
y = df[target_col]

# =========================
# 3. Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training set size:", len(X_train))
print("Test set size:", len(X_test))
print("Failure rate in full dataset:", y.mean())
print("Failure rate in train set:", y_train.mean())
print("Failure rate in test set:", y_test.mean())
print()

# =========================
# 4. Define models
# =========================
models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42))
    ]),

    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42,
            max_depth=5
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
}

# =========================
# 5. Train and evaluate
# =========================
results = []

plt.figure(figsize=(8, 6))
for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    # For ROC-AUC and PR-AUC we need scores/probabilities
    if hasattr(pipeline, "predict_proba"):
        y_score = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_score = pipeline.decision_function(X_test)

    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_score)
    pr_auc = average_precision_score(y_test, y_score)

    results.append({
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    })

    print("=" * 60)
    print(name)
    print("=" * 60)
    print("Confusion Matrix:")
    print(cm)
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")
    print()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_score)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 6. Precision-Recall curves
# =========================
plt.figure(figsize=(8, 6))
for name, pipeline in models.items():
    if hasattr(pipeline, "predict_proba"):
        y_score = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_score = pipeline.decision_function(X_test)

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_score)
    pr_auc = average_precision_score(y_test, y_score)

    plt.plot(recall_curve, precision_curve, label=f"{name} (AP={pr_auc:.3f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 7. Results table
# =========================
results_df = pd.DataFrame(results).sort_values(by="Recall", ascending=False)

pd.set_option("display.float_format", lambda x: f"{x:.4f}")

print("=" * 60)
print("Model comparison")
print("=" * 60)
print(results_df)

results_df.to_csv("baseline_model_results.csv", index=False)

# =========================
# 8. Feature importance / coefficients
# =========================
for name, pipeline in models.items():
    model = pipeline.named_steps["model"]

    print("\n" + "=" * 60)
    print(f"Interpretation for {name}")
    print("=" * 60)

    if name == "Logistic Regression":
        coefs = model.coef_[0]
        coef_df = pd.DataFrame({
            "Feature": feature_cols,
            "Coefficient": coefs
        }).sort_values(by="Coefficient", key=np.abs, ascending=False)

        print(coef_df)

    elif name in ["Decision Tree", "Random Forest"]:
        importances = model.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        print(imp_df)