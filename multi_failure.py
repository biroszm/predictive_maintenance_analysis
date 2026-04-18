import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score
)

# =========================
# 1. Load data
# =========================
df = pd.read_csv("predictive_maintenance.csv")

# =========================
# 2. Define features and target
# =========================
feature_cols = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

target_col = "Failure Type"

feature_cols = [col for col in feature_cols if col in df.columns]

if target_col not in df.columns:
    raise ValueError(f"Column '{target_col}' not found.")

if len(feature_cols) == 0:
    raise ValueError("None of the requested feature columns were found.")

# Exclude No Failure
df = df.dropna(subset=[target_col]).copy()
df = df[df[target_col].astype(str).str.strip().str.lower() != "no failure"].copy()

X = df[feature_cols]
y = df[target_col].astype(str).str.strip()

print("Failure Type counts (excluding No Failure):")
print(y.value_counts())
print()

# =========================
# 3. Encode target labels
# =========================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_

print("Classes:")
print(list(class_names))
print()

# =========================
# 4. Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Training set size:", len(X_train))
print("Test set size:", len(X_test))
print()

# =========================
# 5. Define models
# =========================
models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        random_state=42
        ))
    ]),

    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=6,
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
}

# =========================
# 6. Train and evaluate all models
# =========================
results = {}

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    results[name] = {
        "Accuracy": acc,
        "F1-macro": f1_macro,
        "F1-weighted": f1_weighted
    }

    print("=" * 70)
    print(name)
    print("=" * 70)

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    print("Confusion Matrix:")
    print(cm_df)
    print()

    print("Classification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    ))
    print()

# =========================
# 7. Model comparison table
# =========================
results_df = pd.DataFrame(results).T.reset_index()
results_df = results_df.rename(columns={"index": "Model"})
results_df = results_df.sort_values("F1-macro", ascending=False)

pd.set_option("display.float_format", lambda x: f"{x:.4f}")

print("=" * 70)
print("Multiclass model comparison")
print("=" * 70)
print(results_df)
print()

# =========================
# 8. Plot confusion matrix for Random Forest
# =========================
rf_pipeline = models["Random Forest"]
rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(8, 6))
plt.imshow(cm_rf, interpolation="nearest", aspect="auto")
plt.title("Confusion Matrix - Random Forest")
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha="right")
plt.yticks(tick_marks, class_names)

for i in range(cm_rf.shape[0]):
    for j in range(cm_rf.shape[1]):
        plt.text(j, i, str(cm_rf[i, j]), ha="center", va="center")

plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.show()

# =========================
# 9. Prepare transformed data for SHAP
# =========================
imputer = rf_pipeline.named_steps["imputer"]
rf_model = rf_pipeline.named_steps["model"]

X_train_imp = pd.DataFrame(
    imputer.transform(X_train),
    columns=feature_cols,
    index=X_train.index
)

X_test_imp = pd.DataFrame(
    imputer.transform(X_test),
    columns=feature_cols,
    index=X_test.index
)

# =========================
# 10. SHAP for Random Forest
# =========================
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_imp)

print("X_test_imp shape:", X_test_imp.shape)
print("Type of shap_values:", type(shap_values))

if isinstance(shap_values, list):
    print("SHAP list shapes:", [sv.shape for sv in shap_values])
elif isinstance(shap_values, np.ndarray):
    print("SHAP array shape:", shap_values.shape)

# Handle different SHAP output formats
if isinstance(shap_values, list):
    shap_by_class = shap_values
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_by_class = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
else:
    raise ValueError(
        f"Unexpected SHAP output format: type={type(shap_values)}, "
        f"shape={getattr(shap_values, 'shape', None)}"
    )

# =========================
# 11. SHAP summary plots
# =========================
for i, class_name in enumerate(class_names):
    print(f"Generating SHAP summary plot for class: {class_name}")

    shap.summary_plot(
        shap_by_class[i],
        X_test_imp,
        feature_names=feature_cols,
        show=False
    )
    plt.title(f"SHAP Summary Plot - {class_name}")
    plt.tight_layout()
    plt.show()

# =========================
# 12. SHAP bar plots
# =========================
for i, class_name in enumerate(class_names):
    print(f"Generating SHAP bar plot for class: {class_name}")

    shap.summary_plot(
        shap_by_class[i],
        X_test_imp,
        feature_names=feature_cols,
        plot_type="bar",
        show=False
    )
    plt.title(f"SHAP Feature Importance - {class_name}")
    plt.tight_layout()
    plt.show()

# =========================
# 13. Waterfall plot for one sample
# =========================
sample_idx = 0
predicted_class_idx = y_pred_rf[sample_idx]
predicted_class_name = class_names[predicted_class_idx]

print(f"Explaining test sample index: {sample_idx}")
print(f"Predicted class: {predicted_class_name}")
print("Feature values:")
print(X_test_imp.iloc[sample_idx])
print()

expected_value = explainer.expected_value
if isinstance(expected_value, (list, np.ndarray)) and np.ndim(expected_value) > 0:
    base_value = expected_value[predicted_class_idx]
else:
    base_value = expected_value

shap_explanation = shap.Explanation(
    values=shap_by_class[predicted_class_idx][sample_idx],
    base_values=base_value,
    data=X_test_imp.iloc[sample_idx].values,
    feature_names=feature_cols
)

shap.plots.waterfall(shap_explanation, max_display=len(feature_cols), show=False)
plt.tight_layout()
plt.show()