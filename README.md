# Predictive Maintenance Analysis

This project explores a machine predictive maintenance dataset and develops a full analysis workflow for detecting failures and understanding different failure mechanisms. The work combines exploratory data analysis, statistical comparison, baseline machine learning models, and interpretable multiclass classification in order to answer two main questions:

1. Can machine failure be predicted from sensor and operational variables?
2. If a failure occurs, can its specific failure type also be identified?

The input data used in this project comes from the following Kaggle dataset:

[Kaggle: Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)

---

## Project Objective

Predictive maintenance is a practical machine learning problem where the goal is to identify equipment conditions that are associated with failure before a serious breakdown occurs. In this project, the dataset contains operational and sensor-based variables such as:

- Air temperature \[K]
- Process temperature \[K]
- Rotational speed \[rpm]
- Torque \[Nm]
- Tool wear \[min]

These variables were used to investigate machine behavior under both normal and failure conditions.

The project solves the problem in two stages:

### 1. Binary classification
The first task is to predict whether a machine instance belongs to:
- **No Failure**
- **Failure**

This is based on the `Target` column.

### 2. Failure type classification
The second task focuses only on rows where a failure actually occurred and predicts the specific failure mechanism using the `Failure Type` column, such as:
- Heat Dissipation Failure
- Power Failure
- Overstrain Failure
- Tool Wear Failure
- Random Failures

This second stage helps move beyond simple failure detection and into failure diagnosis.

---

## Core Logic of the Analysis

The project was designed as a step-by-step analytical workflow rather than jumping immediately into model training.

### Step 1: Exploratory inspection
The analysis began by loading the CSV dataset and examining the available variables. Selected numeric columns were visualized first to understand their ranges and distributions. This initial stage helped verify the structure of the dataset and the naming of the relevant columns.

### Step 2: Class-based comparison
The next stage compared the `Target = 0` and `Target = 1` groups statistically. For each major numeric variable, the following were calculated:

- mean
- median
- standard deviation
- Welch’s t-test
- Mann–Whitney U test

This made it possible to evaluate whether the failure and no-failure groups differ significantly. The statistical tests showed that the examined variables are not distributed the same way across the two classes, meaning the features contain useful predictive signal.

### Step 3: Binary predictive modeling
After confirming that the feature distributions differ between failed and non-failed cases, baseline binary classification models were trained:

- Logistic Regression
- Decision Tree
- Random Forest

These models were evaluated using metrics suited to imbalanced classification:

- confusion matrix
- precision
- recall
- F1-score
- ROC-AUC
- PR-AUC

This step was important because the dataset is imbalanced, with relatively few failures compared to non-failures. Therefore, model quality could not be judged by accuracy alone. In predictive maintenance, recall for failure cases is especially important because missed failures can be more costly than false alarms.

### Step 4: Failure-type classification
After solving the binary problem, the project moved to a multiclass setting. Rows labeled as `No Failure` were excluded, and the task became predicting the specific failure type. Again, several baseline models were trained:

- Logistic Regression
- Decision Tree
- Random Forest

This stage showed that the available sensor and operating variables are not only useful for detecting failures, but also for distinguishing among several failure mechanisms.

### Step 5: Interpretation
To understand which variables drive classification decisions, model interpretation was added through:

- logistic regression coefficients
- feature importance from tree-based models
- SHAP-based model explanations for multiclass Random Forest predictions

This makes the project more than a prediction exercise: it also provides insight into which operating conditions are most strongly associated with particular failure types.

---

## How the Problem Was Solved

The problem was solved by treating the dataset as both a **supervised binary classification problem** and a **supervised multiclass classification problem**.

### Binary problem
The `Target` variable was used as the binary label:
- `0` = no failure
- `1` = failure

A train/test split was applied using stratification so that the class proportions remained similar in both subsets. Models were trained on the selected numeric variables, and performance was assessed using metrics that are appropriate for rare-event detection.

The Random Forest model achieved the strongest overall discrimination, while the Decision Tree provided higher recall in some cases. Logistic Regression served as an interpretable baseline.

### Multiclass problem
The `Failure Type` variable was used as the multiclass label. Before training, all `No Failure` rows were removed so that the model focused only on distinguishing real failure mechanisms.

The multiclass models learned to separate categories such as Heat Dissipation Failure, Power Failure, and Tool Wear Failure. This demonstrated that the input features capture distinct information related to different failure processes.

### Interpretation strategy
The project does not stop at reporting scores. It also examines *why* the models perform as they do.

- Logistic Regression coefficients indicate which variables push the prediction toward a given class.
- Decision Tree and Random Forest feature importances show which variables contribute most strongly overall.
- SHAP plots make it possible to inspect how each feature influences a specific class prediction and how feature effects differ across failure types.

This interpretability layer is particularly useful in maintenance-related applications, where stakeholders often need a reasoned explanation for why a machine was flagged as risky.

---

## Main Findings

The analysis suggests that the selected operating variables contain meaningful predictive information for both machine failure detection and failure type classification.

Some of the main observations are:

- The failure and no-failure groups differ significantly across the examined numeric variables.
- Torque, tool wear, and rotational speed emerged as especially influential features in several models.
- Binary failure detection can be performed effectively, though there is an important tradeoff between recall and false alarms.
- Failure type classification is also feasible, especially for the more common and structurally distinct failure categories.
- Rare classes such as Random Failures remain more difficult to model reliably due to limited sample size.

---

## Repository Contents

Typical components of this repository include:

- data loading and preprocessing scripts
- statistical comparison scripts
- plotting and visualization code
- binary classification experiments
- multiclass failure type classification experiments
- SHAP-based interpretation code

---

## Methods Used

### Data processing
- pandas
- NumPy

### Visualization
- Matplotlib

### Statistical analysis
- Welch’s t-test
- Mann–Whitney U test

### Machine learning
- Logistic Regression
- Decision Tree
- Random Forest
- train/test split with stratification
- class balancing through model settings

### Interpretation
- feature importance
- coefficients
- SHAP explanations

---

## Why This Project Matters

Predictive maintenance is valuable because it helps reduce downtime, avoid unexpected breakdowns, and improve operational planning. A model that can detect failure risk early is useful, but a model that can also distinguish the likely type of failure is even more informative.

This project therefore approaches the maintenance problem in a practical and structured way:

1. identify whether failure is likely,
2. distinguish the kind of failure,
3. interpret which variables are associated with that outcome.

That combination of prediction and interpretation is the core contribution of the project.

---

## Future Improvements

Possible future extensions include:

- cross-validated evaluation instead of relying on a single train/test split
- hyperparameter tuning for stronger model performance
- threshold optimization for binary failure detection
- engineered features such as temperature difference or power-related interactions
- one-vs-rest modeling for each failure type
- more advanced explainability and calibration analysis

---

## Dataset Source

This project uses the dataset published on Kaggle:

[Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)
