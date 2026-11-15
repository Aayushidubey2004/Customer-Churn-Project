# -*- coding: utf-8 -*-
"""Customer Churn Analytics & Modeling Pipeline

End-to-end Machine Learning, Explainability, Survival Analysis, Uplift Modeling,
Customer Lifetime Value (CLV), and Customer Segmentation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from lightgbm import LGBMClassifier
import shap

# Safe display helper for both Jupyter/Colab notebooks and standard terminal execution
try:
    from IPython.display import display
except ImportError:
    display = print

print("=" * 60)
print("Welcome to my Customer Churn Project!")
print("I am learning Python step by step.")
print("=" * 60)

# ==========================================
# 1. DATA INGESTION & EXPLORATION
# ==========================================
csv_filename = "Telco-Customer-Churn.csv"
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

if os.path.exists(csv_filename):
    print(f"Loading dataset from local file '{csv_filename}'...")
    df = pd.read_csv(csv_filename)
else:
    print(f"Fetching dataset from remote IBM repository...")
    df = pd.read_csv(url)
    df.to_csv(csv_filename, index=False)
    print(f"Saved local copy to '{csv_filename}'.")

print("Dataset loaded successfully!")
print("Number of rows and columns:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 customers:")
display(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values in each column before cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())
print("Repeated customer IDs:", df["customerID"].duplicated().sum())

# ==========================================
# 2. DATA CLEANING
# ==========================================
print("\nCleaning TotalCharges column...")
print("Original TotalCharges data type:", df["TotalCharges"].dtype)

# Convert TotalCharges to numeric, replacing unparseable spaces with NaN
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("Missing values after numeric conversion:", df["TotalCharges"].isnull().sum())

# Fill missing values with 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)
print("Missing TotalCharges after cleaning:", df["TotalCharges"].isnull().sum())

# ==========================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
print("\nCustomer churn counts:")
print(df["Churn"].value_counts())

print("\nCustomer churn percentages:")
print(df["Churn"].value_counts(normalize=True) * 100)

# Churn Count Bar Plot
plt.figure()
df["Churn"].value_counts().plot(kind="bar", color=["skyblue", "salmon"])
plt.title("Customer Churn Count")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# Churn by Contract Type
churn_by_contract = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100
print("\nChurn Percentage by Contract Type:")
print(churn_by_contract)

plt.figure()
churn_by_contract.plot(kind="bar")
plt.title("Churn Percentage by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print("\nCustomer tenure statistics:")
print(df["tenure"].describe())

print("\nMonthly charges statistics:")
print(df["MonthlyCharges"].describe())

print("\nTotal charges statistics:")
print(df["TotalCharges"].describe())

# Tenure Distribution
plt.figure()
df["tenure"].hist(bins=20, edgecolor="black", color="lightblue")
plt.title("Distribution of Customer Tenure")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# Monthly Charges Distribution
plt.figure()
df["MonthlyCharges"].hist(bins=20, edgecolor="black", color="lightgreen")
plt.title("Distribution of Monthly Charges")
plt.xlabel("Monthly Charges")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# Boxplots by Churn
plt.figure()
df.boxplot(column="tenure", by="Churn")
plt.title("Customer Tenure by Churn Status")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")
plt.tight_layout()
plt.show()

plt.figure()
df.boxplot(column="MonthlyCharges", by="Churn")
plt.title("Monthly Charges by Churn Status")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.show()

# ==========================================
# 4. PREPARING DATA FOR MACHINE LEARNING
# ==========================================
model_df = df.copy()

# Drop unique identifier
model_df = model_df.drop(columns=["customerID"])

# Binary mapping for target
model_df["Churn"] = model_df["Churn"].map({"No": 0, "Yes": 1})

# One-hot encode categorical features
model_df = pd.get_dummies(
    model_df,
    columns=model_df.select_dtypes(include="object").columns,
    drop_first=True,
    dtype=int
)

print("\nPrepared dataset shape:", model_df.shape)
print("\nFirst five rows of encoded data:")
display(model_df.head())

print("\nTotal missing values in encoded data:", model_df.isnull().sum().sum())
print("Churn distribution in encoded data:\n", model_df["Churn"].value_counts())

# Train/Test Split
X = model_df.drop(columns=["Churn"])
y = model_df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining rows: {X_train.shape[0]} | Testing rows: {X_test.shape[0]} | Features: {X_train.shape[1]}")

# ==========================================
# 5. MODEL TRAINING & EVALUATION (LIGHTGBM)
# ==========================================
print("\nTraining LightGBM Classifier...")
model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    random_state=42,
    verbosity=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nFirst 10 predicted churn values:", y_pred[:10])
print("First 10 actual churn values:   ", y_test.iloc[:10].values)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ==========================================
# 6. MODEL EXPLAINABILITY (SHAP)
# ==========================================
print("\nCalculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# In binary classification, handle single array or list of outputs
if isinstance(shap_values, list):
    shap_vals_to_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
else:
    shap_vals_to_plot = shap_values

print("SHAP values calculated successfully!")

shap.summary_plot(shap_vals_to_plot, X_test, show=False)
plt.title("SHAP Feature Summary Plot")
plt.tight_layout()
plt.show()

shap.summary_plot(shap_vals_to_plot, X_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Bar)")
plt.tight_layout()
plt.show()

mean_shap = np.abs(shap_vals_to_plot).mean(axis=0)
feature_importance = pd.DataFrame({
    "Feature": X_test.columns,
    "SHAP Importance": mean_shap
}).sort_values(by="SHAP Importance", ascending=False)

print("\nTop 10 Important Features by SHAP:")
print(feature_importance.head(10).to_string(index=False))

# Individual Customer Explanation
customer_index = 0
customer_data = X_test.iloc[customer_index]
customer_shap_values = shap_vals_to_plot[customer_index]

base_value = explainer.expected_value
if isinstance(base_value, (list, np.ndarray)):
    base_value = np.asarray(base_value).flatten()[-1]

prediction = model.predict(customer_data.to_frame().T)[0]
churn_probability = model.predict_proba(customer_data.to_frame().T)[0][1]
actual_status = "Churn" if y_test.iloc[customer_index] == 1 else "Stayed"

print(f"\n--- Customer #{customer_index} Explanation ---")
print("Predicted Status: ", "Churn" if prediction == 1 else "Stayed")
print(f"Churn Probability: {churn_probability:.4f}")
print("Actual Status:    ", actual_status)

customer_explanation = shap.Explanation(
    values=customer_shap_values,
    base_values=base_value,
    data=customer_data.values,
    feature_names=X_test.columns.tolist()
)
shap.plots.waterfall(customer_explanation, max_display=15, show=False)
plt.title(f"SHAP Waterfall - Customer #{customer_index}")
plt.tight_layout()
plt.show()

# ==========================================
# 7. SURVIVAL ANALYSIS (LIFELINES)
# ==========================================
try:
    from lifelines import KaplanMeierFitter, CoxPHFitter

    print("\n--- Running Survival Analysis with Lifelines ---")
    survival_df = df.copy()
    survival_df["Churn"] = survival_df["Churn"].map({"No": 0, "Yes": 1})
    survival_df = survival_df[["tenure", "Churn"]]

    kmf = KaplanMeierFitter()
    kmf.fit(durations=survival_df["tenure"], event_observed=survival_df["Churn"])

    plt.figure(figsize=(8, 5))
    kmf.plot_survival_function()
    plt.title("Customer Survival Probability (Kaplan-Meier)")
    plt.xlabel("Tenure (Months)")
    plt.ylabel("Survival Probability")
    plt.tight_layout()
    plt.show()

    # Survival by Contract Type
    plt.figure(figsize=(10, 6))
    for contract in df["Contract"].unique():
        mask = df["Contract"] == contract
        kmf.fit(
            durations=df.loc[mask, "tenure"],
            event_observed=df.loc[mask, "Churn"].map({"No": 0, "Yes": 1}),
            label=contract
        )
        kmf.plot_survival_function()

    plt.title("Customer Survival by Contract Type")
    plt.xlabel("Tenure (Months)")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Cox Proportional Hazards Model
    cox_df = df[[
        "tenure",
        "Churn",
        "MonthlyCharges",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "InternetService",
        "Contract",
        "PaymentMethod"
    ]].copy()

    cox_df["Churn"] = cox_df["Churn"].map({"No": 0, "Yes": 1})
    cox_df = pd.get_dummies(
        cox_df,
        columns=cox_df.select_dtypes(include="object").columns,
        drop_first=True,
        dtype=int
    )

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(cox_df, duration_col="tenure", event_col="Churn")
    print("\nCox Proportional Hazards Model Summary:")
    cph.print_summary()

    plt.figure(figsize=(10, 6))
    cph.plot()
    plt.title("Cox Proportional Hazards Coefficients")
    plt.tight_layout()
    plt.show()

    hazard_ratios = np.exp(cph.params_)
    print("\nHazard Ratios (sorted):")
    print(hazard_ratios.sort_values(ascending=False))

except ImportError:
    print("\n[Notice] 'lifelines' not installed; skipping survival analysis.")

# ==========================================
# 8. UPLIFT MODELING (CAUSAL ML / TWO-LEARNER)
# ==========================================
print("\n--- Running Uplift Modeling ---")
uplift_df = df.copy()
uplift_df["Churn"] = uplift_df["Churn"].map({"No": 0, "Yes": 1})
uplift_df["Stayed"] = 1 - uplift_df["Churn"]

uplift_encoded = pd.get_dummies(
    uplift_df.drop(columns=["customerID", "Churn", "Stayed"]),
    drop_first=True,
    dtype=int
)

np.random.seed(42)
uplift_df["Treatment"] = np.random.binomial(1, 0.5, size=len(uplift_df))
X_uplift = uplift_encoded
y_uplift = uplift_df["Stayed"]
treatment = uplift_df["Treatment"]

try:
    from causalml.inference.meta import BaseTClassifier
    print("Using CausalML BaseTClassifier...")
    uplift_model = BaseTClassifier(
        learner=LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbosity=-1)
    )
    uplift_model.fit(X=X_uplift, treatment=treatment, y=y_uplift)
    uplift_scores = uplift_model.predict(X_uplift, treatment=treatment)
except (ImportError, Exception):
    print("Using native LightGBM Two-Learner (T-Learner)...")
    class NativeTLearner:
        def __init__(self):
            self.model0 = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbosity=-1)
            self.model1 = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbosity=-1)

        def fit(self, X, treatment, y):
            mask0 = (treatment == 0)
            mask1 = (treatment == 1)
            self.model0.fit(X[mask0], y[mask0])
            self.model1.fit(X[mask1], y[mask1])
            return self

        def predict(self, X):
            p1 = self.model1.predict_proba(X)[:, 1]
            p0 = self.model0.predict_proba(X)[:, 1]
            return p1 - p0

    uplift_model = NativeTLearner()
    uplift_model.fit(X_uplift, treatment, y_uplift)
    uplift_scores = uplift_model.predict(X_uplift)

uplift_df["Uplift_Score"] = np.asarray(uplift_scores).reshape(-1)
uplift_ranked = uplift_df.sort_values(by="Uplift_Score", ascending=False)

print("\nTop 10 Potential Offer-Responsive Customers:")
print(uplift_ranked[["customerID", "Treatment", "Stayed", "Uplift_Score"]].head(10).to_string(index=False))

# ==========================================
# 9. CUSTOMER LIFETIME VALUE (CLV)
# ==========================================
print("\n--- Estimating Customer Lifetime Value (CLV) ---")
clv_df = df[["customerID", "tenure", "MonthlyCharges", "Churn"]].copy()
clv_df["Churn"] = clv_df["Churn"].map({"No": 0, "Yes": 1})

clv_df["Historical_Revenue"] = clv_df["MonthlyCharges"] * clv_df["tenure"]

X_clv = model_df.drop(columns=["Churn"])
clv_df["Churn_Probability"] = model.predict_proba(X_clv)[:, 1]
clv_df["Retention_Probability"] = 1 - clv_df["Churn_Probability"]

clv_df["Expected_Future_Months"] = 12 * clv_df["Retention_Probability"]
clv_df["Expected_Future_Revenue"] = clv_df["MonthlyCharges"] * clv_df["Expected_Future_Months"]
clv_df["Estimated_CLV"] = clv_df["Historical_Revenue"] + clv_df["Expected_Future_Revenue"]

clv_sorted = clv_df.sort_values(by="Estimated_CLV", ascending=False)
print("\nTop 10 High-Value Customers by Estimated CLV:")
print(clv_sorted[["customerID", "Historical_Revenue", "Expected_Future_Revenue", "Estimated_CLV"]].head(10).to_string(index=False))

# ==========================================
# 10. CUSTOMER SEGMENTATION (K-MEANS)
# ==========================================
print("\n--- Running Customer Segmentation (K-Means Clustering) ---")
cluster_df = df[["tenure", "MonthlyCharges", "TotalCharges"]].copy()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_df)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_df["Cluster"] = kmeans.fit_predict(scaled_data)

print("\nCustomer Cluster Summary (Averages):")
cluster_summary = cluster_df.groupby("Cluster").mean()
print(cluster_summary)

plt.figure(figsize=(9, 6))
plt.scatter(
    cluster_df["tenure"],
    cluster_df["MonthlyCharges"],
    c=cluster_df["Cluster"],
    cmap="viridis",
    alpha=0.6,
    edgecolors="none"
)
plt.title("Customer Segmentation (K-Means Clustering)")
plt.xlabel("Tenure (Months)")
plt.ylabel("Monthly Charges")
plt.colorbar(label="Cluster ID")
plt.tight_layout()
plt.show()

print("\nCustomer counts in each cluster:")
print(cluster_df["Cluster"].value_counts().sort_index())

print("\n" + "=" * 60)
print("PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
print("To launch the interactive web dashboard, run:")
print("    streamlit run app.py")
print("=" * 60)
