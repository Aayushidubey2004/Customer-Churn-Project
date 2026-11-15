# Customer Churn Intelligence & Machine Learning Project

An end-to-end Data Science project analyzing customer churn, machine learning predictions with explainability, survival analysis, uplift modeling, Customer Lifetime Value (CLV) estimation, customer segmentation, and an interactive Streamlit retention dashboard.

---

## 📁 Project Structure

```text
├── Telco-Customer-Churn.csv     # Local copy of the Telco customer dataset
├── customer_churn_project.py    # Complete Data Science & ML pipeline script
├── app.py                       # Interactive Streamlit Web Application
├── requirements.txt             # Required Python dependencies
├── .gitignore                   # Git exclusion rules
└── README.md                    # Project documentation
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Machine Learning & Analytics Pipeline
To execute the complete end-to-end data analysis and model training (EDA, LightGBM, SHAP, Lifelines, Uplift Modeling, CLV, K-Means Clustering):

```bash
python customer_churn_project.py
```

### 3. Launch the Interactive Web Dashboard
To start the interactive Streamlit dashboard:

```bash
streamlit run app.py
```
This opens the web interface in your browser at `http://localhost:8501`.

---

## 📊 Pipeline Components

1. **Data Ingestion & Cleaning**: Handles missing values in `TotalCharges` and verifies data integrity.
2. **Exploratory Data Analysis (EDA)**: Visualizes churn rates, tenure distributions, and contract type trends.
3. **Machine Learning Model**: Trains a tuned `LGBMClassifier` (LightGBM) to predict customer churn.
4. **Explainable AI (SHAP)**: Explains global feature importance and individual customer churn drivers with waterfall plots.
5. **Survival Analysis**: Computes Kaplan-Meier survival probabilities and Cox Proportional Hazards hazard ratios using `lifelines`.
6. **Uplift Modeling**: Estimates incremental offer responsiveness using a Two-Learner (T-Learner) architecture.
7. **Customer Lifetime Value (CLV)**: Estimates future customer revenue based on historical spend and churn risk.
8. **Customer Segmentation**: Clusters customer profiles using K-Means into 3 distinct behavioral segments.
9. **Interactive Dashboard**: Streamlit interface with dynamic retention offer ROI simulation.

