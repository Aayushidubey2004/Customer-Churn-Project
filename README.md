# Customer Churn Intelligence: Enterprise Analysis & Strategy

An end-to-end Data Science project analyzing customer churn, machine learning predictions with explainability, survival analysis, uplift modeling, Customer Lifetime Value (CLV) estimation, customer segmentation, and an interactive Streamlit retention dashboard.

---

## 📁 Project Structure

```text
├── Customer_Churn_Intelligence.ipynb # Complete professional Data Science notebook (Google Colab / Jupyter)
├── Telco-Customer-Churn.csv         # Local copy of the Telco customer dataset
├── customer_churn_project.py        # Complete Data Science & ML pipeline script
├── app.py                           # Interactive Streamlit Web Application & ROI Simulator
├── ARCHITECTURE_AND_API_DOCUMENTATION.md # Architecture diagrams, data dictionary & API specs
├── requirements.txt                 # Required Python dependencies
├── .gitignore                       # Git exclusion rules
└── README.md                        # Project documentation
```

---

## 🚀 How to Run

### 1. Run in Google Colab / Jupyter Notebook
Open and run **`Customer_Churn_Intelligence.ipynb`** in Google Colab or your local Jupyter environment. It contains all 8 parts with comprehensive markdown documentation, graph headings, observations, inferences, and conclusions.

### 2. Install Dependencies Locally
```bash
pip install -r requirements.txt
```

### 3. Run the Machine Learning & Analytics Pipeline Script
To execute the complete end-to-end pipeline via terminal:

```bash
python customer_churn_project.py
```

### 4. Launch the Interactive Web Dashboard
To start the interactive Streamlit dashboard:

```bash
streamlit run app.py
```
This opens the web interface in your browser at `http://localhost:8501`.

### 5. Architecture & API Specifications
For detailed system flowcharts, 21-feature raw data dictionary, 30-feature encoded matrix, Python module contracts, and production REST API endpoints (FastAPI/OpenAPI), consult **[ARCHITECTURE_AND_API_DOCUMENTATION.md](file:///c:/Users/dubey/OneDrive/Desktop/DATA%20SCIENCE/ARCHITECTURE_AND_API_DOCUMENTATION.md)**.

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

