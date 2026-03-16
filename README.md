# 🏦 CreditLens Risk Intelligence

<div align="center">

**ML-powered credit risk scoring, interest rate prediction, and loan repayment planning**

[![Live App](https://img.shields.io/badge/🚀%20Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://creditriskprediction-dgrpbjdweg27gpaabf4mwo.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

> **Live App →** [https://creditriskprediction-dgrpbjdweg27gpaabf4mwo.streamlit.app/](https://creditriskprediction-dgrpbjdweg27gpaabf4mwo.streamlit.app/)

</div>

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Business Problem](#-business-problem)
3. [Dataset](#-dataset)
4. [ML Pipeline](#-ml-pipeline)
5. [Exploratory Data Analysis](#-exploratory-data-analysis)
6. [Feature Engineering](#-feature-engineering)
7. [SMOTE — Class Balancing](#-smote--class-balancing)
8. [Model Training & Evaluation](#-model-training--evaluation)
9. [Results — Actual vs Predicted](#-results--actual-vs-predicted)
10. [Streamlit Application](#-streamlit-application)
11. [Project Structure](#-project-structure)
12. [Installation & Setup](#-installation--setup)
13. [Tech Stack](#-tech-stack)

---

## 🎯 Project Overview

**CreditLens Risk Intelligence** is a complete end-to-end machine learning project that predicts the **credit risk score of a loan applicant**, represented by the loan interest rate (`loan_int_rate`). A higher predicted interest rate signals a riskier borrower, enabling banks and FinTech companies to make faster, fairer, and data-driven lending decisions.

The project includes:

- A full ML pipeline from raw data to a deployed production model
- A professional Streamlit web application with four modules: **Home**, **Loan Application**, **Loan Repayment Planner**, and **Analytics Dashboard**
- Banking-accurate **EMI calculation** and full **amortisation schedule generation**
- Downloadable **Credit Risk Report** (TXT) and **Installment Schedule** (CSV)

---

## 💼 Business Problem

Banks and lending institutions face a critical challenge: evaluating millions of loan applications accurately and efficiently. Manual review is slow, expensive, and prone to human bias. Without a fast, objective, data-driven scoring system, institutions risk:

- Approving high-risk borrowers → financial losses and non-performing assets
- Rejecting creditworthy applicants → lost business and competitive disadvantage

**Solution:** Use `loan_int_rate` as a proxy credit risk score. A higher predicted interest rate = higher perceived borrower risk. This enables automated, consistent, and scalable risk assessment at the point of application.

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | Credit Risk Dataset (`credit_risk_dataset.csv`) |
| **Rows** | 32,581 loan records |
| **Original Columns** | 12 features |
| **Target Variable** | `loan_int_rate` (Annual interest rate, %) |

### Raw Features

| Feature | Type | Description |
|---|---|---|
| `person_age` | int | Applicant age in years |
| `person_income` | int | Annual income ($) |
| `person_home_ownership` | categorical | RENT / OWN / MORTGAGE / OTHER |
| `person_emp_length` | float | Employment length (years) |
| `loan_intent` | categorical | Purpose of loan (PERSONAL, EDUCATION, MEDICAL, etc.) |
| `loan_grade` | categorical | Loan grade assigned by the institution (A–G) |
| `loan_amnt` | int | Requested loan amount ($) |
| `loan_int_rate` | float | **Target — Annual interest rate (%)** |
| `loan_status` | int | 0 = No default, 1 = Default |
| `loan_percent_income` | float | Loan amount as a fraction of income |
| `cb_person_default_on_file` | categorical | Prior default on record (Y/N) |
| `cb_person_cred_hist_length` | int | Credit history length (years) |

---

## 🔧 ML Pipeline

The project follows a structured 9-step end-to-end machine learning pipeline:

```
 Data Collection  →  Data Preparation  →  EDA  →  Feature Engineering
       ↓
 SMOTE Balancing  →  Train/Test Split  →  Model Training
       ↓
 Hyperparameter Tuning  →  Evaluation  →  Deployment
```

| Step | Description |
|---|---|
| **1. Data Collection** | Load dataset (32,581 rows × 12 columns) and inspect structure |
| **2. Data Preparation** | Handle missing values, label encoding, duplicate removal, IQR outlier detection |
| **3. EDA** | Histograms, boxplots, scatter plots, correlation heatmap |
| **4. Feature Engineering** | Created 4 new domain-informed derived features |
| **5. SMOTE Balancing** | Fixed 78/22 class imbalance → balanced 50/50 split |
| **6. Train/Test Split** | 80/20 split; StandardScaler applied for Linear Regression |
| **7. Model Training** | Linear Regression, Decision Tree Regressor, Random Forest Regressor |
| **8. Hyperparameter Tuning** | GridSearchCV + 5-Fold Cross-Validation |
| **9. Evaluation** | RMSE, MAE, R², ROC AUC; Predicted vs Actual plots |

---

## 📈 Exploratory Data Analysis

### Categorical Feature Distributions

Understanding the distribution of categorical variables — loan intent, loan grade, home ownership, and default history — provides key context for risk segmentation and model interpretability.

[![Categorical Distributions](https://github.com/OmPatil2806/Credit_risk_prediction/raw/main/categorical_distributions.png)](https://github.com/OmPatil2806/Credit_risk_prediction/blob/main/categorical_distributions.png)

> **Key Observations:**
> - Loan grades B and C dominate the portfolio, suggesting moderate-risk applicants are the majority
> - RENT is the most common home ownership status, associated with higher interest rate tiers
> - Medical and personal loans make up the largest share of loan intents
> - ~18% of applicants have a prior default on file, strongly correlated with higher interest rates

---

## ⚙️ Feature Engineering

Four new domain-informed features were engineered to improve predictive performance:

| Feature | Formula | Business Meaning |
|---|---|---|
| `DebtIncomeRatio` | `loan_amnt / person_income` | Measures loan affordability relative to income |
| `CreditUtilization` | `loan_percent_income × 100` | Percentage of income committed to this loan |
| `LoanBurden` | `loan_amnt × loan_int_rate` | Absolute cost burden of the loan |
| `EmploymentIncomeRatio` | `person_emp_length / (person_income + 1)` | Employment stability relative to income level |

These features capture interactions not explicitly present in raw data and gave the Random Forest model significantly richer signals.

---

## ⚖️ SMOTE — Class Balancing

The original dataset had a significant class imbalance (78% non-default / 22% default), which would have biased the model toward predicting the majority class. **SMOTE (Synthetic Minority Oversampling Technique)** was applied to generate synthetic minority-class samples and produce a balanced 50/50 split before training.

[![SMOTE Balance](https://github.com/OmPatil2806/Credit_risk_prediction/raw/main/smote_balance.png)](https://github.com/OmPatil2806/Credit_risk_prediction/blob/main/smote_balance.png)

> Before SMOTE: **78% Non-Default / 22% Default**
> After SMOTE: **50% Non-Default / 50% Default** — balanced for fair model training

---

## 🏆 Model Training & Evaluation

Three regression models were trained, tuned with GridSearchCV, and evaluated on the held-out test set.

### Performance Metrics

| Model | R² Score | RMSE | MAE | ROC AUC | CV R² |
|---|---|---|---|---|---|
| Linear Regression | 0.7976 | 1.3998 | 1.0741 | ~0.93 | ~0.79 |
| Decision Tree | 0.8237 | 1.3063 | 0.9539 | ~0.95 | ~0.82 |
| ✅ **Random Forest (Best)** | **0.8288** | **1.2873** | **0.9489** | **~0.96** | **0.8227** |

**Random Forest** was selected as the champion model — it achieves the highest R² (82.88%), lowest RMSE (1.2873) and MAE (0.9489). The near-identical CV R² (0.8227) and Test R² (0.8288) confirm excellent generalisation with no overfitting.

### Model Comparison Chart

[![Model Comparison](https://github.com/OmPatil2806/Credit_risk_prediction/raw/main/model_comparison.png)](https://github.com/OmPatil2806/Credit_risk_prediction/blob/main/model_comparison.png)

> **Key Insight:** Random Forest consistently outperforms across all three metrics — for RMSE and MAE lower is better, for R² higher is better.

---

## 📉 Results — Actual vs Predicted

[![Actual vs Predicted](https://github.com/OmPatil2806/Credit_risk_prediction/raw/main/line_predicted_vs_actual.png)](https://github.com/OmPatil2806/Credit_risk_prediction/blob/main/line_predicted_vs_actual.png)

The line chart comparing **Actual vs Predicted** interest rates shows that the Random Forest model closely tracks the true values across the test set. Predictions remain tightly aligned with actuals, with minimal deviation — confirming the model's accuracy and reliability for real-world deployment.

---

## 🖥️ Streamlit Application

The full ML model is deployed as a professional dark-themed fintech web application with four pages.

> **🚀 Live App:** [https://creditriskprediction-dgrpbjdweg27gpaabf4mwo.streamlit.app/](https://creditriskprediction-dgrpbjdweg27gpaabf4mwo.streamlit.app/)

### Application Pages

#### 🏠 Home
Platform overview with feature highlights, step-by-step workflow guide, and navigation cards to all sections.

#### 📋 Loan Application
- Input form: personal details, loan specifics, financial ratios (auto-computed live)
- Instant risk prediction on submission
- Result tab: risk gauge chart, approval probability donut, risk band position chart
- Risk interpretation: APPROVE / CONDITIONAL APPROVAL / REVIEW & DECLINE
- Applicant financial summary table
- **One-click download:** Credit Risk Report (`.txt`) and Application Summary (`.csv`)
- CTA button to jump directly into the Repayment Planner with pre-filled values

#### 💳 Loan Repayment Planner
- Inputs: Loan Amount, Annual Interest Rate (pre-filled from prediction), Loan Term, First Payment Date
- Live EMI calculation using the **reducing-balance formula** (standard banking method)
- Visual summary: Monthly EMI, Principal, Total Interest, Total Repayment
- Cost breakdown progress bar (Principal % vs Interest %)
- First & Last payment date chips
- **Amortisation charts:** Outstanding Balance over time + Cumulative Principal vs Interest stacked area
- Expandable monthly EMI breakdown bar chart
- Full **Installment Schedule** table: No. · Payment Date · EMI · Principal · Interest · Balance
- **Download:** Installment Report (`.txt`) and Full Schedule (`.csv`)

#### 📊 Analytics Dashboard
- Model snapshot KPI cards: R², ROC AUC, RMSE, Feature Count
- Risk band distribution: bar chart + donut (portfolio split)
- Average interest rate by loan grade (A→G)
- Approval probability vs interest rate curve
- Live session prediction summary with gauge and detail breakdown

### Risk Classification

| Risk Level | Interest Rate | Approval Probability |
|---|---|---|
| 🟢 **Low Risk** | < 10% | ~88% |
| 🟡 **Medium Risk** | 10% – 16% | ~62% |
| 🔴 **High Risk** | > 16% | ~28% |

---

## 📁 Project Structure

```
Credit_risk_prediction/
│
├── app.py                          # Streamlit application (CreditLens UI)
├── credit_risk_prediction.ipynb    # Full ML pipeline notebook
├── rf_model.pkl                    # Trained Random Forest model (serialised)
├── credit_risk_dataset.csv         # Raw dataset
│
├── categorical_distributions.png   # EDA — categorical feature distributions
├── smote_balance.png               # Class balance before & after SMOTE
├── model_comparison.png            # RMSE / MAE / R² comparison chart
├── line_predicted_vs_actual.png    # Actual vs Predicted interest rate plot
│
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/OmPatil2806/Credit_risk_prediction.git
cd Credit_risk_prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit pandas numpy matplotlib scikit-learn joblib python-dateutil
```

### 3. Ensure Model File is Present

Make sure `rf_model.pkl` is in the same directory as `app.py`.

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **ML / Data** | Scikit-Learn, Pandas, NumPy |
| **Visualisation** | Matplotlib |
| **Class Balancing** | imbalanced-learn (SMOTE) |
| **Model Serialisation** | Joblib |
| **Web Application** | Streamlit |
| **Date Handling** | python-dateutil |
| **Deployment** | Streamlit Community Cloud |

---

## 👤 Author

**Om Patil**
GitHub: [@OmPatil2806](https://github.com/OmPatil2806)

---

<div align="center">

**⭐ If you found this project useful, please give it a star!**

[![Live App](https://img.shields.io/badge/🚀%20Try%20the%20Live%20App-CreditLens-FF4B4B?style=for-the-badge&logo=streamlit)](https://creditriskprediction-dgrpbjdweg27gpaabf4mwo.streamlit.app/)

</div>
