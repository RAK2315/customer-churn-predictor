# Customer Churn Prediction Project

## Project Overview

This project focuses on predicting customer churn for a telecom company. The primary goal is to **identify customers likely to churn** and provide actionable insights to reduce churn. While the project includes a **Streamlit frontend for dataset exploration**, the **main emphasis is on the ML workflow**, feature engineering, and model performance.

---

## Dataset
## Dataset  
This project uses the **Telco Customer Churn**: [Customer Churn Prediction – Kaggle](https://www.kaggle.com/code/bhartiprasad17/customer-churn-prediction/data)  
containing information about 7,043 customers, including:
- Customer demographics (gender, senior citizen status, partner/dependent)
- Account information (tenure, contract type, payment method)
- Service usage details (phone service, internet service, streaming, tech support)
- Target: `Churn` (Yes/No)

---

## Features Explored

- **Categorical:** gender, Partner, Dependents, InternetService, Contract, PaymentMethod, etc.
- **Numerical:** tenure, MonthlyCharges, TotalCharges
- **Target variable:** Churn (Yes/No)

---

## Streamlit App (Optional Frontend)

The Streamlit app allows:

- **Dataset exploration** with filters and summary statistics
- **Visualizations**:
  - Churn distribution
  - Tenure vs Churn
  - MonthlyCharges and TotalCharges distributions
  - Contract type & Internet service analysis
  - Correlation heatmap
  - Custom scatter plots

**Note:** The app is optional; the ML workflow is the core of the project.

---

## Machine Learning Workflow

### Preprocessing

1. Drop unnecessary columns (`customerID`)
2. One-hot encoding for categorical features
3. Train-test split (90%-10%)
4. Handle class imbalance using **Balanced Accuracy** metric

### Model Selection

Tested several models:

| Model             | Best CV Balanced Accuracy | Test Set Churn Recall | Test Set Balanced Accuracy |
|------------------|-------------------------|---------------------|---------------------------|
| Decision Tree     | 0.7157                  | 0.44                | 0.66                      |
| Random Forest     | 0.7148                  | 0.46                | 0.69                      |
| Gradient Boosting | 0.7194                  | 0.50                | 0.70                      |

**Gradient Boosting Classifier (GBC)** was selected as the **best model**.

### Hyperparameter Tuning (Grid Search)

- `learning_rate`: 0.1  
- `max_depth`: 3  
- `n_estimators`: 100  

### Model Evaluation

- **Classification Report**:

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| No    | 0.87      | 0.90   | 0.89     | 557     |
| Yes   | 0.57      | 0.50   | 0.53     | 147     |
| **Accuracy** | - | - | 0.82 | 704 |
| **Macro avg** | 0.72 | 0.70 | 0.71 | 704 |
| **Weighted avg** | 0.81 | 0.82 | 0.81 | 704 |

- **Confusion Matrix**: Highlights correct predictions of churn and non-churn customers

---

## Business Implications

- **Target high-risk customers:** True positives (predicted to churn) can be offered promotions or personalized retention campaigns.
- **Prioritize churn reduction:** Focus on new customers and those with higher monthly charges, as they are more likely to churn.
- **Balanced Accuracy:** 0.70, meaning the model fairly accounts for both churned and retained customers.
- **Churn Recall:** 50%, which ensures that a significant portion of churners are correctly identified for action.

---

## How to Run the Project

### Streamlit App (Optional)

```bash
# Clone repo
git clone https://github.com/YourUsername/customer-churn-predictor.git
cd customer-churn-predictor

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# or source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run Home.py
