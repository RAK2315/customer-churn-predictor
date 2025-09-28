import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="ML Workflow & EDA",page_icon="⚙️")
st.title("⚙️ ML Workflow & EDA")

# ----------------------------
# Load dataset robustly
# ----------------------------
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "..", "Telco-Customer-Churn.csv")
df = pd.read_csv(csv_path)

# ----------------------------
# EDA Section
# ----------------------------
st.subheader("Exploratory Data Analysis (EDA)")

# Churn distribution
st.write("### Churn Distribution")
fig, ax = plt.subplots(figsize=(3,3))
sns.countplot(data=df, x="Churn", ax=ax)
st.pyplot(fig)
st.write("**Inference:** Most customers do not churn. The dataset is slightly imbalanced, so we will focus on metrics like Balanced Accuracy.")

# Tenure vs Churn
st.write("### Tenure vs Churn")
fig2, ax2 = plt.subplots(figsize=(4,2))
sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", bins=30, ax=ax2)
plt.tight_layout()
st.pyplot(fig2)
st.write("**Inference:** Customers with shorter tenure are more likely to churn. Retention strategies should target new customers.")

# Monthly Charges vs Churn
st.write("### Monthly Charges vs Churn")
fig3, ax3 = plt.subplots(figsize=(4,2))
sns.kdeplot(data=df[df['Churn']=='Yes'], x="MonthlyCharges", label='Churn', fill=True)
sns.kdeplot(data=df[df['Churn']=='No'], x="MonthlyCharges", label='No Churn', fill=True)
ax3.legend()
st.pyplot(fig3)
st.write("**Inference:** Customers with higher monthly charges have slightly higher churn probability. Pricing or incentives could help retain them.")

# ----------------------------
# ML Model Logic Explanation
# ----------------------------
st.subheader("ML Model Logic: Gradient Boosting Classifier")

st.write("""
1. **Preprocessing:** Dropped `customerID`, applied one-hot encoding to categorical features.  
2. **Train/Test Split:** 90% training, 10% testing to validate model performance.  
3. **Model Choice:** Gradient Boosting Classifier chosen due to strong performance on imbalanced datasets.  
4. **Hyperparameter Tuning:** GridSearchCV used to find best combination of `n_estimators`, `learning_rate`, and `max_depth`.
""")


# ----------------------------
# ML Model Training (Optional demo)
# ----------------------------
if st.checkbox("Run ML Model Training (may take a few minutes)"):
    st.write("Training Gradient Boosting Classifier with Grid Search...")
    
    # Show spinner while training
    with st.spinner("⏳ Training in progress... Please wait."):
        # Prepare data
        X = df.drop(['Churn','customerID'], axis=1)
        X = pd.get_dummies(X, drop_first=True)
        y = df['Churn']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=101
        )

        # Initialize model & grid search
        gbc = GradientBoostingClassifier(random_state=101)
        gbc_params = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 4]
        }

        gbc_grid = GridSearchCV(
            gbc, param_grid=gbc_params, cv=5, 
            scoring='balanced_accuracy', n_jobs=-1, verbose=0
        )
        gbc_grid.fit(X_train, y_train)

    # Training complete message
    st.success("✅ ML Training Completed!")

    best_gbc = gbc_grid.best_estimator_
    y_pred = best_gbc.predict(X_test)

    st.write("### Best Gradient Boosting Parameters")
    st.write(gbc_grid.best_params_)

    st.write("### Classification Report on Test Set")
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.style.format("{:.2f}"))

    st.write("### Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(2.5,2))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

# ----------------------------
# Conclusion & Inferences
# ----------------------------
st.write("")  # 1 line gap
st.write("") 
st.write("") 
st.write("") 
st.write("") 

st.subheader("📌 Project Conclusion & Business Implications")
st.write("""
### Key Findings
- **Contract Type & Tenure:** Month-to-month contracts and shorter tenure customers have higher churn rates.  
- **Demographics:** Customers without partners or dependents, and those who are senior citizens, are more likely to churn.  
- **Services & Payment:** Electronic check users and customers with fewer add-on services (e.g., online security, backup) churn more frequently.  

### Model Performance
- **Best Model:** Grid-Search Gradient Boosting Classifier (GBC)  
- **Balanced Accuracy:** 0.70 on the test set, showing fair handling of class imbalance.  
- **Churn Recall:** 50% of actual churns correctly identified — key for proactive retention.  
- **True Positives:** 73 high-risk customers correctly predicted.  
- **False Negatives:** 74 customers missed — opportunity for model improvement.

### Business Implications
- High-risk customers can be **targeted with retention campaigns**, e.g., discounts or personalized services.  
- **Pricing & Bundles:** Consider revising pricing or offering add-ons to reduce churn among at-risk segments.  
- **Retention Strategy:** Focus on month-to-month, high monthly charges, and service-limited customers.  
- **Monitoring:** Continuous model retraining is recommended as customer behavior evolves.

### Next Steps
- Collect additional behavioral features for better predictive power.  
- Fine-tune models for higher churn recall.  
- Deploy as a **dashboard** for real-time monitoring of churn predictions.  

### Hyperparameters of Best Model
- `learning_rate = 0.1`  
- `max_depth = 3`  
- `n_estimators = 100`
""")
