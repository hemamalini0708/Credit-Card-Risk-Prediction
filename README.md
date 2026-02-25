#  Credit Risk Prediction System

An end-to-end Machine Learning project that predicts whether a customer is a **Good** or **Bad** credit risk based on financial and demographic attributes.

This project follows a structured ML pipeline including data preprocessing, feature engineering, model training, evaluation, and deployment using Streamlit.

![App Screenshot](https://i0.wp.com/dataaspirant.com/wp-content/uploads/2020/09/1-Credit-card-fraud-detection-with-classification-algorithms.png?w=750&ssl=1)

---

##  Problem Statement

Banks need to identify high-risk customers before approving loans.

The objective of this project is to build a classification model that segments customers into:

- **Good Customer (Low Risk)**
- **Bad Customer (High Risk)**

This helps financial institutions reduce default risk and improve decision-making.

---

##  Dataset

The dataset contains customer financial and demographic information including:

- Debt Ratio  
- Monthly Income  
- Number of Open Credit Lines  
- Real Estate Loans  
- Number of Dependents  
- Education Level  
- Region  
- NPA Status (Target Variable)

### 🎯 Target Variable

- **Good → 1**
- **Bad → 0**

---

##  Machine Learning Pipeline

The project strictly follows a production-style ML workflow:

###  Data Preprocessing

- Removed invalid rows  
- Handled missing values using random sampling  
- Log transformation on skewed features  
- Outlier treatment using quantile capping  
- Train-test split (before feature engineering to prevent data leakage)  

---

###  Feature Engineering

- One-Hot Encoding for nominal features  
- Ordinal Encoding for ordered categorical features  
- Variance Threshold feature selection  
- Hypothesis testing (p-value filtering)  
- SMOTE for handling class imbalance  
- StandardScaler for feature scaling  

---

###  Model Training

Models implemented:

- K-Nearest Neighbors  
- Naive Bayes  
- Logistic Regression  
- Decision Tree  
- Random Forest  

---

###  Model Evaluation

- Accuracy Score  
- Confusion Matrix  
- Classification Report  
- ROC Curve comparison  

<img width="745" height="304" alt="image" src="https://github.com/user-attachments/assets/dfd82087-9a05-4bc8-bbfb-149b85f1b811" />

---

###  Final Model Selection

Based on performance comparison and ROC analysis, the best performing model was selected and saved as:

credit_card.pkl

---

##  Deployment

The trained model is deployed using **Streamlit**.

The web app allows users to:

- Enter financial details  
- Automatically scale inputs  
- Predict credit risk  
- Display risk level with confidence score  

### Run Locally

```bash
pip install -r requirements.txt
streamlit run app/credit_risk_streamlit.py
```

**Tech Stack**

- Python
- Pandas
- NumPy
- Scikit-Learn
- Imbalanced-Learn (SMOTE)
- Matplotlib
- Streamlit

**credit-risk-prediction/**
```
│
├── data/
├── models/
├── src/
├── app/
├── README.md
├── requirements.txt
└── .gitignore
```

**Business Impact**

- Reduces loan default risk
- Supports automated credit approval decisions
- Enables risk-based customer segmentation
- Improves financial risk management strategy

**Author**

Hema Malini Gangumalla

Aspiring Data Scientist

📧 hemamalinig07@gmail.com
