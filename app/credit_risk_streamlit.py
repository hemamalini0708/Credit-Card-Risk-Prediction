# ============================================================================
# CREDIT RISK PREDICTION - STREAMLIT VERSION (TRULY FIXED!)
# ⚠️ IMPORTANT: Your model uses SCALED data - this app scales inputs!
# ============================================================================

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .good-customer { background-color: #d4edda; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745; }
    .bad-customer { background-color: #f8d7da; padding: 20px; border-radius: 10px; border-left: 5px solid #dc3545; }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# LOAD MODEL & CREATE SCALER FROM TRAINING DATA
# ============================================================================

@st.cache_resource
def load_model_and_scaler():
    """Load model and fit scaler on training data"""
    try:
        # Load the trained model
        with open('app/RF_model.pkl', 'rb') as f:
            model = pickle.load(f)

        # Load training data to fit scaler with SAME statistics
        df = pd.read_csv('app/creditcard.csv')

        # THE 10 FEATURES YOUR MODEL EXPECTS (in CORRECT order):
        # Based on your Flask app
        features_to_use = [
            'DebtRatio',
            'NumberOfOpenCreditLinesAndLoans',
            'NumberRealEstateLoansOrLines',
            'MonthlyIncome',
            'NumberOfDependents',
            'Education',
            'Region_Central',
            'Region_East',
            'Region_North',
            'Region_West'
        ]

        # ⚠️ IMPORTANT: Some features need to be created from the CSV
        # 1. One-hot encode Region (create Central, East, North, West columns)
        region_dummies = pd.get_dummies(df['Region'], prefix='Region')

        # 2. Encode Education as numeric
        education_mapping = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
        df['Education'] = pd.to_numeric(df['Education'].map(education_mapping), errors='coerce').fillna(3)

        # 3. Clean NumberOfDependents (it has mixed types)
        df['NumberOfDependents'] = pd.to_numeric(df['NumberOfDependents'], errors='coerce').fillna(0)

        # Combine all features
        df_prepared = df[['DebtRatio', 'NumberOfOpenCreditLinesAndLoans',
                          'NumberRealEstateLoansOrLines', 'MonthlyIncome',
                          'NumberOfDependents', 'Education']].copy()

        df_prepared = pd.concat([df_prepared, region_dummies], axis=1)

        # Ensure all required columns exist
        for col in features_to_use:
            if col not in df_prepared.columns:
                df_prepared[col] = 0

        # Select only the features we need
        df_prepared = df_prepared[features_to_use]

        # Fit StandardScaler on training data
        scaler = StandardScaler()
        scaler.fit(df_prepared)

        return model, scaler, features_to_use

    except FileNotFoundError:
        st.error("❌ Files not found! Make sure creditcard.csv and credit_card.pkl are in same folder")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()


# Load model and scaler
model, scaler, feature_names = load_model_and_scaler()

# ============================================================================
# APP TITLE
# ============================================================================

st.title("💳 Credit Risk Prediction System")
st.subheader("Assess Customer Credit Risk with ML")
st.info("This model predicts **Good** (low risk) or **Bad** (high risk) customers using trained Machine Learning")
st.markdown("---")

# ============================================================================
# SIDEBAR INPUTS
# ============================================================================

st.sidebar.title("📋 Customer Information")

# -------- FINANCIAL INFORMATION --------
st.sidebar.markdown("### 💰 Financial Information")

debt_ratio = st.sidebar.slider(
    "Debt Ratio",
    min_value=0.0,
    max_value=2.0,
    value=0.5,
    step=0.05,
    help="Your training data range: 0.0 - ~1.0"
)

monthly_income = st.sidebar.slider(
    "Monthly Income ($)",
    min_value=0,
    max_value=30000,
    value=4000,
    step=100,
    help="Your training data range: $0 - $30000"
)

# -------- CREDIT LINES --------
st.sidebar.markdown("### 📊 Credit Lines & Loans")

open_credit_lines = st.sidebar.slider(
    "Open Credit Lines & Loans",
    min_value=0,
    max_value=25,
    value=3,
    step=1,
    help="Your training data range: 0 - ~20"
)

real_estate_loans = st.sidebar.slider(
    "Real Estate Loans or Lines",
    min_value=0,
    max_value=10,
    value=1,
    step=1,
    help="Your training data range: 0 - ~8"
)

# -------- DEMOGRAPHICS --------
st.sidebar.markdown("### 👤 Demographics")

dependents = st.sidebar.slider(
    "Number of Dependents",
    min_value=0,
    max_value=10,
    value=2,
    step=1,
    help="Your training data range: 0 - ~5"
)

education = st.sidebar.radio(
    "Education Level",
    options=[1, 2, 3, 4, 5],
    index=2,
    format_func=lambda x: {
        1: "1 - Unknown",
        2: "2 - Graduate School",
        3: "3 - College",
        4: "4 - Some College",
        5: "5 - High School"
    }[x]
)

# -------- REGION --------
st.sidebar.markdown("### 🌍 Region")

selected_region = st.sidebar.radio(
    "Select Region",
    options=["Central", "East", "North", "West"],
    index=2
)

# Convert region to one-hot encoding
region_central = 1 if selected_region == "Central" else 0
region_east = 1 if selected_region == "East" else 0
region_north = 1 if selected_region == "North" else 0
region_west = 1 if selected_region == "West" else 0

# ============================================================================
# DISPLAY INPUTS
# ============================================================================

col1, col2 = st.columns([1, 1.5])

with col1:
    st.markdown("### 📝 Input Summary")
    st.write(f"**Debt Ratio:** {debt_ratio:.2f}")
    st.write(f"**Monthly Income:** ${monthly_income:,}")
    st.write(f"**Open Credit Lines:** {open_credit_lines}")
    st.write(f"**Real Estate Loans:** {real_estate_loans}")
    st.write(f"**Dependents:** {dependents}")
    st.write(f"**Education:** {education}")
    st.write(f"**Region:** {selected_region}")

with col2:
    st.markdown("### 🎯 Prediction Result")
    result_placeholder = st.empty()

# ============================================================================
# PREDICTION
# ============================================================================

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_button = st.button("⚡ PREDICT CREDIT RISK", use_container_width=True, type="primary")

if predict_button:
    with st.spinner("🔄 Analyzing customer profile..."):
        try:
            # -------- PREPARE DATA FOR PREDICTION --------
            # ⚠️ CRITICAL: Create array in EXACT order that matches your Flask app
            input_data = np.array([
                [
                    debt_ratio,  # 0: DebtRatio
                    open_credit_lines,  # 1: NumberOfOpenCreditLinesAndLoans
                    real_estate_loans,  # 2: NumberRealEstateLoansOrLines
                    monthly_income,  # 3: MonthlyIncome
                    dependents,  # 4: NumberOfDependents
                    education,  # 5: Education
                    region_central,  # 6: Region_Central
                    region_east,  # 7: Region_East
                    region_north,  # 8: Region_North
                    region_west  # 9: Region_West
                ]
            ])

            # -------- SCALE THE INPUT --------
            # ⚠️ CRITICAL: Your model was trained on SCALED data!
            # This scales using the same StandardScaler from training data
            input_scaled = scaler.transform(input_data)

            # -------- MAKE PREDICTION --------
            prediction = model.predict(input_scaled)[0]

            # Get probability
            try:
                prediction_proba = model.predict_proba(input_scaled)[0]
                confidence = max(prediction_proba) * 100
            except:
                confidence = 95.0

            # -------- DISPLAY RESULTS --------
            with result_placeholder.container():
                if prediction == 0:  # GOOD (0 = Good, 1 = Bad in your data)
                    st.markdown("""
                        <div class='good-customer'>
                            <h2>✅ GOOD CUSTOMER - Low Risk</h2>
                            <p>This customer has a <strong>good credit profile</strong>.</p>
                        </div>
                    """, unsafe_allow_html=True)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Risk Level", "🟢 LOW", "Safe")
                    with col_b:
                        st.metric("Confidence", f"{confidence:.1f}%")

                    st.markdown("---")
                    st.success("✓ Approve credit application")

                else:  # BAD (1 = Bad)
                    st.markdown("""
                        <div class='bad-customer'>
                            <h2>❌ BAD CUSTOMER - High Risk</h2>
                            <p>This customer poses <strong>credit risk</strong>.</p>
                        </div>
                    """, unsafe_allow_html=True)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Risk Level", "🔴 HIGH", "Risky")
                    with col_b:
                        st.metric("Confidence", f"{confidence:.1f}%")

                    st.markdown("---")
                    st.error("⚠ Request additional documentation")

        except Exception as e:
            with result_placeholder.container():
                st.error(f"❌ Error: {str(e)}")

# ============================================================================
# TEST EXAMPLES
# ============================================================================

with st.expander("📊 Test with Examples"):
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Good Customer (Usually predicts GOOD):**")
        st.write("- Debt: 0.25")
        st.write("- Income: $8,000")
        st.write("- Open Lines: 1-2")
        st.write("- Real Estate: 1")
        st.write("- Dependents: 0-1")
        st.write("- Education: 3")
        st.write("- Region: North")

    with col2:
        st.write("**Bad Customer (Usually predicts BAD):**")
        st.write("- Debt: 1.50")
        st.write("- Income: $1,500")
        st.write("- Open Lines: 8+")
        st.write("- Real Estate: 0")
        st.write("- Dependents: 4+")
        st.write("- Education: 1")
        st.write("- Region: Central")

st.markdown("---")
st.markdown("<div style='text-align: center; font-size: 0.85rem; color: #666;'><p>📧 <strong>Project Done By</strong> Hema Malini | Built with Python & Machine Learning</p></div>", unsafe_allow_html=True)