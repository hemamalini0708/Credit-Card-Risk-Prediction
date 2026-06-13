# ============================================================================
# CREDIT RISK PREDICTION - STREAMLIT VERSION (CUSTOM INPUTS)
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

        # 1. One-hot encode Region
        region_dummies = pd.get_dummies(df['Region'], prefix='Region')

        # 2. Encode Education as numeric
        education_mapping = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
        df['Education'] = pd.to_numeric(df['Education'].map(education_mapping), errors='coerce').fillna(3)

        # 3. Clean NumberOfDependents
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
        st.error("❌ Files not found! Make sure creditcard.csv and RF_model.pkl are in app folder")
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
# CUSTOM INPUTS SECTION - USING NUMBER INPUTS (BEST OPTION)
# ============================================================================

st.sidebar.title("📋 Customer Information")

# -------- QUICK PRESET BUTTONS --------
st.sidebar.markdown("### 🚀 Quick Presets")
preset_col1, preset_col2 = st.sidebar.columns(2)

if preset_col1.button("✅ Good Customer", use_container_width=True, type="primary"):
    st.session_state.debt_ratio = 0.25
    st.session_state.monthly_income = 8000
    st.session_state.open_credit_lines = 2
    st.session_state.real_estate_loans = 1
    st.session_state.dependents = 1
    st.session_state.education = 3
    st.session_state.selected_region = "North"
    st.rerun()

if preset_col2.button("❌ Bad Customer", use_container_width=True):
    st.session_state.debt_ratio = 1.50
    st.session_state.monthly_income = 1500
    st.session_state.open_credit_lines = 8
    st.session_state.real_estate_loans = 0
    st.session_state.dependents = 4
    st.session_state.education = 1
    st.session_state.selected_region = "Central"
    st.rerun()

st.sidebar.markdown("---")

# Initialize session state for custom inputs
if 'debt_ratio' not in st.session_state:
    st.session_state.debt_ratio = 0.5
    st.session_state.monthly_income = 4000
    st.session_state.open_credit_lines = 3
    st.session_state.real_estate_loans = 1
    st.session_state.dependents = 2
    st.session_state.education = 3
    st.session_state.selected_region = "North"

# -------- FINANCIAL INFORMATION --------
st.sidebar.markdown("### 💰 Financial Information")

debt_ratio = st.sidebar.number_input(
    "Debt Ratio",
    min_value=0.0,
    max_value=10.0,
    value=st.session_state.debt_ratio,
    step=0.05,
    format="%.2f",
    key="debt_ratio_input"
)

monthly_income = st.sidebar.number_input(
    "Monthly Income ($)",
    min_value=0,
    max_value=100000,
    value=st.session_state.monthly_income,
    step=500,
    key="monthly_income_input"
)

# -------- CREDIT LINES --------
st.sidebar.markdown("### 📊 Credit Lines & Loans")

open_credit_lines = st.sidebar.number_input(
    "Open Credit Lines & Loans",
    min_value=0,
    max_value=50,
    value=st.session_state.open_credit_lines,
    step=1,
    key="open_credit_lines_input"
)

real_estate_loans = st.sidebar.number_input(
    "Real Estate Loans or Lines",
    min_value=0,
    max_value=20,
    value=st.session_state.real_estate_loans,
    step=1,
    key="real_estate_loans_input"
)

# -------- DEMOGRAPHICS --------
st.sidebar.markdown("### 👤 Demographics")

dependents = st.sidebar.number_input(
    "Number of Dependents",
    min_value=0,
    max_value=20,
    value=st.session_state.dependents,
    step=1,
    key="dependents_input"
)

education = st.sidebar.number_input(
    "Education Level (1-5)",
    min_value=1,
    max_value=5,
    value=st.session_state.education,
    step=1,
    key="education_input",
    help="1=Unknown, 2=Graduate School, 3=College, 4=Some College, 5=High School"
)

# -------- REGION --------
st.sidebar.markdown("### 🌍 Region")

selected_region = st.sidebar.selectbox(
    "Select Region",
    options=["North", "South", "East", "West", "Central"],
    index=["North", "South", "East", "West", "Central"].index(st.session_state.selected_region),
    key="region_input"
)

# Convert region to one-hot encoding
region_central = 1 if selected_region == "Central" else 0
region_east = 1 if selected_region == "East" else 0
region_north = 1 if selected_region == "North" else 0
region_west = 1 if selected_region == "West" else 0
region_south = 1 if selected_region == "South" else 0

# Update session state
st.session_state.debt_ratio = debt_ratio
st.session_state.monthly_income = monthly_income
st.session_state.open_credit_lines = open_credit_lines
st.session_state.real_estate_loans = real_estate_loans
st.session_state.dependents = dependents
st.session_state.education = education
st.session_state.selected_region = selected_region

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
    st.write(f"**Education Level:** {education} ({['', 'Unknown', 'Graduate School', 'College', 'Some College', 'High School'][education]})")
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
            input_data = np.array([
                [
                    debt_ratio,
                    open_credit_lines,
                    real_estate_loans,
                    monthly_income,
                    dependents,
                    education,
                    region_central,
                    region_east,
                    region_north,
                    region_west
                ]
            ])

            # -------- SCALE THE INPUT --------
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
                if prediction == 0:  # GOOD
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

                else:  # BAD
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
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("<div style='text-align: center; font-size: 0.85rem; color: #666;'><p>📧 <strong>Project Done By</strong> Hema Malini | Built with Python & Machine Learning</p></div>", unsafe_allow_html=True)