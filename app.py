import streamlit as st
import pandas as pd
import pickle

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL AND SCALER
# ---------------------------------------------------

model = pickle.load(open('customer_churn_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("📌 About Project")

st.sidebar.info(
    """
    This project predicts customer churn
    using Machine Learning techniques.

    Models Used:
    - Logistic Regression
    - Decision Tree
    - Random Forest

    Technologies:
    - Python
    - Scikit-learn
    - Streamlit
    """
)

# ---------------------------------------------------
# MAIN TITLE
# ---------------------------------------------------

st.title("📊 Customer Churn Prediction Dashboard")

st.markdown(
    """
    Predict whether a telecom customer
    is likely to churn based on
    customer and service information.
    """
)

st.markdown("---")

# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------

st.subheader("📝 Customer Information")

# Create Columns
col1, col2 = st.columns(2)

# ---------------------------------------------------
# COLUMN 1
# ---------------------------------------------------

with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        12
    )

    phoneservice = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    multiplelines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internetservice = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    onlinesecurity = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    onlinebackup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

# ---------------------------------------------------
# COLUMN 2
# ---------------------------------------------------

with col2:

    deviceprotection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

    techsupport = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

    streamingtv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

    streamingmovies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

    paperlessbilling = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

    paymentmethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthlycharges = st.number_input(
        "Monthly Charges",
        min_value=0.0
    )

    totalcharges = st.number_input(
        "Total Charges",
        min_value=0.0
    )

# ---------------------------------------------------
# ENCODING
# ---------------------------------------------------

binary_map = {
    "Yes": 1,
    "No": 0,
    "Male": 1,
    "Female": 0
}

gender = binary_map[gender]
partner = binary_map[partner]
dependents = binary_map[dependents]
phoneservice = binary_map[phoneservice]
paperlessbilling = binary_map[paperlessbilling]

# Multiple Lines Encoding
multiplelines_map = {
    "No": 0,
    "Yes": 1,
    "No phone service": 2
}

multiplelines = multiplelines_map[multiplelines]

# Internet Service Encoding
internet_map = {
    "DSL": 0,
    "Fiber optic": 1,
    "No": 2
}

internetservice = internet_map[internetservice]

# Common Service Encoding
service_map = {
    "No": 0,
    "Yes": 1,
    "No internet service": 2
}

onlinesecurity = service_map[onlinesecurity]
onlinebackup = service_map[onlinebackup]
deviceprotection = service_map[deviceprotection]
techsupport = service_map[techsupport]
streamingtv = service_map[streamingtv]
streamingmovies = service_map[streamingmovies]

# Contract Encoding
contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

contract = contract_map[contract]

# Payment Method Encoding
payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3
}

paymentmethod = payment_map[paymentmethod]

# ---------------------------------------------------
# CREATE DATAFRAME
# ---------------------------------------------------

sample = pd.DataFrame({

    'gender': [gender],
    'SeniorCitizen': [senior],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'PhoneService': [phoneservice],
    'MultipleLines': [multiplelines],
    'InternetService': [internetservice],
    'OnlineSecurity': [onlinesecurity],
    'OnlineBackup': [onlinebackup],
    'DeviceProtection': [deviceprotection],
    'TechSupport': [techsupport],
    'StreamingTV': [streamingtv],
    'StreamingMovies': [streamingmovies],
    'Contract': [contract],
    'PaperlessBilling': [paperlessbilling],
    'PaymentMethod': [paymentmethod],
    'MonthlyCharges': [monthlycharges],
    'TotalCharges': [totalcharges]

})

# ---------------------------------------------------
# SCALING
# ---------------------------------------------------

sample_scaled = scaler.transform(sample)

# ---------------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------------

st.markdown("---")

if st.button("🔍 Predict Churn"):

    prediction = model.predict(sample_scaled)

    probability = model.predict_proba(sample_scaled)[0][1]

    st.subheader("📈 Prediction Result")

    if prediction[0] == 1:

        st.error(
            "⚠️ Customer is likely to churn"
        )

        st.warning(
            "Retention strategies are recommended for this customer."
        )

    else:

        st.success(
            "✅ Customer is likely to stay"
        )

    st.write(
        f"### Churn Probability: {probability:.2f}"
    )

    st.progress(float(probability))

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.caption(
    "Built with Streamlit and Machine Learning"
)