
# Employee Attrition Predictor - Enhanced Dashboard


import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# 1️⃣ Load Trained Pipeline
# -------------------------------
pipeline_path = "employee_attrition_pipeline_11feat.pkl"
pipeline = joblib.load(pipeline_path)

# -------------------------------
# 2️⃣ Page Setup
# -------------------------------
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Employee Attrition Predictor")
st.markdown(
    """
    Predict if an employee is likely to **leave the company** based on key features.
    Adjust inputs and see predictions and risk indicators instantly!
    """
)

# -------------------------------
# 3️⃣ Sidebar - User Inputs
# -------------------------------
st.sidebar.header("Employee Information")

def user_input_features():
    OverTime = st.sidebar.radio("OverTime", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    TotalWorkingYears = st.sidebar.slider("Total Working Years", 0, 50, 5)
    JobLevel = st.sidebar.selectbox("Job Level", [1, 2, 3, 4, 5])
    YearsAtCompany = st.sidebar.slider("Years at Company", 0, 40, 3)
    MonthlyIncome = st.sidebar.slider("Monthly Income", 1000, 50000, 5000, step=500)
    Age = st.sidebar.slider("Age", 18, 65, 30)
    MaritalStatus = st.sidebar.selectbox("Marital Status", [0,1,2], format_func=lambda x: ["Single","Married","Divorced"][x])
    YearsWithCurrManager = st.sidebar.slider("Years with Current Manager", 0, 20, 2)
    YearsInCurrentRole = st.sidebar.slider("Years in Current Role", 0, 20, 2)
    JobSatisfaction = st.sidebar.selectbox("Job Satisfaction", [1,2,3,4])
    WorkLifeBalance = st.sidebar.selectbox("Work-Life Balance", [1,2,3,4])

    data = {
        'OverTime': OverTime,
        'TotalWorkingYears': TotalWorkingYears,
        'JobLevel': JobLevel,
        'YearsAtCompany': YearsAtCompany,
        'MonthlyIncome': MonthlyIncome,
        'Age': Age,
        'MaritalStatus': MaritalStatus,
        'YearsWithCurrManager': YearsWithCurrManager,
        'YearsInCurrentRole': YearsInCurrentRole,
        'JobSatisfaction': JobSatisfaction,
        'WorkLifeBalance': WorkLifeBalance
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# -------------------------------
# 4️⃣ Input Validation
# -------------------------------
def validate_input(df):
    errors = []
    if df['YearsAtCompany'][0] > df['TotalWorkingYears'][0]:
        errors.append("❌ Years at Company cannot exceed Total Working Years.")
    if df['YearsWithCurrManager'][0] > df['YearsAtCompany'][0]:
        errors.append("❌ Years with Current Manager cannot exceed Years at Company.")
    if df['YearsInCurrentRole'][0] > df['TotalWorkingYears'][0]:
        errors.append("❌ Years in Current Role cannot exceed Total Working Years.")
    return errors

errors = validate_input(input_df)

# -------------------------------
# 5️⃣ Main Prediction
# -------------------------------
st.subheader("Prediction & Risk Assessment")

if errors:
    st.error("Please correct the following input inconsistencies:")
    for err in errors:
        st.write(f"- {err}")
else:
    if st.button("Predict Attrition"):
        prediction = pipeline.predict(input_df)[0]
        prediction_proba = pipeline.predict_proba(input_df)[0][1]

        # # Probability bar
        # st.subheader("Attrition Probability")
        # st.progress(int(prediction_proba*100))
        #  Probability: {prediction_proba:.2f}"
        # Probability: {prediction_proba:.2f}

        # Prediction result
        if prediction == 1:
            st.error(f"⚠️ Likely to **leave**.")
        else:
            st.success(f"✅ Likely to **stay**. ")

        # Risk Indicators
        risk_messages = []
        if input_df['OverTime'][0] == 1:
            risk_messages.append("⚠️ Employee is doing overtime frequently.")
        if input_df['JobSatisfaction'][0] <= 2:
            risk_messages.append("⚠️ Low job satisfaction may increase attrition risk.")
        if input_df['WorkLifeBalance'][0] <= 2:
            risk_messages.append("⚠️ Poor work-life balance may increase attrition risk.")

        if risk_messages:
            st.warning("**Key Risk Indicators:**")
            for msg in risk_messages:
                st.write(f"- {msg}")

# -------------------------------
# 6️⃣ Display Input Data
# -------------------------------
st.subheader("Employee Data Entered")
col1, col2 = st.columns(2)
for i, col in enumerate(input_df.columns):
    if i % 2 == 0:
        col1.write(f"**{col}**: {input_df[col].values[0]}")
    else:
        col2.write(f"**{col}**: {input_df[col].values[0]}")

# -------------------------------
# 7️⃣ Footer
# -------------------------------
st.markdown("---")
st.markdown("Made with ❤️ ")
