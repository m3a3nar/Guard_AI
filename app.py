import streamlit as st
import pandas as pd
import numpy as np
import pickle
st.set_page_config(
    page_title="Sepsis Risk Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)
@st.cache_resource
def load_artifacts():
    with open('best_sepsis_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_artifacts()
    st.sidebar.success("✅ Model & Scaler Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"❌ Error loading files: {e}")
# ==========================================
st.title("🏥 Sepsis GuardAI")
st.markdown("""
This application uses an **XGBoost Machine Learning Model** to assess the risk of **Sepsis** in ICU patients 
based on real-time vital signs and laboratory blood tests.
""")
st.divider()
st.header("📋 Patient Clinical Parameters")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🫀 Vital Signs")
    hr = st.number_input("Heart Rate (HR) [bpm]", min_value=20.0, max_value=220.0, value=85.0, step=1.0)
    o2sat = st.number_input("Oxygen Saturation (O2Sat) [%]", min_value=50.0, max_value=100.0, value=97.0, step=0.5)
    temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0, step=0.1)
    sbp = st.number_input("Systolic BP (SBP) [mmHg]", min_value=40.0, max_value=250.0, value=120.0, step=1.0)
    map_val = st.number_input("Mean Arterial Pressure (MAP) [mmHg]", min_value=20.0, max_value=200.0, value=85.0, step=1.0)
    dbp = st.number_input("Diastolic BP (DBP) [mmHg]", min_value=20.0, max_value=180.0, value=75.0, step=1.0)
    resp = st.number_input("Respiration Rate (Resp) [bpm]", min_value=5.0, max_value=60.0, value=18.0, step=1.0)

with col2:
    st.subheader("🧪 Laboratory Values")
    wbc = st.number_input("White Blood Cells (WBC) [k/μL]", min_value=0.1, max_value=100.0, value=9.5, step=0.1)
    glucose = st.number_input("Glucose [mg/dL]", min_value=20.0, max_value=50.0, value=110.0, step=1.0)
    creatinine = st.number_input("Creatinine [mg/dL]", min_value=0.1, max_value=20.0, value=1.0, step=0.1)
    platelets = st.number_input("Platelets [k/μL]", min_value=5.0, max_value=1000.0, value=250.0, step=5.0)
    ph = st.number_input("Blood pH", min_value=6.5, max_value=8.0, value=7.4, step=0.01)
    lactate = st.number_input("Lactate [mmol/L]", min_value=0.1, max_value=30.0, value=1.5, step=0.1)

with col3:
    st.subheader("👤 Demographics & ICU Stay")
    age = st.number_input("Patient Age [Years]", min_value=1, max_value=110, value=55, step=1)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    iculos = st.number_input("ICU Length of Stay (ICULOS) [Hours]", min_value=1, max_value=500, value=12, step=1)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Predict Sepsis Risk", type="primary", use_container_width=True)

gender_val = 1 if gender == "Male" else 0
if predict_btn:
    input_data = pd.DataFrame([{
        'HR': hr,
        'O2Sat': o2sat,
        'Temp': temp,
        'SBP': sbp,
        'MAP': map_val,
        'DBP': dbp,
        'Resp': resp,
        'WBC': wbc,
        'Glucose': glucose,
        'Creatinine': creatinine,
        'Platelets': platelets,
        'Age': age,
        'ICULOS': iculos
    }])

    # ملحوظة: إذا كان النموذج يدعم ميزات إضافية تم تحضيرها أثناء التدريب، تأكدي من تطابق الأعمدة هنا مع X_train.columns
    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.divider()
        st.header("📊 Diagnostic Assessment Result")

        res_col1, res_col2 = st.columns([2, 1])

        with res_col1:
            if prediction == 1:
                st.error("🚨 **HIGH RISK OF SEPSIS DETECTED**")
                st.warning("Immediate clinical evaluation and monitoring recommended.")
            else:
                st.success("✅ **LOW RISK / NO SEPSIS DETECTED**")
                st.info("Patient vital parameters are within acceptable risk thresholds.")

        with res_col2:
            st.metric(
                label="Sepsis Probability Score",
                value=f"{probability * 100:.2f}%",
                delta="High Danger" if probability > 0.5 else "Stable State",
                delta_color="inverse"
            )

    except Exception as err:
        st.error(f"Error during prediction: {err}")