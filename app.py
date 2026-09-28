import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. إعدادات الصفحة
st.set_page_config(
    page_title="ICU Sepsis Clinical Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. تحميل الموديل والـ Scaler
@st.cache_resource
def load_assets():
    try:
        with open('best_sepsis_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler, True
    except Exception as e:
        return None, None, False

model, scaler, is_loaded = load_assets()

# 3. القائمة الجانبية لإدخال البيانات
st.sidebar.title("🩺 Input Patient Parameters")

if is_loaded:
    st.sidebar.success("✅ Model & Scaler Loaded Successfully")
else:
    st.sidebar.error("❌ Failed to load model files from Repository")

st.sidebar.subheader("🩸 Vital Signs & Labs")

hr = st.sidebar.slider("Heart Rate (HR) [bpm]", 30.0, 200.0, 85.0)
o2sat = st.sidebar.slider("Oxygen Saturation (O2Sat) [%]", 50.0, 100.0, 97.0)
temp = st.sidebar.slider("Temperature (°C)", 30.0, 43.0, 37.0)
sbp = st.sidebar.slider("Systolic BP (SBP) [mmHg]", 50.0, 220.0, 120.0)
map_val = st.sidebar.slider("Mean Arterial Pressure (MAP) [mmHg]", 40.0, 150.0, 85.0)
dbp = st.sidebar.slider("Diastolic BP (DBP) [mmHg]", 30.0, 140.0, 75.0)
resp = st.sidebar.slider("Respiration Rate (Resp) [bpm]", 8.0, 50.0, 18.0)
glucose = st.sidebar.slider("Glucose [mg/dL]", 40.0, 500.0, 110.0)
wbc = st.sidebar.slider("WBC [k/µL]", 1.0, 50.0, 9.5)
creatinine = st.sidebar.slider("Creatinine [mg/dL]", 0.2, 12.0, 1.0)
platelets = st.sidebar.slider("Platelets [k/µL]", 10.0, 800.0, 250.0)
age = st.sidebar.slider("Age [years]", 18, 100, 55)

predict_btn = st.sidebar.button("🔍 Predict Sepsis Risk", use_container_width=True)

# 4. الواجهة الرئيسية
st.title("🏥 ICU Sepsis Clinical Risk Prediction")
st.write("This application uses an XGBoost Machine Learning Model to assess the risk of Sepsis in ICU patients based on real-time vital signs and laboratory blood tests.")

st.markdown("---")
st.subheader("🔍 Current Patient Selected Values")

col1, col2, col3 = st.columns(3)

with col1:
    st.write(f"**HR:** {hr} bpm")
    st.write(f"**O2Sat:** {o2sat}%")
    st.write(f"**Temp:** {temp} °C")
    st.write(f"**SBP:** {sbp} mmHg")

with col2:
    st.write(f"**MAP:** {map_val} mmHg")
    st.write(f"**DBP:** {dbp} mmHg")
    st.write(f"**Resp:** {resp} bpm")
    st.write(f"**WBC:** {wbc} k/µL")

with col3:
    st.write(f"**Glucose:** {glucose} mg/dL")
    st.write(f"**Creatinine:** {creatinine} mg/dL")
    st.write(f"**Platelets:** {platelets} k/µL")
    st.write(f"**Age:** {age} years")

st.markdown("---")

# 5. معالجة التنبؤ وتمرير القيم بأسماء الأعمدة الصحيحة
if predict_btn:
    if not is_loaded:
        st.error("Model is not loaded. Please check model files on GitHub.")
    else:
        user_inputs = {
            'HR': hr,
            'O2Sat': o2sat,
            'Temp': temp,
            'SBP': sbp,
            'MAP': map_val,
            'DBP': dbp,
            'Resp': resp,
            'Glucose': glucose,
            'WBC': wbc,
            'Creatinine': creatinine,
            'Platelets': platelets,
            'Age': age
        }

        try:
            # الحصول على قائمة أسماء الأعمدة المتوقعة من الموديل أو الـ Scaler
            feature_names = None
            if hasattr(model, "feature_names_in_"):
                feature_names = list(model.feature_names_in_)
            elif hasattr(scaler, "feature_names_in_"):
                feature_names = list(scaler.feature_names_in_)

            if feature_names:
                # إنشاء DataFrame يحتوي على كافة الأعمدة المطلوبة بالترتيب الصحيح
                df_input = pd.DataFrame(columns=feature_names)
                df_input.loc[0] = 0.0  # التعبئة المبدئية بأصفار
                
                # وضع قيم المستخدم في الأعمدة المطابقة فقط
                for col, val in user_inputs.items():
                    if col in df_input.columns:
                        df_input.loc[0, col] = val
            else:
                # في حالة عدم وجود أسماء محددة
                df_input = pd.DataFrame([user_inputs])

            # تطبيق الـ Scaler والتنبؤ
            try:
                input_scaled = scaler.transform(df_input)
            except Exception:
                input_scaled = df_input

            prediction = model.predict(input_scaled)[0]
            
            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(input_scaled)[0][1]
            else:
                probability = float(prediction)

            st.header("📊 Diagnostic Assessment Result")
            res_col1, res_col2 = st.columns([2, 1])

            with res_col1:
                if prediction == 1 or probability > 0.5:
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
