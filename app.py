# app.py
# -*- coding: utf-8 -*-

import streamlit as st
import joblib
import numpy as np

# --------------------------
# 1️⃣ تحميل الموديل
# --------------------------
model = joblib.load("air_quality_model_4features.joblib")

# --------------------------
# 2️⃣ واجهة Streamlit
# --------------------------
st.title("🌍 التنبؤ بجودة الهواء (CO) باستخدام Random Forest")

st.write("أدخل قيم المتغيرات التالية للتنبؤ بـ CO(GT):")

T_value = st.number_input("درجة الحرارة (T)", min_value=-50.0, max_value=60.0, value=20.0)
RH_value = st.number_input("الرطوبة النسبية (%) (RH)", min_value=0.0, max_value=100.0, value=50.0)
NO2_value = st.number_input("تركيز NO2", min_value=0.0, max_value=500.0, value=30.0)
O3_value = st.number_input("تركيز O3", min_value=0.0, max_value=500.0, value=20.0)

# --------------------------
# 3️⃣ التنبؤ عند الضغط على الزر
# --------------------------
if st.button("تنبؤ"):
    X_new = np.array([[T_value, RH_value, NO2_value, O3_value]])
    prediction = model.predict(X_new)
    st.success(f"🔹 القيمة المتوقعة لـ CO(GT): {prediction[0]:.2f}")
