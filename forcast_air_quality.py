# app_forecast.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --------------------------
# 1️⃣ تحميل البيانات والموديل
# --------------------------
df = pd.read_excel("AirQualityUCI (1).xlsx", sheet_name="AirQualityUCI")
df = df.replace(-200, np.nan)
df = df.fillna(df.select_dtypes(include="number").mean())

# دمج التاريخ والوقت
df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
df = df.sort_values('datetime').reset_index(drop=True)
df['hour'] = df['datetime'].dt.hour

# Features و Target
features = ["T", "RH", "NO2(GT)", "PT08.S5(O3)", "hour"]
target = "CO(GT)"

X = df[features]
y = df[target]

# تحميل الموديل المدرب
model = joblib.load("forecast_model_4features_hour.pkl")

# --------------------------
# 2️⃣ واجهة المستخدم
# --------------------------
st.title("🌍 Forecast جودة الهواء (CO) كـ Time Series")

st.write("يمكنك رؤية التنبؤات المستقبلية باستخدام Random Forest")

# اختر عدد الساعات للتنبؤ
n_forecast = st.slider("عدد الساعات المستقبلية للتنبؤ:", min_value=1, max_value=24, value=5)

# استخدام آخر n_forecast صفوف كمدخلات
X_future = X.tail(n_forecast)
datetime_future = df['datetime'].tail(n_forecast)

# التنبؤ
y_future_pred = model.predict(X_future)

# --------------------------
# 3️⃣ عرض Forecast كـ Time Series
# --------------------------
forecast_df = pd.DataFrame({
    'datetime': datetime_future,
    'Predicted_CO': y_future_pred
})

st.subheader("📈 الرسم البياني للتنبؤ")
st.line_chart(forecast_df.set_index('datetime'))

# --------------------------
# 4️⃣ عرض القيم التنبؤية
# --------------------------
st.subheader("القيم التنبؤية")
st.write(forecast_df)
