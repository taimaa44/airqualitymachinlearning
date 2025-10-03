"""

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# --------------------------
# 1️⃣ قراءة البيانات
# --------------------------
df = pd.read_excel("AirQualityUCI (1).xlsx", sheet_name="AirQualityUCI")

# استبدال القيم -200 بـ NaN
df = df.replace(-200, np.nan)

# تعويض القيم المفقودة بالمتوسط للأعمدة الرقمية
df = df.fillna(df.select_dtypes(include="number").mean())

# --------------------------
# 2️⃣ إنشاء عمود datetime
# --------------------------
df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
df = df.sort_values('datetime')
df = df.reset_index(drop=True)

# --------------------------
# 3️⃣ Features و Target
# --------------------------
features = ["T", "RH", "NO2(GT)", "PT08.S5(O3)"]
df['hour'] = df['datetime'].dt.hour
features.append('hour')

target = "CO(GT)"

X = df[features]
y = df[target]

print("X shape:", X.shape)
print("y shape:", y.shape)

# --------------------------
# 4️⃣ تقسيم البيانات
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False  # shuffle=False مهم للسلاسل الزمنية
)

print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)

# --------------------------
# 5️⃣ تدريب Random Forest
# --------------------------
model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# التنبؤ على Test Set
y_pred = model.predict(X_test)

# تقييم الأداء
print("RandomForestRegressor (Forecast)")
print("R²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# --------------------------
# 6️⃣ Forecast على المستقبل (مثال)
# --------------------------
# افترض عندك بيانات المستقبل بنفس الأعمدة + hour
# هنا مثال: استخدام آخر 5 ساعات للتنبؤ
X_future = X_test.tail(5)  # يمكنك تعديلها لأي بيانات مستقبلية
y_future_pred = model.predict(X_future)

print("\nForecast القيم المستقبلية:")
for dt, val in zip(df['datetime'].tail(5), y_future_pred):
    print(f"{dt} -> {val:.2f}")

# --------------------------
# 7️⃣ حفظ الموديل
# --------------------------
joblib.dump(model, "forecast_model_4features_hour.pkl")
print("\nتم حفظ الموديل باسم 'forecast_model_4features_hour.pkl'")

