import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

st.title("Pakistan Sub-Division Population Analysis & Prediction")

# 1. Load and preprocess data
df = pd.read_csv('sub-division_population_of_pakistan.csv')
df.columns = df.columns.str.strip()
df.fillna(0, inplace=True)

# 2. Feature engineering
df['URBAN_DOMINATED'] = (df['ALL SEXES (URBAN)'] > df['ALL SEXES (RURAL)']).astype(int)
df['POP_DENSITY'] = (df['ALL SEXES (RURAL)'] + df['ALL SEXES (URBAN)']) / (df['AREA (sq.km)'] + 1)
df['RURAL_RATIO'] = df['ALL SEXES (RURAL)'] / (df['ALL SEXES (RURAL)'] + df['ALL SEXES (URBAN)'] + 1)
df['URBAN_RATIO'] = df['ALL SEXES (URBAN)'] / (df['ALL SEXES (RURAL)'] + df['ALL SEXES (URBAN)'] + 1)

st.header("Exploratory Data Analysis")
# 3. EDA
st.subheader("Population Distribution")
fig, ax = plt.subplots(figsize=(10,5))
sns.histplot(df['ALL SEXES (RURAL)'], bins=50, color='blue', label='Rural', ax=ax)
sns.histplot(df['ALL SEXES (URBAN)'], bins=50, color='red', label='Urban', ax=ax)
plt.legend()
plt.title('Population Distribution')
st.pyplot(fig)

st.subheader("Correlation Heatmap")
fig2, ax2 = plt.subplots(figsize=(12,8))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, fmt=".2f", ax=ax2)
plt.title('Correlation Heatmap')
st.pyplot(fig2)

# 4. Model building and comparison
features = [
    'AREA (sq.km)', 'ALL SEXES (RURAL)', 'ALL SEXES (URBAN)',
    'SEX RATIO (RURAL)', 'SEX RATIO (URBAN)',
    'AVG HOUSEHOLD SIZE (RURAL)', 'AVG HOUSEHOLD SIZE (URBAN)',
    'POP_DENSITY', 'RURAL_RATIO', 'URBAN_RATIO'
]
X = df[features]
y = df['URBAN_DOMINATED']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
st.subheader("Logistic Regression Classification Report")
st.text(classification_report(y_test, y_pred_lr))

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
st.subheader("Random Forest Classification Report")
st.text(classification_report(y_test, y_pred_rf))

# 5. Hyperparameter tuning and cross-validation
st.subheader("Random Forest Hyperparameter Tuning (GridSearchCV)")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
st.write("Best Random Forest Params:", grid.best_params_)
st.write("Best CV Score:", grid.best_score_)

# Save the best model and scaler for deployment (optional, for reuse)
joblib.dump(grid.best_estimator_, 'best_rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# 6. Streamlit Prediction Interface
st.header("Predict Urban or Rural Dominance for a Sub-Division")
with st.form("prediction_form"):
    area = st.number_input("Area (sq.km)", min_value=0.0, value=1000.0)
    rural = st.number_input("Rural Population", min_value=0.0, value=50000.0)
    urban = st.number_input("Urban Population", min_value=0.0, value=50000.0)
    sex_ratio_rural = st.number_input("Sex Ratio (Rural)", min_value=0.0, value=100.0)
    sex_ratio_urban = st.number_input("Sex Ratio (Urban)", min_value=0.0, value=100.0)
    avg_hh_rural = st.number_input("Avg Household Size (Rural)", min_value=0.0, value=6.0)
    avg_hh_urban = st.number_input("Avg Household Size (Urban)", min_value=0.0, value=6.0)
    submit = st.form_submit_button("Predict")

if submit:
    pop_density = (rural + urban) / (area + 1)
    rural_ratio = rural / (rural + urban + 1)
    urban_ratio = urban / (rural + urban + 1)
    features_input = np.array([[area, rural, urban, sex_ratio_rural, sex_ratio_urban, avg_hh_rural, avg_hh_urban, pop_density, rural_ratio, urban_ratio]])
    features_scaled = scaler.transform(features_input)
    pred = grid.best_estimator_.predict(features_scaled)[0]
    result = "Urban Dominated" if pred == 1 else "Rural Dominated"
    st.success(f"Prediction: {result}")