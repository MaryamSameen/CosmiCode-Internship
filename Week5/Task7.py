import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

st.title("ML vs Deep Learning: Model Comparison on Digits Dataset")

st.write("""
This app compares traditional ML algorithms (Logistic Regression, SVM, Decision Tree) with a simple Deep Learning model on the digits dataset.
""")

# Load dataset
digits = load_digits()
X = digits.data
y = digits.target

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)

# SVM
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)

# Deep Learning Model
X_train_dl = X_train.reshape(-1, 8, 8, 1)
X_test_dl = X_test.reshape(-1, 8, 8, 1)
y_train_dl = keras.utils.to_categorical(y_train, 10)
y_test_dl = keras.utils.to_categorical(y_test, 10)

dl_model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(8, 8, 1)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
dl_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
dl_model.fit(X_train_dl, y_train_dl, epochs=10, batch_size=32, verbose=0)
dl_loss, dl_acc = dl_model.evaluate(X_test_dl, y_test_dl, verbose=0)
y_pred_dl = np.argmax(dl_model.predict(X_test_dl, verbose=0), axis=1)

# Results Table
results = pd.DataFrame({
    "Model": ["Logistic Regression", "SVM", "Decision Tree", "Deep Learning (CNN)"],
    "Accuracy": [acc_lr, acc_svm, acc_dt, dl_acc]
})
st.write("## Accuracy Comparison")
st.dataframe(results)

# Classification Reports
st.write("## Classification Reports")
model_choice = st.selectbox("Select model for detailed report:", results["Model"])
if model_choice == "Logistic Regression":
    st.text(classification_report(y_test, y_pred_lr))
elif model_choice == "SVM":
    st.text(classification_report(y_test, y_pred_svm))
elif model_choice == "Decision Tree":
    st.text(classification_report(y_test, y_pred_dt))
else:
    st.text(classification_report(y_test, y_pred_dl))

# Confusion Matrix
st.write("## Confusion Matrix")
fig, ax = plt.subplots()
if model_choice == "Logistic Regression":
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr, ax=ax)
elif model_choice == "SVM":
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_svm, ax=ax)
elif model_choice == "Decision Tree":
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_dt, ax=ax)
else:
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_dl, ax=ax)
st.pyplot(fig)