import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

# Load dataset
data = pd.read_csv('/mnt/data/diabetes.csv')

# Exploratory Data Analysis (EDA)
st.title("Diabetes Prediction - EDA and Model Deployment")

st.subheader("Dataset Preview")
st.write(data.head())

st.subheader("Dataset Info")
st.write(data.info())

st.subheader("Missing Values")
st.write(data.isnull().sum())

st.subheader("Data Distribution")
st.write(data.describe())

# Data Visualization
st.subheader("Class Distribution")
fig, ax = plt.subplots()
sns.countplot(x='Outcome', data=data, ax=ax)
st.pyplot(fig)

# Splitting data
X = data.drop(columns=['Outcome'])
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model without Data Preparation
model_raw = RandomForestClassifier(random_state=42)
model_raw.fit(X_train, y_train)
preds_raw = model_raw.predict(X_test)
accuracy_raw = accuracy_score(y_test, preds_raw)

# Data Preparation (Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model with Data Preparation
model_prepared = LogisticRegression()
model_prepared.fit(X_train_scaled, y_train)
preds_prepared = model_prepared.predict(X_test_scaled)
accuracy_prepared = accuracy_score(y_test, preds_prepared)

# Select Best Model
best_model = model_raw if accuracy_raw > accuracy_prepared else model_prepared
model_filename = "best_model.pkl"
with open(model_filename, 'wb') as file:
    pickle.dump(best_model, file)

# Deployment with Streamlit
st.subheader("Model Accuracy")
st.write(f"Accuracy without Data Preparation: {accuracy_raw:.2f}")
st.write(f"Accuracy with Data Preparation: {accuracy_prepared:.2f}")

st.subheader("Upload Your Data for Prediction")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)
    if accuracy_raw > accuracy_prepared:
        predictions = best_model.predict(new_data)
    else:
        new_data_scaled = scaler.transform(new_data)
        predictions = best_model.predict(new_data_scaled)
    st.write("Predictions:", predictions)
