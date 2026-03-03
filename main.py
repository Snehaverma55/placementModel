import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("placement_data.csv")

print("Dataset Loaded Successfully ")
print(data.head())

X = data[['CGPA', 'Internship', 'Coding_Rating', 'Projects']]
y = data['Placed']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

print("\nModel Training Completed 🚀")

y_pred = model.predict(X_test_scaled)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and Scaler saved successfully as model.pkl and scaler.pkl 🎯")