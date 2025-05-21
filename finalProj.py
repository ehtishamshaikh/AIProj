# student_performance_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('student-mat.csv', sep=';')

# Combine grade columns to define a final grade
df['G_avg'] = (df['G1'] + df['G2'] + df['G3']) / 3

# Binary classification: Pass (1) if average grade >=10, else Fail (0)
df['pass'] = df['G_avg'].apply(lambda x: 1 if x >= 10 else 0)

# Drop original grades to avoid leakage
df.drop(['G1', 'G2', 'G3', 'G_avg'], axis=1, inplace=True)

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df.drop('pass', axis=1)

y = df['pass']
#spilt data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#train model
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

#use predictionn
log_pred = log_model.predict(X_test_scaled)
rf_pred = rf_model.predict(X_test)

# check models
print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, log_pred))

print("Classification Report:\n", classification_report(y_test, log_pred))

print("\n=== Random Forest Classifier ===")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred))





#below is confusion matrix

def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

plot_confusion(y_test, log_pred, "Logistic Regression - Confusion Matrix")
plot_confusion(y_test, rf_pred, "Random Forest - Confusion Matrix")
