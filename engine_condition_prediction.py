# train_engine_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
df = pd.read_csv('engine_data.csv')

print("📄 First 5 rows:")
print(df.head())

print("\n🔍 Info:")
print(df.info())

# Features and target
X = df.drop('Engine Condition', axis=1)
y = df['Engine Condition']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("\n⚖️ Class distribution after SMOTE:")
print(pd.Series(y_train_res).value_counts())

# Try different models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

print("\n📊 Model Comparison:")
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    acc = model.score(X_test, y_test)
    print(f"{name}: {acc:.2f}")

# Grid search on Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid.fit(X_train_res, y_train_res)

print("\n✅ Best Parameters:")
print(grid.best_params_)

# Final evaluation
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred))

print("🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross-validation score
cv_scores = cross_val_score(best_model, X_scaled, y, cv=5)
print("\n📉 Cross-Validated Accuracy: {:.2f}%".format(cv_scores.mean() * 100))

# Feature importance
importances = best_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# Save model and scaler
joblib.dump(best_model, 'engine_condition_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\n💾 Model and scaler saved as 'engine_condition_model.pkl' and 'scaler.pkl'")
