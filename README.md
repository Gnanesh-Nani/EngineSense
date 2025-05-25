# 🔧 EngineSense

This project is a machine learning pipeline that predicts **engine condition** (healthy or faulty) based on various engine sensor parameters like RPM, oil pressure, fuel pressure, temperature, etc. It uses **classification algorithms** and **SMOTE** to handle class imbalance, followed by **hyperparameter tuning** and model selection.

---

## 📁 Dataset

The dataset (`engine_data.csv`) contains engine sensor readings and a binary target variable:
- **Features:**
  - `Engine rpm`
  - `Lub oil pressure`
  - `Fuel pressure`
  - `Coolant pressure`
  - `lub oil temp`
  - `Coolant temp`
- **Target:**
  - `Engine Condition` (0 = Healthy, 1 = Faulty)

---

## 🧪 Models Used

- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

Best performing model: **Random Forest** (after GridSearchCV)
```python
{'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200}
```

--- 

## ⚙️ Pipeline Overview

1. **Data Loading**  
2. **Preprocessing:**
   - Feature scaling with `StandardScaler`
   - Train-test split
   - Handling class imbalance using `SMOTE`
3. **Model Training & Evaluation:**
   - Fit multiple models
   - Compare accuracy on test set
4. **Hyperparameter Tuning:**
   - GridSearchCV on Random Forest
5. **Final Evaluation:**
   - Classification report
   - Confusion matrix
   - Cross-validation score
   - Feature importance plot
6. **Model Saving:**
   - Save best model and scaler using `joblib`

---

## 📊 Evaluation Metrics

```
📈 Classification Report:
              precision    recall  f1-score   support

           0       0.50      0.56      0.53      1459
           1       0.72      0.67      0.69      2448

    accuracy                           0.63      3907
   macro avg       0.61      0.61      0.61      3907
weighted avg       0.64      0.63      0.63      3907

🧮 Confusion Matrix:
[[ 813  646]
 [ 810 1638]]
    
📉 Cross-Validated Accuracy: 65.29%
```

---

## 📂 Files

| File                     | Description                              |
|--------------------------|------------------------------------------|
| `engine_data.csv`        | Input dataset                            |
| `engine_condition_prediction.py`  | Training script                          |
| `engine_condition_model.pkl` | Trained best model (Random Forest) |
| `scaler.pkl`             | Scaler used during training              |

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
```

### 2. Execute training
```bash
python engine_condition_prediction.py
```

### 3. Run Streamlit app
```bash
streamlit run app.py
```
