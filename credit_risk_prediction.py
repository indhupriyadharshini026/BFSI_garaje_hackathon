# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.use_label_encoder.')

# 2. LOAD DATA

df = pd.read_csv('data/german_credit_data.csv')

selected_features = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account',
                     'Credit amount', 'Duration', 'Purpose', 'Risk']
df = df[selected_features]

df.fillna('unknown', inplace=True)

label_encoders = {}
categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Purpose', 'Checking account']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

target_encoder = LabelEncoder()
df['Risk'] = target_encoder.fit_transform(df['Risk'])
label_encoders['Risk'] = target_encoder

X = df.drop('Risk', axis=1)
y = df['Risk']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. TRAIN LOGISTIC REGRESSION
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

# 4. TRAIN RANDOM FOREST
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# 5. HYPERTUNE AND TRAIN XGBOOST
param_dist = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# 🔥 Modified here: Removed use_label_encoder=False
xgb = XGBClassifier(eval_metric='logloss', random_state=42)

random_search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=20, cv=3,
                                   scoring='f1', random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

xgb_model = random_search.best_estimator_

# 6. EVALUATE MODELS

models = {
    'Logistic Regression': lr_model,
    'Random Forest': rf_model,
    'XGBoost': xgb_model
}

results = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    }

# 7. FIND THE BEST MODEL
best_model_name = max(results, key=lambda x: results[x]['F1 Score'])
best_model = models[best_model_name]

# 8. SAVE ARTIFACTS

pickle.dump(best_model, open('model/model.pkl', 'wb'))
pickle.dump(scaler, open('model/scaler.pkl', 'wb'))
pickle.dump(list(X.columns), open('model/features.pkl', 'wb'))

# 9. PRINT RESULTS

print(f"\nBest Model: {best_model_name}")
print(f"Accuracy of Best Model: {results[best_model_name]['Accuracy']:.4f}\n")

for model_name, metrics in results.items():
    print(f"{model_name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    print()

# 10. PLOT ROC CURVES AND SAVE TO outputs FOLDER

os.makedirs('outputs', exist_ok=True)

plt.figure(figsize=(10, 8))

for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for All Models')
plt.legend(loc='lower right')
plt.grid(True)

plt.savefig('outputs/roc_curve.png')
plt.close()

# 11. CLASSIFICATION REPORT FOR BEST MODEL

y_pred_best = best_model.predict(X_test)
print("Classification Report for Best Model:\n")
print(classification_report(y_test, y_pred_best))