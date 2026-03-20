"""
MADUT'S RANDOM FOREST MODEL
APP4080 Assignment - Part A
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, mean_absolute_error,
                           mean_squared_error)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PART A: RANDOM FOREST MODEL EVALUATION")
print("Student: Madut")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATASET
# ============================================================================
print("\n[1] LOADING DATASET...")
print("-"*50)

try:
    df = pd.read_csv('DataCoSupplyChainDataset.csv', encoding='unicode_escape')
    print(f"Dataset loaded successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()[:10]}... (showing first 10)")
except FileNotFoundError:
    print("Dataset not found. Please download from:")
    print("   https://data.mendeley.com/datasets/8gx2fvg2k6/5")
    exit()

# ============================================================================
# STEP 2: PREPARE DATA FOR CLASSIFICATION (Fraud Detection)
# ============================================================================
print("\n[2] PREPARING DATA FOR FRAUD DETECTION...")
print("-"*50)

# Create binary target for fraud detection
df['FRAUD'] = (df['Order Status'] == 'SUSPECTED_FRAUD').astype(int)
print(f"Created fraud target variable")
print(f"   Fraud cases: {df['FRAUD'].sum()} ({df['FRAUD'].mean()*100:.2f}%)")
print(f"   Non-fraud: {len(df) - df['FRAUD'].sum()} ({(1-df['FRAUD'].mean())*100:.2f}%)")

# Select features for fraud detection (based on the notebook)
fraud_features = [
    'Days for shipping (real)',
    'Days for shipment (scheduled)',
    'Benefit per order',
    'Sales per customer',
    'Order Item Discount',
    'Order Item Quantity',
    'Product Price'
]

# Keep only features that exist in the dataset
available_features = [f for f in fraud_features if f in df.columns]
print(f"\nUsing {len(available_features)} features:")
for f in available_features:
    print(f"   - {f}")

X_fraud = df[available_features].fillna(0)
y_fraud = df['FRAUD']

# Split data
Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    X_fraud, y_fraud, test_size=0.3, random_state=42, stratify=y_fraud
)
print(f"\n Data split:")
print(f"   Training set: {Xf_train.shape}")
print(f"   Test set: {Xf_test.shape}")

# ============================================================================
# STEP 3: TRAIN RANDOM FOREST CLASSIFIER (Fraud Detection)
# ============================================================================
print("\n[3] TRAINING RANDOM FOREST FOR FRAUD DETECTION...")
print("-"*50)

start_time = time.time()

rf_fraud = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_fraud.fit(Xf_train, yf_train)

train_time_fraud = time.time() - start_time
print(f"Training completed in {train_time_fraud:.2f} seconds")

# Predict
yf_pred = rf_fraud.predict(Xf_test)

# Calculate metrics
accuracy_f = accuracy_score(yf_test, yf_pred)
precision_f = precision_score(yf_test, yf_pred)
recall_f = recall_score(yf_test, yf_pred)
f1_f = f1_score(yf_test, yf_pred)

print(f"\n FRAUD DETECTION RESULTS:")
print(f"   Accuracy:  {accuracy_f*100:.2f}%")
print(f"   Precision: {precision_f*100:.2f}%")
print(f"   Recall:    {recall_f*100:.2f}%")
print(f"   F1-Score:  {f1_f*100:.2f}%")

# Confusion Matrix
cm_fraud = confusion_matrix(yf_test, yf_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm_fraud, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Fraud Detection (Random Forest)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('madut_fraud_cm.png', dpi=150, bbox_inches='tight')
print(f"\n Confusion matrix saved as 'madut_fraud_cm.png'")

# Feature Importance
importance_fraud = pd.DataFrame({
    'Feature': available_features,
    'Importance': rf_fraud.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n Feature Importance (Fraud Detection):")
print(importance_fraud.to_string(index=False))

# ============================================================================
# STEP 4: PREPARE DATA FOR LATE DELIVERY PREDICTION
# ============================================================================
print("\n[4] PREPARING DATA FOR LATE DELIVERY PREDICTION...")
print("-"*50)

# Create binary target for late delivery
df['LATE_DELIVERY'] = (df['Delivery Status'] == 'Late delivery').astype(int)
print(f" Created late delivery target variable")
print(f"   Late deliveries: {df['LATE_DELIVERY'].sum()} ({df['LATE_DELIVERY'].mean()*100:.2f}%)")

# Select features for late delivery
late_features = [
    'Days for shipping (real)',
    'Days for shipment (scheduled)',
    'Benefit per order',
    'Sales per customer',
    'Order Item Quantity',
    'Product Price'
]

available_late_features = [f for f in late_features if f in df.columns]
X_late = df[available_late_features].fillna(0)
y_late = df['LATE_DELIVERY']

# Split data
Xl_train, Xl_test, yl_train, yl_test = train_test_split(
    X_late, y_late, test_size=0.3, random_state=42, stratify=y_late
)

# ============================================================================
# STEP 5: TRAIN RANDOM FOREST FOR LATE DELIVERY
# ============================================================================
print("\n[5] TRAINING RANDOM FOREST FOR LATE DELIVERY...")
print("-"*50)

start_time = time.time()

rf_late = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_late.fit(Xl_train, yl_train)

train_time_late = time.time() - start_time
print(f"Training completed in {train_time_late:.2f} seconds")

# Predict
yl_pred = rf_late.predict(Xl_test)

# Calculate metrics
accuracy_l = accuracy_score(yl_test, yl_pred)
precision_l = precision_score(yl_test, yl_pred)
recall_l = recall_score(yl_test, yl_pred)
f1_l = f1_score(yl_test, yl_pred)

print(f"\n LATE DELIVERY RESULTS:")
print(f"   Accuracy:  {accuracy_l*100:.2f}%")
print(f"   Precision: {precision_l*100:.2f}%")
print(f"   Recall:    {recall_l*100:.2f}%")
print(f"   F1-Score:  {f1_l*100:.2f}%")

# Confusion Matrix
cm_late = confusion_matrix(yl_test, yl_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm_late, annot=True, fmt='d', cmap='Oranges')
plt.title('Confusion Matrix - Late Delivery (Random Forest)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('madut_late_cm.png', dpi=150, bbox_inches='tight')
print(f"\n Confusion matrix saved as 'madut_late_cm.png'")

# ============================================================================
# STEP 6: PREPARE DATA FOR REGRESSION (Sales Prediction)
# ============================================================================
print("\n[6] PREPARING DATA FOR SALES PREDICTION...")
print("-"*50)

# Select features for sales prediction
sales_features = [
    'Days for shipping (real)',
    'Days for shipment (scheduled)',
    'Benefit per order',
    'Order Item Discount',
    'Order Item Quantity',
    'Product Price'
]

available_sales_features = [f for f in sales_features if f in df.columns]
X_sales = df[available_sales_features].fillna(0)
y_sales = df['Sales']

# Split data
Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    X_sales, y_sales, test_size=0.3, random_state=42
)

# Scale features for regression
scaler = StandardScaler()
Xs_train_scaled = scaler.fit_transform(Xs_train)
Xs_test_scaled = scaler.transform(Xs_test)

# ============================================================================
# STEP 7: TRAIN RANDOM FOREST REGRESSOR (Sales)
# ============================================================================
print("\n[7] TRAINING RANDOM FOREST FOR SALES PREDICTION...")
print("-"*50)

start_time = time.time()

rf_sales = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_sales.fit(Xs_train_scaled, ys_train)

train_time_sales = time.time() - start_time
print(f" Training completed in {train_time_sales:.2f} seconds")

# Predict
ys_pred = rf_sales.predict(Xs_test_scaled)

# Calculate metrics
mae_sales = mean_absolute_error(ys_test, ys_pred)
rmse_sales = np.sqrt(mean_squared_error(ys_test, ys_pred))

print(f"\n SALES PREDICTION RESULTS:")
print(f"   MAE:  {mae_sales:.4f}")
print(f"   RMSE: {rmse_sales:.4f}")

# ============================================================================
# STEP 8: PREPARE DATA FOR QUANTITY PREDICTION
# ============================================================================
print("\n[8] PREPARING DATA FOR QUANTITY PREDICTION...")
print("-"*50)

# Select features for quantity prediction
qty_features = [
    'Days for shipping (real)',
    'Days for shipment (scheduled)',
    'Benefit per order',
    'Sales per customer',
    'Product Price',
    'Sales'
]

available_qty_features = [f for f in qty_features if f in df.columns]
X_qty = df[available_qty_features].fillna(0)
y_qty = df['Order Item Quantity']

# Split data
Xq_train, Xq_test, yq_train, yq_test = train_test_split(
    X_qty, y_qty, test_size=0.3, random_state=42
)

# Scale features
scaler_q = StandardScaler()
Xq_train_scaled = scaler_q.fit_transform(Xq_train)
Xq_test_scaled = scaler_q.transform(Xq_test)

# ============================================================================
# STEP 9: TRAIN RANDOM FOREST REGRESSOR (Quantity)
# ============================================================================
print("\n[9] TRAINING RANDOM FOREST FOR QUANTITY PREDICTION...")
print("-"*50)

start_time = time.time()

rf_qty = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_qty.fit(Xq_train_scaled, yq_train)

train_time_qty = time.time() - start_time
print(f"Training completed in {train_time_qty:.2f} seconds")

# Predict
yq_pred = rf_qty.predict(Xq_test_scaled)

# Calculate metrics
mae_qty = mean_absolute_error(yq_test, yq_pred)
rmse_qty = np.sqrt(mean_squared_error(yq_test, yq_pred))

print(f"\nQUANTITY PREDICTION RESULTS:")
print(f"   MAE:  {mae_qty:.4f}")
print(f"   RMSE: {rmse_qty:.4f}")

# ============================================================================
# STEP 10: SAVE ALL RESULTS
# ============================================================================
print("\n[10] SAVING RESULTS...")
print("-"*50)

results = {
    'Task': ['Fraud Detection', 'Late Delivery', 'Sales Prediction', 'Quantity Prediction'],
    'Accuracy (%)': [f"{accuracy_f*100:.2f}", f"{accuracy_l*100:.2f}", 'N/A', 'N/A'],
    'Precision (%)': [f"{precision_f*100:.2f}", f"{precision_l*100:.2f}", 'N/A', 'N/A'],
    'Recall (%)': [f"{recall_f*100:.2f}", f"{recall_l*100:.2f}", 'N/A', 'N/A'],
    'F1-Score (%)': [f"{f1_f*100:.2f}", f"{f1_l*100:.2f}", 'N/A', 'N/A'],
    'MAE': ['N/A', 'N/A', f"{mae_sales:.4f}", f"{mae_qty:.4f}"],
    'RMSE': ['N/A', 'N/A', f"{rmse_sales:.4f}", f"{rmse_qty:.4f}"],
    'Training Time (s)': [f"{train_time_fraud:.2f}", f"{train_time_late:.2f}", 
                         f"{train_time_sales:.2f}", f"{train_time_qty:.2f}"]
}

results_df = pd.DataFrame(results)
results_df.to_csv('madut_random_forest_results.csv', index=False)
print("\nResults saved to 'madut_random_forest_results.csv'")
print("\nFINAL RESULTS TABLE:")
print(results_df.to_string(index=False))

print("\n" + "="*70)
print(" PART A COMPLETED SUCCESSFULLY")
print("="*70)