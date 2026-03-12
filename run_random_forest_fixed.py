#!/usr/bin/env python3
"""
Random Forest Model - Individual Assignment
Fixed version with correct target columns
"""

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("RANDOM FOREST MODEL - INDIVIDUAL ASSIGNMENT (FIXED)")
print("="*70)

# Step 1: Load the dataset
print("\n📁 1. Loading dataset...")
dataset = pd.read_csv("DataCoSupplyChainDataset.csv", encoding='unicode_escape')
print(f"   ✅ Dataset loaded successfully!")
print(f"   Dataset shape: {dataset.shape}")
print(f"   Columns: {list(dataset.columns)}")

# Step 2: Identify correct target columns
print("\n🔍 2. Identifying target columns...")

# Based on your notebook screenshot, the targets are:
# fraud_status and late_delivery
# In this dataset, they might be named differently

# Check for fraud-related column
fraud_candidates = ['Late_delivery_risk', 'fraud_status', 'FRAUD', 'is_fraud']
fraud_column = None
for col in fraud_candidates:
    if col in dataset.columns:
        fraud_column = col
        break

# Check for late delivery column  
late_candidates = ['Delivery Status', 'late_delivery', 'LATE_DELIVERY', 'delivery_status']
late_column = None
for col in late_candidates:
    if col in dataset.columns:
        late_column = col
        break

print(f"   Using for fraud target: {fraud_column}")
print(f"   Using for late delivery target: {late_column}")

# Step 3: Prepare features and targets
print("\n🔧 3. Preparing features and targets...")

# Select only numeric columns for features
numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()

# Remove target columns from features if they're numeric
if fraud_column in numeric_cols:
    numeric_cols.remove(fraud_column)
if late_column in numeric_cols and late_column in numeric_cols:
    numeric_cols.remove(late_column)

X = dataset[numeric_cols].fillna(0)
print(f"   Using {len(numeric_cols)} numeric features")

# Get target values
if fraud_column:
    y_fraud = dataset[fraud_column]
    print(f"\n   Fraud status distribution:")
    print(y_fraud.value_counts())
else:
    print("   ⚠️ Fraud column not found, using Late_delivery_risk")
    y_fraud = dataset['Late_delivery_risk']

if late_column:
    y_late = dataset[late_column]
    print(f"\n   Late delivery distribution:")
    print(y_late.value_counts())
else:
    print("   ⚠️ Late delivery column not found, using Delivery Status")
    y_late = dataset['Delivery Status']

# Step 4: Split the data
print("\n✂️ 4. Splitting data into train/test sets...")

# For fraud (binary classification)
xf_train, xf_test, yf_train, yf_test = train_test_split(
    X, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud
)

# For late delivery (might be multi-class)
xl_train, xl_test, yl_train, yl_test = train_test_split(
    X, y_late, test_size=0.2, random_state=42, stratify=y_late
)

print(f"   Fraud - Train size: {len(xf_train)}, Test size: {len(xf_test)}")
print(f"   Late - Train size: {len(xl_train)}, Test size: {len(xl_test)}")

# Step 5: Train Random Forest models
print("\n🌲 5. Training Random Forest models...")

# Model for fraud (using tuned parameters from your screenshot)
rf_fraud = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
print("   Training Random Forest for fraud detection...")
rf_fraud.fit(xf_train, yf_train)

# Model for late delivery
rf_late = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
print("   Training Random Forest for late delivery prediction...")
rf_late.fit(xl_train, yl_train)

# Step 6: Evaluate models
print("\n📊 6. Evaluating models...")
print("-" * 70)

# Fraud predictions
yf_pred = rf_fraud.predict(xf_test)
print("📈 RANDOM FOREST - FRAUD DETECTION RESULTS:")
print(f"   Accuracy: {accuracy_score(yf_test, yf_pred)*100:.4f}%")
print(f"   Recall: {recall_score(yf_test, yf_pred, average='weighted')*100:.4f}%")
print(f"   F1 Score: {f1_score(yf_test, yf_pred, average='weighted')*100:.4f}%")
print("   Confusion Matrix:")
print(confusion_matrix(yf_test, yf_pred))

# Late delivery predictions
yl_pred = rf_late.predict(xl_test)
print("\n📈 RANDOM FOREST - LATE DELIVERY RESULTS:")
print(f"   Accuracy: {accuracy_score(yl_test, yl_pred)*100:.4f}%")
print(f"   Recall: {recall_score(yl_test, yl_pred, average='weighted')*100:.4f}%")
print(f"   F1 Score: {f1_score(yl_test, yl_pred, average='weighted')*100:.4f}%")
print("   Confusion Matrix:")
print(confusion_matrix(yl_test, yl_pred))

# Step 7: Summary for your report
print("\n" + "="*70)
print("📋 SUMMARY FOR YOUR REPORT")
print("="*70)
print("\nRandom Forest Model Performance:")
print("-" * 40)
print(f"Fraud Detection:")
print(f"  • Accuracy: {accuracy_score(yf_test, yf_pred)*100:.2f}%")
print(f"  • Recall: {recall_score(yf_test, yf_pred, average='weighted')*100:.2f}%")
print(f"  • F1-Score: {f1_score(yf_test, yf_pred, average='weighted')*100:.2f}%")
print(f"\nLate Delivery Prediction:")
print(f"  • Accuracy: {accuracy_score(yl_test, yl_pred)*100:.2f}%")
print(f"  • Recall: {recall_score(yl_test, yl_pred, average='weighted')*100:.2f}%")
print(f"  • F1-Score: {f1_score(yl_test, yl_pred, average='weighted')*100:.2f}%")
print("="*70)


