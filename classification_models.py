# Classification part: 9 classifiers x 2 tasks (fraud detection, late delivery)
# Each "# %%" block = one notebook cell. Metrics: accuracy, recall, F1.
# Update DATA_PATH, and swap in the team's shared preprocessing when it is ready.

# %% 1. Imports and settings
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

DATA_PATH = "DataCoSupplyChainDataset.csv"   # <- change to your path
RANDOM_STATE = 42                            # agree on this with the team
TEST_SIZE = 0.2                              # 80/20 split, same as team
SLOW_SUBSAMPLE = 20000                       # SVM and kNN train on a subsample
os.makedirs("results", exist_ok=True)

# %% 2. Load data and build targets
df = pd.read_csv(DATA_PATH, encoding="latin-1")
df["is_fraud"] = (df["Order Status"] == "SUSPECTED_FRAUD").astype(int)
print(df.shape)
print("Fraud rate:", df["is_fraud"].mean().round(4))
print("Late rate :", df["Late_delivery_risk"].mean().round(4))

# %% 3. Preprocessing (basic version; replace with the team's shared one)
# Date features from the order date only (shipping date would leak lateness)
order_dt = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
df["order_month"] = order_dt.dt.month
df["order_weekday"] = order_dt.dt.weekday
df["order_hour"] = order_dt.dt.hour

# IDs, personal info, free text, raw dates
useless = ["Customer Email", "Customer Password", "Customer Fname",
           "Customer Lname", "Customer Street", "Customer Zipcode",
           "Product Image", "Product Description", "Order Zipcode",
           "Order Id", "Order Customer Id", "Order Item Id", "Customer Id",
           "Product Card Id", "Product Category Id", "Category Id",
           "Department Id", "order date (DateOrders)",
           "shipping date (DateOrders)"]

# LEAKAGE: columns that reveal or are derived from the targets
leaky_common = ["Order Status", "Delivery Status", "Late_delivery_risk",
                "Days for shipping (real)", "is_fraud"]

X = df.drop(columns=[c for c in useless + leaky_common if c in df.columns])
cat_cols = X.select_dtypes(include="object").columns
X[cat_cols] = OrdinalEncoder(handle_unknown="use_encoded_value",
                             unknown_value=-1).fit_transform(X[cat_cols].astype(str))
X = X.fillna(0)
print("Features used:", X.shape[1])

# %% 4. Models
def get_models(pos_weight):
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "LDA": LinearDiscriminantAnalysis(),
        "Gaussian NB": GaussianNB(),
        "SVM": SVC(class_weight="balanced"),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                                class_weight="balanced",
                                                random_state=RANDOM_STATE),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, n_jobs=-1,
                                            class_weight="balanced",
                                            random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(eval_metric="logloss", scale_pos_weight=pos_weight,
                                 n_jobs=-1, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced",
                                                random_state=RANDOM_STATE),
    }

# %% 5. Train and evaluate on one task
def run_task(X, y, task_name):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    scaler = StandardScaler().fit(Xtr)            # fit on train only
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    pos_weight = (ytr == 0).sum() / max((ytr == 1).sum(), 1)

    rows, preds = [], {}
    for name, model in get_models(pos_weight).items():
        t0 = time.time()
        if name in ("SVM", "kNN"):                # too slow on the full set
            idx = np.random.RandomState(RANDOM_STATE).choice(
                len(Xtr_s), min(SLOW_SUBSAMPLE, len(Xtr_s)), replace=False)
            model.fit(Xtr_s[idx], ytr.iloc[idx])
        else:
            model.fit(Xtr_s, ytr)
        p = model.predict(Xte_s)
        preds[name] = p
        rows.append({"task": task_name, "model": name,
                     "accuracy": accuracy_score(yte, p),
                     "recall": recall_score(yte, p),
                     "f1": f1_score(yte, p),
                     "train_time_s": round(time.time() - t0, 1)})
        print(f"{task_name:14s} {name:20s} F1={rows[-1]['f1']:.3f}  "
              f"({rows[-1]['train_time_s']}s)")
    return pd.DataFrame(rows), preds, yte

# %% 6. Run both tasks
fraud_res, fraud_preds, fraud_yte = run_task(X, df["is_fraud"], "fraud")
late_res, late_preds, late_yte = run_task(X, df["Late_delivery_risk"], "late_delivery")

results = pd.concat([fraud_res, late_res], ignore_index=True)
results.to_csv("results/classification_results.csv", index=False)
print(results.round(4).to_string(index=False))

# %% 7. Comparison charts (one per task)
for task in ["fraud", "late_delivery"]:
    sub = results[results["task"] == task].set_index("model")[["accuracy", "recall", "f1"]]
    sub.plot(kind="bar", figsize=(10, 5))
    plt.title(f"Classifier comparison: {task}")
    plt.ylabel("Score"); plt.ylim(0, 1.05); plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"results/classification_{task}.png", dpi=150)
    plt.show()

# %% 8. Confusion matrix of the best model (by F1) for each task
for task, preds, yte in [("fraud", fraud_preds, fraud_yte),
                         ("late_delivery", late_preds, late_yte)]:
    best = results[results["task"] == task].sort_values("f1", ascending=False).iloc[0]["model"]
    ConfusionMatrixDisplay(confusion_matrix(yte, preds[best])).plot(cmap="Blues")
    plt.title(f"{task}: {best}")
    plt.savefig(f"results/confusion_{task}.png", dpi=150)
    plt.show()
# %% 9. Feature importance check (sanity check for leakage)
from sklearn.tree import DecisionTreeClassifier
Xtr, Xte, ytr, yte = train_test_split(X, df["Late_delivery_risk"], test_size=TEST_SIZE,
                                      stratify=df["Late_delivery_risk"], random_state=RANDOM_STATE)
dt = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE).fit(Xtr, ytr)
imp = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.head(10))