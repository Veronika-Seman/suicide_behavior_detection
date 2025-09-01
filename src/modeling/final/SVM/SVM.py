import pandas as pd
import time
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib

# ===== ROOT & PATHS =====
# Resolve project root
ROOT = Path(__file__).resolve().parents[4]

# Data splits (train/validation + test)
SPLITS_DIR   = ROOT / "data" / "processed" / "splits"
TRAINVAL_CSV = SPLITS_DIR / "train_val.csv"
TEST_CSV     = SPLITS_DIR / "test.csv"

# Output directory for this model
FINAL_DIR = ROOT / "modeling" / "final" / "SVM_TFIDF"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

# Output files
MODEL_FILE      = FINAL_DIR / "svm_final_model.joblib"
VECTORIZER_FILE = FINAL_DIR / "svm_vectorizer.joblib"
PREDICTIONS_CSV = FINAL_DIR / "svm_test_predictions.csv"
METRICS_CSV     = FINAL_DIR / "svm_final_metrics.csv"
PARAMS_CSV      = FINAL_DIR / "svm_used_hyperparameters.csv"

# ===== Load data =====
train_df = pd.read_csv(TRAINVAL_CSV)
test_df  = pd.read_csv(TEST_CSV)

X_train = train_df["Text"]
y_train = train_df["Lable"]
X_test  = test_df["Text"]
y_test  = test_df["Lable"]

# ===== TF-IDF Vectorization =====
# - max_features=5000: restrict to 5000 most informative terms
# - ngram_range=(1,2): use both unigrams and bigrams
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ===== Best parameters (from hyperparameter optimization / Optuna) =====
best_params = {
    "C": 1.0728927164663,     # regularization strength
    "kernel": "linear",       # linear kernel SVM
    "probability": True       # enable probability estimates (slower but useful)
}

# ===== Train model =====
start = time.time()
svm_model = SVC(**best_params, random_state=42)
svm_model.fit(X_train_vec, y_train)
runtime = time.time() - start

# ===== Predict & evaluate =====
y_pred = svm_model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")

print("Accuracy:", acc)
print("Precision:", p)
print("Recall:", r)
print("F1:", f1)

# ===== Save model & artifacts =====
# Save the trained SVM and vectorizer to disk for later inference
joblib.dump(svm_model, MODEL_FILE)
joblib.dump(vectorizer, VECTORIZER_FILE)

# Save predictions on test set
test_df["prediction"] = y_pred
test_df.to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8")

# Save metrics
metrics = {
    "accuracy": acc,
    "precision": p,
    "recall": r,
    "f1": f1,
    "runtime_seconds": runtime
}
pd.DataFrame([metrics]).to_csv(METRICS_CSV, index=False)

# Save hyperparameters used
param_log = best_params.copy()
param_log["model"] = "SVM"
pd.DataFrame([param_log]).to_csv(PARAMS_CSV, index=False)

print("Done! Saved to:", FINAL_DIR)
