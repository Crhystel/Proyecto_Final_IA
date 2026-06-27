import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, recall_score, precision_score,
    f1_score, precision_recall_curve
)
from imblearn.over_sampling import SMOTE

TRAIN_URL = "https://raw.githubusercontent.com/chaitra31595/Machine-Learning---APS-Failure-at-Scania-Trucks-Data-Set/refs/heads/master/Data/aps_failure_training_set_SMALLER.csv"

TEST_URL = "https://raw.githubusercontent.com/chaitra31595/Machine-Learning---APS-Failure-at-Scania-Trucks-Data-Set/refs/heads/master/Data/aps_failure_test_set.csv"

COST_FN = 500
COST_FP = 10


def load_and_preprocess(progress_callback=None):
    def log(msg):
        if progress_callback:
            progress_callback(msg)

    log("Cargando datasets...")
    train_df = pd.read_csv(TRAIN_URL)
    test_df  = pd.read_csv(TEST_URL)

    log("Preprocesando datos...")
    train_df['class'] = train_df['class'].map({'pos': 1, 'neg': 0})
    test_df['class']  = test_df['class'].map({'pos': 1, 'neg': 0})

    # eliminar columnas con >75% missing
    missing_thresh = 0.75
    cols_to_drop = train_df.columns[train_df.isna().mean() > missing_thresh]
    train_df.drop(columns=cols_to_drop, inplace=True)
    test_df.drop(columns=cols_to_drop, inplace=True)

    # imputar con media de entrenamiento
    train_means = train_df.mean()
    train_df.fillna(train_means, inplace=True)
    test_df.fillna(train_means, inplace=True)

    X_train = train_df.drop('class', axis=1)
    y_train = train_df['class']
    X_test  = test_df.drop('class', axis=1)
    y_test  = test_df['class']

    log("Estandarizando...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    log("Aplicando SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

    return X_train_bal, y_train_bal, X_test_scaled, y_test

def train_random_forest(X_train, y_train, progress_callback=None):
    if progress_callback:
        progress_callback("Entrenando Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf




def find_best_threshold(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    best_thr = 0.5
    min_cost = float("inf")
    for thr in np.arange(0.10, 0.90, 0.01):
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        cost = fn * COST_FN + fp * COST_FP
        if cost < min_cost:
            min_cost = cost
            best_thr = round(thr, 2)
    return best_thr, min_cost


def evaluate_model(model, X_test, y_test, threshold=None):
    y_prob = model.predict_proba(X_test)[:, 1]

    if threshold is None:
        threshold, _ = find_best_threshold(model, X_test, y_test)

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cost = fn * COST_FN + fp * COST_FP

    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_prob)

    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "cost": int(cost),
        "threshold": threshold,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "pr_precision": precision_vals,
        "pr_recall": recall_vals,
    }