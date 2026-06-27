import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

TRAIN_URL = "data/aps_failure_training_set.csv"
TEST_URL  = "data/aps_failure_test_set.csv"

COST_FN = 500
COST_FP = 10


def run_experiment(progress_callback=None):
    def log(msg):
        if progress_callback:
            progress_callback(msg)

    log("Cargando datasets...")
    train_df = pd.read_csv(TRAIN_URL)
    test_df  = pd.read_csv(TEST_URL)

    log("Preprocesando datos...")
    train_df['class'] = train_df['class'].map({'pos': 1, 'neg': 0})
    test_df['class']  = test_df['class'].map({'pos': 1, 'neg': 0})

    for col in train_df.columns:
        if col != 'class':
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
            test_df[col]  = pd.to_numeric(test_df[col],  errors='coerce')

    cols_to_drop = train_df.columns[train_df.isna().mean() > 0.75]
    train_df.drop(columns=cols_to_drop, inplace=True)
    test_df.drop(columns=cols_to_drop, inplace=True)

    train_df.fillna(train_df.mean(numeric_only=True), inplace=True)
    test_df.fillna(train_df.mean(numeric_only=True), inplace=True)

    X_train = train_df.drop('class', axis=1)
    y_train = train_df['class']
    X_test  = test_df.drop('class', axis=1)
    y_test  = test_df['class']

    log("Estandarizando...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    log("Aplicando SMOTE...")
    smote = SMOTE(random_state=42, sampling_strategy=0.25)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

    log("Entrenando Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train_bal, y_train_bal)

    log("Evaluando modelo...")
    y_prob = rf.predict_proba(X_test_scaled)[:, 1]

    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)

    log("Optimizando threshold...")
    best_threshold = 0.35
    y_pred_opt = (y_prob >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_opt).ravel()
    min_cost = fn * COST_FN + fp * COST_FP

    recall    = recall_score(y_test, y_pred_opt)
    precision = precision_score(y_test, y_pred_opt, zero_division=0)
    f1        = f1_score(y_test, y_pred_opt, zero_division=0)

    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "recall":    round(recall, 3),
        "precision": round(precision, 3),
        "f1":        round(f1, 3),
        "cost":      int(min_cost),
        "threshold": round(best_threshold, 2),
        "pr_precision": precision_vals,
        "pr_recall":    recall_vals,
    }