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

    # 1. Cargar datasets
    log("Cargando datasets...")
    train_df = pd.read_csv(TRAIN_URL)
    test_df  = pd.read_csv(TEST_URL)

    # 2. Convertir target a binario
    log("Preprocesando datos...")
    train_df['class'] = train_df['class'].map({'pos': 1, 'neg': 0})
    test_df['class']  = test_df['class'].map({'pos': 1, 'neg': 0})

    # 3.1 Convertir a numérico (compatibilidad con pandas moderno)
    for col in train_df.columns:
        if col != 'class':
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
            test_df[col]  = pd.to_numeric(test_df[col],  errors='coerce')

    # 3.2 Remover columnas con >75% missing
    cols_to_drop = train_df.columns[train_df.isna().mean() > 0.75]
    train_df.drop(columns=cols_to_drop, inplace=True)
    test_df.drop(columns=cols_to_drop, inplace=True)

    # 3.3 Imputación con media de entrenamiento
    train_df.fillna(train_df.mean(numeric_only=True), inplace=True)
    test_df.fillna(train_df.mean(numeric_only=True), inplace=True)

    # 3.3 Separar features y target
    X_train = train_df.drop('class', axis=1)
    y_train = train_df['class']
    X_test  = test_df.drop('class', axis=1)
    y_test  = test_df['class']

    # 3.4 Estandarización
    log("Estandarizando...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 3.5 Balanceo con SMOTE
    log("Aplicando SMOTE...")
    smote = SMOTE(random_state=42, sampling_strategy=0.25)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

    # 4. Entrenamiento Random Forest
    log("Entrenando Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        random_state=42
    )
    rf.fit(X_train_bal, y_train_bal)

    # 5. Predicción con threshold por defecto (0.5)
    log("Evaluando modelo...")
    y_pred = rf.predict(X_test_scaled)
    y_prob = rf.predict_proba(X_test_scaled)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    cost_total = fn * COST_FN + fp * COST_FP
    recall    = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)

    # 5.4 Curva Precision-Recall
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)

    # 6. Ajuste de threshold para minimizar costo
    log("Optimizando threshold...")
    # 6. Threshold fijo en 0.18 según experimento original
    best_threshold = 0.35
    y_pred_opt = (y_prob >= best_threshold).astype(int)
    tn2, fp2, fn2, tp2 = confusion_matrix(y_test, y_pred_opt).ravel()
    min_cost = fn2 * COST_FN + fp2 * COST_FP

    # Recalcular métricas con el threshold óptimo
    y_pred_opt = (y_prob >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_opt).ravel()
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
