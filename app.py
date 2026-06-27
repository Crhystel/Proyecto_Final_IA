import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from model.train import (
    load_and_preprocess,
    train_random_forest,
    train_decision_tree,
    evaluate_model,
)

st.set_page_config(
    page_title="APS Failure — AndesCarga S.A.",
    page_icon="🚛",
    layout="wide",
)

st.markdown("""
<style>
    .metric-card {
        background: #1e2530;
        border-radius: 10px;
        padding: 18px 22px;
        text-align: center;
        border: 1px solid #2e3a4e;
    }
    .metric-label { color: #8899aa; font-size: 13px; margin-bottom: 4px; }
    .metric-value { color: #e8edf3; font-size: 28px; font-weight: 700; }
    .metric-sub   { color: #556677; font-size: 11px; margin-top: 2px; }
    .section-title {
        font-size: 18px; font-weight: 600;
        color: #c8d8e8; margin: 24px 0 12px 0;
        border-left: 3px solid #3d8bcd; padding-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚛 Mantenimiento Predictivo — Sistema APS")
st.caption("AndesCarga S.A. · Ingeniería de Software — Inteligencia Artificial I")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuración")
    modelo_elegido = st.radio(
        "Modelo",
        ["Random Forest", "Árbol de Decisión", "Comparar ambos"],
        index=0,
    )
    usar_threshold_optimo = st.checkbox("Optimizar threshold automáticamente", value=True)
    if not usar_threshold_optimo:
        threshold_manual = st.slider("Threshold manual", 0.10, 0.90, 0.50, 0.01)

    st.markdown("---")
    st.markdown("**Costos de error**")
    st.markdown("- FN (falla no detectada): **500**")
    st.markdown("- FP (chequeo innecesario): **10**")
    st.markdown("---")
    run_btn = st.button("▶ Ejecutar modelo", use_container_width=True, type="primary")

if "results" not in st.session_state:
    st.session_state.results = None

if run_btn:
    log_placeholder = st.empty()
    progress = st.progress(0)
    steps = []

    def log(msg):
        steps.append(msg)
        log_placeholder.info(f"⏳ {msg}")
        progress.progress(min(len(steps) / 8, 1.0))

    X_train, y_train, X_test, y_test = load_and_preprocess(log)

    results = {}

    if modelo_elegido in ["Random Forest", "Comparar ambos"]:
        rf = train_random_forest(X_train, y_train, log)
        thr = None if usar_threshold_optimo else threshold_manual
        results["Random Forest"] = evaluate_model(rf, X_test, y_test, thr)

    if modelo_elegido in ["Árbol de Decisión", "Comparar ambos"]:
        dt = train_decision_tree(X_train, y_train, log)
        thr = None if usar_threshold_optimo else threshold_manual
        results["Árbol de Decisión"] = evaluate_model(dt, X_test, y_test, thr)

    st.session_state.results = results
    log_placeholder.empty()
    progress.empty()
    st.success("✅ Modelo ejecutado correctamente")


def render_results(name, r):
    st.markdown(f'<div class="section-title">{name}</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "Recall", f"{r['recall']:.3f}", "objetivo ≥ 0.85", "#4caf80" if r['recall'] >= 0.85 else "#e05c5c"),
        (c2, "Precision", f"{r['precision']:.3f}", "", "#4da8da"),
        (c3, "F1-Score", f"{r['f1']:.3f}", "", "#4da8da"),
        (c4, "Costo Total", f"{r['cost']:,}", f"Threshold: {r['threshold']}", "#f0a050"),
        (c5, "Threshold", f"{r['threshold']}", "óptimo para costo", "#9b7fe0"),
    ]
    for col, label, val, sub, color in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Matriz de Confusión**")
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Pred: No falla", "Pred: Falla"],
            yticklabels=["Real: No falla", "Real: Falla"],
            linewidths=0.5, linecolor="#2e3a4e",
            annot_kws={"color": "white", "size": 13}
        )
        ax.tick_params(colors="#8899aa")
        ax.set_xlabel("Predicción", color="#8899aa")
        ax.set_ylabel("Valor Real", color="#8899aa")
        ax.set_title(f"Matriz de Confusión — {name}", color="#c8d8e8", pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.markdown("**Curva Precision-Recall**")
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#151c27")
        ax.plot(r["pr_recall"], r["pr_precision"], color="#4da8da", linewidth=2)
        ax.axvline(r["recall"], color="#4caf80", linestyle="--", linewidth=1.2, label=f"Recall = {r['recall']:.3f}")
        ax.axhline(r["precision"], color="#f0a050", linestyle="--", linewidth=1.2, label=f"Precision = {r['precision']:.3f}")
        ax.set_xlabel("Recall", color="#8899aa")
        ax.set_ylabel("Precision", color="#8899aa")
        ax.set_title("Curva Precision-Recall", color="#c8d8e8")
        ax.tick_params(colors="#8899aa")
        ax.spines["bottom"].set_color("#2e3a4e")
        ax.spines["left"].set_color("#2e3a4e")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(facecolor="#1e2530", labelcolor="#c8d8e8", fontsize=9)
        ax.grid(True, color="#1e2530", linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("**Desglose de predicciones**")
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#151c27")
    labels = ["TP\n(Fallas detectadas)", "TN\n(No fallas correctas)", "FP\n(Falsas alarmas)", "FN\n(Fallas perdidas)"]
    values = [r["tp"], r["tn"], r["fp"], r["fn"]]
    colors = ["#4caf80", "#4da8da", "#f0a050", "#e05c5c"]
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="#0e1117")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(val), ha="center", va="bottom", color="#c8d8e8", fontsize=11)
    ax.tick_params(colors="#8899aa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#2e3a4e")
    ax.spines["left"].set_color("#2e3a4e")
    ax.set_title("Distribución de predicciones", color="#c8d8e8")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    with st.expander("📋 Interpretación de resultados"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
**Detección de fallas reales:**
- De {r['tp'] + r['fn']} fallas APS reales, el modelo detectó **{r['tp']} ({r['recall']*100:.1f}%)**
- Solo **{r['fn']} fallas** no fueron detectadas (FN)
- Hipótesis H₁ {'✅ cumplida' if r['recall'] >= 0.85 else '❌ no cumplida'} (Recall {'≥' if r['recall'] >= 0.85 else '<'} 0.85)
""")
        with col_b:
            st.markdown(f"""
**Análisis de costos:**
- FN × 500 = **{r['fn'] * 500:,}**
- FP × 10 = **{r['fp'] * 10:,}**
- **Costo total: {r['cost']:,}**
- Threshold usado: **{r['threshold']}**
""")


if st.session_state.results:
    results = st.session_state.results

    if len(results) == 2:
        st.markdown('<div class="section-title">Comparación de modelos</div>', unsafe_allow_html=True)
        comp_data = {
            "Modelo": list(results.keys()),
            "Recall": [r["recall"] for r in results.values()],
            "Precision": [r["precision"] for r in results.values()],
            "F1-Score": [r["f1"] for r in results.values()],
            "Costo Total": [r["cost"] for r in results.values()],
            "Threshold": [r["threshold"] for r in results.values()],
        }
        st.dataframe(pd.DataFrame(comp_data).set_index("Modelo"), use_container_width=True)
        st.markdown("---")

    for name, r in results.items():
        render_results(name, r)
        st.markdown("---")

else:
    st.info("👈 Configura el modelo en el panel izquierdo y presiona **Ejecutar modelo**")
    st.markdown("""
    ### ¿Qué hace este sistema?
    Predice fallas en el **Air Pressurized System (APS)** de camiones Scania usando datos de sensores OBD-II.

    | Métrica | Objetivo |
    |---|---|
    | Recall | ≥ 0.85 (detectar la mayoría de fallas) |
    | Costo total | Minimizar FN×500 + FP×10 |

    **Modelos disponibles:** Random Forest · Árbol de Decisión · Comparación
    """)