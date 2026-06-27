import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from model.train import (
    load_and_preprocess,
    train_random_forest,
    evaluate_model,
)

st.set_page_config(
    page_title="APS Failure — AndesCarga S.A.",
    page_icon="🚛",
    layout="wide",
)

st.markdown("""
<style>
    .stApp {
        background-color: #ffe6f2 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #ffb6c1 !important;
        border-right: 2px solid #ff1493;
    }
    [data-testid="stSidebar"] * {
        color: #80003a !important;
    }
    .stButton > button {
        background-color: #ff1493 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
    }
    .stButton > button:hover {
        background-color: #c71585 !important;
        color: white !important;
    }
    .metric-card {
        background: #ff69b4;
        border-radius: 10px;
        padding: 18px 22px;
        text-align: center;
        border: 2px solid #ff1493;
    }
    .metric-label { color: #80003a; font-size: 13px; margin-bottom: 4px; font-weight: 600; }
    .metric-value { color: #ffffff; font-size: 28px; font-weight: 700; }
    .metric-sub   { color: #4d0026; font-size: 11px; margin-top: 2px; }
    .section-title {
        font-size: 18px; font-weight: 600;
        color: #c71585; margin: 24px 0 12px 0;
        border-left: 4px solid #ff1493; padding-left: 10px;
    }
    h1, h2, h3, h4, p, label, span, div {
        color: #80003a;
    }
    [data-testid="stDataFrame"] {
        border: 2px solid #ff1493 !important;
        border-radius: 8px;
    }
    .stProgress > div > div {
        background-color: #ff1493 !important;
    }
    [data-testid="stAlert"] {
        background-color: #ffb6c1 !important;
        border: 1px solid #ff1493 !important;
        color: #80003a !important;
    }
    [data-testid="stExpander"] {
        background-color: #ffd6eb !important;
        border: 1px solid #ff69b4 !important;
        border-radius: 8px;
    }
    hr {
        border-color: #ff69b4 !important;
    }
    .stCheckbox label, .stRadio label {
        color: #80003a !important;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

st.title("Mantenimiento Predictivo — Sistema APS")
st.caption("AndesCarga S.A. · Ingeniería de Software — Inteligencia Artificial I")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuración")
    st.markdown("**Modelo:** Random Forest")
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
    thr = None if usar_threshold_optimo else threshold_manual
    rf = train_random_forest(X_train, y_train, log)
    results = {"Random Forest": evaluate_model(rf, X_test, y_test, thr)}
    st.session_state.results = results
    log_placeholder.empty()
    progress.empty()
    st.success("Modelo ejecutado correctamente")


def render_results(name, r):
    st.markdown(f'<div class="section-title">{name}</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "Recall",      f"{r['recall']:.3f}",    "objetivo ≥ 0.85",              "#ff1493" if r['recall'] >= 0.85 else "#c71585"),
        (c2, "Precision",   f"{r['precision']:.3f}", "",                              "#c71585"),
        (c3, "F1-Score",    f"{r['f1']:.3f}",        "",                              "#c71585"),
        (c4, "Costo Total", f"{r['cost']:,}",         f"Threshold: {r['threshold']}", "#80003a"),
        (c5, "Threshold",   f"{r['threshold']}",      "óptimo para costo",            "#ff1493"),
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
        fig.patch.set_facecolor("#ffe6f2")
        ax.set_facecolor("#ffe6f2")
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="RdPu", ax=ax,
            xticklabels=["Pred: No falla", "Pred: Falla"],
            yticklabels=["Real: No falla", "Real: Falla"],
            linewidths=0.5, linecolor="#ffb6c1",
            annot_kws={"color": "white", "size": 13}
        )
        ax.tick_params(colors="#80003a")
        ax.set_xlabel("Predicción", color="#80003a")
        ax.set_ylabel("Valor Real", color="#80003a")
        ax.set_title("Matriz de Confusión", color="#c71585", pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.markdown("**Curva Precision-Recall**")
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#ffe6f2")
        ax.set_facecolor("#ffd6eb")
        ax.plot(r["pr_recall"], r["pr_precision"], color="#ff1493", linewidth=2)
        ax.axvline(r["recall"], color="#c71585", linestyle="--", linewidth=1.2, label=f"Recall = {r['recall']:.3f}")
        ax.axhline(r["precision"], color="#80003a", linestyle="--", linewidth=1.2, label=f"Precision = {r['precision']:.3f}")
        ax.set_xlabel("Recall", color="#80003a")
        ax.set_ylabel("Precision", color="#80003a")
        ax.set_title("Curva Precision-Recall", color="#c71585")
        ax.tick_params(colors="#80003a")
        ax.spines["bottom"].set_color("#ff69b4")
        ax.spines["left"].set_color("#ff69b4")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(facecolor="#ffb6c1", labelcolor="#80003a", fontsize=9)
        ax.grid(True, color="#ffd6eb", linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("**Desglose de predicciones**")
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#ffe6f2")
    ax.set_facecolor("#ffd6eb")
    labels = ["TP\n(Fallas detectadas)", "TN\n(No fallas correctas)", "FP\n(Falsas alarmas)", "FN\n(Fallas perdidas)"]
    values = [r["tp"], r["tn"], r["fp"], r["fn"]]
    colors = ["#ff1493", "#ff69b4", "#ffb6c1", "#c71585"]
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="#ffe6f2")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(val), ha="center", va="bottom", color="#80003a", fontsize=11)
    ax.tick_params(colors="#80003a")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#ff69b4")
    ax.spines["left"].set_color("#ff69b4")
    ax.set_title("Distribución de predicciones", color="#c71585")
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
- Hipótesis H₁ {'cumplida' if r['recall'] >= 0.85 else '❌ no cumplida'} (Recall {'≥' if r['recall'] >= 0.85 else '<'} 0.85)
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
    for name, r in st.session_state.results.items():
        render_results(name, r)
        st.markdown("---")
else:
    st.info(" Configura el modelo en el panel izquierdo y presiona **Ejecutar modelo**")
    st.markdown("""
    ### ¿Qué hace este sistema?
    Predice fallas en el **Air Pressurized System (APS)** de camiones Scania usando datos de sensores OBD-II.

    | Métrica | Objetivo |
    |---|---|
    | Recall | ≥ 0.85 (detectar la mayoría de fallas) |
    | Costo total | Minimizar FN×500 + FP×10 |

    **Modelo:** Random Forest (200 árboles, profundidad máxima 10, threshold optimizado)
    """)
