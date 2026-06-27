import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from model.train import run_experiment

st.set_page_config(
    page_title="APS Failure — AndesCarga S.A.",
    page_icon="",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;600;700&display=swap');

    .stApp {
        background-color: #fff0f6 !important;
        font-family: 'Quicksand', sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #ffc2d4 !important;
        border-right: 2px solid #ff85a1;
    }
    [data-testid="stSidebar"] * {
        color: #6d0026 !important;
        font-family: 'Quicksand', sans-serif !important;
    }
    .stButton > button {
        background-color: #ff4d80 !important;
        color: white !important;
        border: none !important;
        border-radius: 20px !important;
        font-weight: 700 !important;
        font-family: 'Quicksand', sans-serif !important;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        background-color: #cc0044 !important;
    }
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 20px 16px;
        text-align: center;
        border: 2px solid #ffb3c6;
        box-shadow: 0 4px 12px rgba(255,77,128,0.1);
    }
    .metric-label {
        color: #cc0055;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        margin: 4px 0;
    }
    .metric-sub {
        color: #ff85a1;
        font-size: 11px;
        margin-top: 4px;
    }
    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #cc0055;
        margin: 32px 0 16px 0;
        text-align: center;
        letter-spacing: 0.5px;
    }
    .grafico-titulo {
        font-size: 16px;
        font-weight: 700;
        color: #cc0055;
        text-align: center;
        margin: 24px 0 4px 0;
    }
    .grafico-sub {
        font-size: 12px;
        color: #ff85a1;
        text-align: center;
        margin-bottom: 8px;
    }
    .interp-card {
        background: white;
        border-radius: 16px;
        padding: 20px 24px;
        border: 2px solid #ffb3c6;
        margin: 8px 0;
        box-shadow: 0 4px 12px rgba(255,77,128,0.08);
    }
    .interp-titulo {
        font-size: 14px;
        font-weight: 700;
        color: #cc0055;
        margin-bottom: 10px;
    }
    .interp-fila {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid #ffe0ec;
        font-size: 14px;
        color: #6d0026;
    }
    .interp-fila:last-child { border-bottom: none; }
    .interp-val {
        font-weight: 700;
        color: #ff4d80;
    }
    .hipotesis-card {
        background: #ff4d80;
        border-radius: 16px;
        padding: 24px;
        color: white;
        text-align: center;
        margin: 16px 0;
    }
    .hipotesis-titulo {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .hipotesis-texto {
        font-size: 14px;
        opacity: 0.95;
        line-height: 1.6;
    }
    h1, h2, h3 { color: #cc0055 !important; }
    .stProgress > div > div { background-color: #ff4d80 !important; }
    [data-testid="stAlert"] {
        background-color: #ffe0ec !important;
        border: 1px solid #ff85a1 !important;
        color: #6d0026 !important;
        border-radius: 12px !important;
    }
    hr { border-color: #ffb3c6 !important; }
</style>
""", unsafe_allow_html=True)

st.title("Mantenimiento Predictivo — Sistema APS")
st.caption("AndesCarga S.A.  ·  Ingenieria de Software — Inteligencia Artificial I")
st.markdown("---")

with st.sidebar:
    st.markdown("### Configuracion")
    st.markdown("**Modelo:** Random Forest")
    st.markdown("**Arboles:** 200 | **Profundidad:** 10")
    st.markdown("**Semilla:** 42 | **Balanceo:** SMOTE 80/20")
    st.markdown("---")
    st.markdown("**Costos de error**")
    st.markdown("FN (falla no detectada): **500**")
    st.markdown("FP (chequeo innecesario): **10**")
    st.markdown("---")
    run_btn = st.button("Ejecutar experimento", use_container_width=True, type="primary")

if "results" not in st.session_state:
    st.session_state.results = None

if run_btn:
    log_placeholder = st.empty()
    progress = st.progress(0)
    steps = []

    def log(msg):
        steps.append(msg)
        log_placeholder.info(f"{msg}")
        progress.progress(min(len(steps) / 7, 1.0))

    results = run_experiment(log)
    st.session_state.results = results
    log_placeholder.empty()
    progress.empty()
    st.success("Experimento completado")

if st.session_state.results:
    r = st.session_state.results

    st.markdown('<div class="section-title">Resultados del modelo</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "Recall",      f"{r['recall']:.3f}",    "objetivo >= 0.85",      "#ff4d80" if r['recall'] >= 0.85 else "#cc0055"),
        (c2, "Precision",   f"{r['precision']:.3f}", "alertas correctas",     "#cc0055"),
        (c3, "F1-Score",    f"{r['f1']:.3f}",        "balance P y R",         "#cc0055"),
        (c4, "Costo Total", f"{r['cost']:,}",         "FN x500 + FP x10",     "#6d0026"),
        (c5, "Threshold",   f"{r['threshold']}",      "umbral de decision",   "#ff4d80"),
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
    st.markdown("---")

    # matriz de confusion
    st.markdown('<div class="grafico-titulo">Matriz de Confusion</div>', unsafe_allow_html=True)
    st.markdown('<div class="grafico-sub">Compara las predicciones del modelo contra los valores reales</div>', unsafe_allow_html=True)

    col_cm, col_esp = st.columns([2, 1])
    with col_cm:
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#fff0f6")
        ax.set_facecolor("#fff0f6")
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        sns.heatmap(
            cm, annot=False, cmap="RdPu", ax=ax,
            xticklabels=["Pred: No falla", "Pred: Falla"],
            yticklabels=["Real: No falla", "Real: Falla"],
            linewidths=2, linecolor="#fff0f6",
            cbar=False, vmin=0
        )
        for i in range(2):
            for j in range(2):
                valor = cm[i, j]
                color_texto = "white" if valor > cm.max() * 0.4 else "#6d0026"
                ax.text(j + 0.5, i + 0.5, f"{valor:,}",
                        ha="center", va="center",
                        color=color_texto, fontsize=16, fontweight="bold")
        ax.tick_params(colors="#6d0026", labelsize=11)
        ax.set_xlabel("Prediccion del modelo", color="#cc0055", fontsize=12)
        ax.set_ylabel("Valor real", color="#cc0055", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_esp:
        st.markdown(f"""
        <div style="margin-top: 20px;">
            <div class="interp-card">
                <div class="interp-titulo">Verdaderos Negativos (TN)</div>
                <p style="font-size:13px; color:#6d0026; margin:0;">Camiones sanos identificados correctamente.</p>
                <p style="font-size:22px; font-weight:700; color:#ff4d80; margin:8px 0 0 0;">{r['tn']:,}</p>
            </div>
            <div class="interp-card" style="margin-top:8px;">
                <div class="interp-titulo">Verdaderos Positivos (TP)</div>
                <p style="font-size:13px; color:#6d0026; margin:0;">Fallas APS detectadas correctamente.</p>
                <p style="font-size:22px; font-weight:700; color:#ff4d80; margin:8px 0 0 0;">{r['tp']:,}</p>
            </div>
            <div class="interp-card" style="margin-top:8px;">
                <div class="interp-titulo">Falsos Positivos (FP)</div>
                <p style="font-size:13px; color:#6d0026; margin:0;">Alarmas innecesarias — camiones sanos marcados como falla.</p>
                <p style="font-size:22px; font-weight:700; color:#cc0055; margin:8px 0 0 0;">{r['fp']:,}</p>
            </div>
            <div class="interp-card" style="margin-top:8px;">
                <div class="interp-titulo">Falsos Negativos (FN)</div>
                <p style="font-size:13px; color:#6d0026; margin:0;">Fallas reales NO detectadas — el error mas costoso.</p>
                <p style="font-size:22px; font-weight:700; color:#cc0055; margin:8px 0 0 0;">{r['fn']:,}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # curva precision-recall
    st.markdown('<div class="grafico-titulo">Curva Precision-Recall</div>', unsafe_allow_html=True)
    st.markdown('<div class="grafico-sub">Muestra el balance entre detectar fallas y evitar falsas alarmas segun el threshold</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#fff0f6")
    ax.set_facecolor("#ffe0ec")
    ax.plot(r["pr_recall"], r["pr_precision"], color="#ff4d80", linewidth=2.5)
    ax.axvline(r["recall"], color="#cc0055", linestyle="--", linewidth=1.5,
               label=f"Recall = {r['recall']:.3f}")
    ax.axhline(r["precision"], color="#6d0026", linestyle="--", linewidth=1.5,
               label=f"Precision = {r['precision']:.3f}")
    ax.scatter([r["recall"]], [r["precision"]], color="#ff4d80", s=120, zorder=5)
    ax.set_xlabel("Recall (cobertura de fallas reales)", color="#6d0026", fontsize=12)
    ax.set_ylabel("Precision (exactitud de alertas)", color="#6d0026", fontsize=12)
    ax.tick_params(colors="#6d0026")
    ax.spines["bottom"].set_color("#ffb3c6")
    ax.spines["left"].set_color("#ffb3c6")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="white", labelcolor="#6d0026", fontsize=10,
              framealpha=0.9, edgecolor="#ffb3c6")
    ax.grid(True, color="#ffcce0", linewidth=0.6)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # desglose de predicciones
    st.markdown('<div class="grafico-titulo">Desglose de predicciones</div>', unsafe_allow_html=True)
    st.markdown('<div class="grafico-sub">Distribucion de los 16,000 registros del conjunto de prueba</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#fff0f6")
    ax.set_facecolor("#ffe0ec")
    labels = ["TN\nNo fallas correctas", "TP\nFallas detectadas", "FP\nFalsas alarmas", "FN\nFallas perdidas"]
    values = [r["tn"], r["tp"], r["fp"], r["fn"]]
    colores = ["#ffb3c6", "#ff4d80", "#ff85a1", "#cc0055"]
    bars = ax.bar(labels, values, color=colores, width=0.5, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{val:,}", ha="center", va="bottom", color="#6d0026",
                fontsize=12, fontweight="bold")
    ax.tick_params(colors="#6d0026", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#ffb3c6")
    ax.spines["left"].set_color("#ffb3c6")
    ax.set_ylim(0, max(values) * 1.15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # interpretacion
    st.markdown('<div class="section-title">Interpretacion de resultados</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="interp-card">
            <div class="interp-titulo">Deteccion de fallas</div>
            <div class="interp-fila">
                <span>Fallas APS reales en el test set</span>
                <span class="interp-val">{r['tp'] + r['fn']}</span>
            </div>
            <div class="interp-fila">
                <span>Fallas detectadas correctamente (TP)</span>
                <span class="interp-val">{r['tp']}</span>
            </div>
            <div class="interp-fila">
                <span>Fallas NO detectadas (FN)</span>
                <span class="interp-val">{r['fn']}</span>
            </div>
            <div class="interp-fila">
                <span>Tasa de deteccion (Recall)</span>
                <span class="interp-val">{r['recall']*100:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div class="interp-card">
            <div class="interp-titulo">Analisis de costos</div>
            <div class="interp-fila">
                <span>Costo por falla no detectada (FN x 500)</span>
                <span class="interp-val">{r['fn'] * 500:,}</span>
            </div>
            <div class="interp-fila">
                <span>Costo por alarma innecesaria (FP x 10)</span>
                <span class="interp-val">{r['fp'] * 10:,}</span>
            </div>
            <div class="interp-fila">
                <span>Costo total del modelo</span>
                <span class="interp-val">{r['cost']:,}</span>
            </div>
            <div class="interp-fila">
                <span>Umbral de decision usado</span>
                <span class="interp-val">{r['threshold']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    estado_h1 = r['recall'] >= 0.85
    st.markdown(f"""
    <div class="hipotesis-card">
        <div class="hipotesis-titulo">
            {'Hipotesis H1 CUMPLIDA' if estado_h1 else 'Hipotesis H1 NO cumplida'}
        </div>
        <div class="hipotesis-texto">
            El modelo {'detecto' if estado_h1 else 'no detecto'} la mayoria de fallas APS reales
            con Recall = {r['recall']:.3f} {'mayor o igual' if estado_h1 else 'menor'} a 0.85.<br>
            {'H0 se rechaza: el modelo predictivo mejora significativamente la deteccion frente al enfoque reactivo actual.' if estado_h1 else 'H0 no se puede rechazar con los resultados actuales.'}
        </div>
    </div>
    """, unsafe_allow_html=True)

