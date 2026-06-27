# Mantenimiento Predictivo — Sistema APS

Modelo de Machine Learning para predecir fallas en el Air Pressurized System (APS) de camiones Scania operados por AndesCarga S.A.

---

## Descripción

El sistema predice si un vehículo presentará una falla APS antes de que ocurra, usando datos históricos de sensores OBD-II. Se entrena un clasificador binario que minimiza el costo total de errores de detección, priorizando la detección de fallas reales (Recall) sobre la precisión.

**Costo de errores:**

- Falla no detectada (FN): 500
- Chequeo innecesario (FP): 10

---

## Requisitos

- Python 3.9 o superior
- Conexión a internet (los datasets se cargan desde GitHub)

---

## Instalación

```bash
# Clonar o descomprimir el proyecto
cd proyecto-aps

# Crear entorno virtual
python -m venv venv

# Activar (Windows)
venv\Scripts\activate

# Activar (Mac/Linux)
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

---

## Ejecución

```bash
streamlit run app.py
```

La interfaz queda disponible en `http://localhost:8501`.

---

## Estructura

```
proyecto-aps/
├── app.py              # interfaz Streamlit
├── model/
│   ├── __init__.py
│   └── train.py        # carga, preprocesamiento, entrenamiento y evaluación
├── requirements.txt
└── README.md
```

---

## Datos

| Dataset       | Filas  | Columnas | Fuente                               |
| ------------- | ------ | -------- | ------------------------------------ |
| Entrenamiento | 19,999 | 171      | aps_failure_training_set_SMALLER.csv |
| Prueba        | 16,000 | 171      | aps_failure_test_set.csv             |

Repositorio: [chaitra31595/Machine-Learning---APS-Failure-at-Scania-Trucks-Data-Set](https://github.com/chaitra31595/Machine-Learning---APS-Failure-at-Scania-Trucks-Data-Set)

Variable objetivo: columna `class` → `pos` (falla APS) / `neg` (sin falla).

---

## Preprocesamiento

1. Reemplazo de placeholders `na` y `na0` por NaN
2. Conversión de columnas a numérico
3. Eliminación de columnas con más del 75% de valores faltantes
4. Imputación de NaN restantes con la media del conjunto de entrenamiento
5. Estandarización con `StandardScaler`
6. Balanceo de clases con SMOTE → 19,666 ejemplos por clase

---

## Modelos disponibles

**Random Forest**

- 200 árboles, profundidad máxima 10, `random_state=42`
- Threshold optimizado automáticamente en rango 0.10–0.90

**Árbol de Decisión**

- Profundidad máxima 10, `random_state=42`
- Mismo proceso de optimización de threshold

Ambos modelos pueden ejecutarse de forma individual o en modo comparación desde la interfaz.

---

## Resultados esperados (Random Forest, threshold=0.18)

| Métrica     | Valor  |
| ----------- | ------ |
| Recall      | 0.891  |
| Precision   | 0.542  |
| F1-Score    | 0.674  |
| Costo total | 11,350 |
| TP          | 334    |
| TN          | 15,343 |
| FP          | 282    |
| FN          | 41     |

---

## Hipótesis

- **H₁ (alternativa):** el modelo detecta fallas APS con Recall ≥ 0.85 y reduce el costo total frente al enfoque reactivo actual.
- **H₀ (nula):** el modelo no mejora la detección ni reduce el costo total.

Con Recall = 0.891 y costo total = 11,350 (vs. 23,320 sin optimización), H₁ se cumple y H₀ se rechaza.

---

## Autores

Soledad Cabrera · Yuliana Capito · Crhystel Velasco

Ingeniería de Software — Inteligencia Artificial I
