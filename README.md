# 👶 Clinical Decision Support System (CDSS): Automated SAPI Score for Pediatric Deterioration

Prototipo de **Sistema de Ayuda a la Decisión Clínica (SADC)** diseñado para automatizar y estandarizar el cálculo del score **SAPI** (*Sistemas de Alerta Precoz Infantil*), una escala tipo **PEWS** (*Pediatric Early Warning Score*) empleada en la hospitalización pediátrica para anticipar el deterioro hemodinámico y respiratorio grave en lactantes (0 a 11 meses).

Desarrollado en la asignatura *Sistemas de Ayuda a la Decisión* (Grado en Ingeniería Biomédica, ETSIT - Universidad Politécnica de Madrid).

---

## 🏥 Contexto Clínico y Justificación

El score SAPI evalúa el compromiso fisiológico del paciente asignando una puntuación agregada (0-9) a través de cuatro parámetros cardinales:
* **Frecuencia Cardíaca (FC)**
* **Frecuencia Respiratoria (FR)**
* **Saturación de Oxígeno ($SpO_2$)**
* **Apariencia clínica** (esfuerzo respiratorio, reactividad y tono)

### Limitaciones de la praxis manual actual:
* **Variabilidad interobservador:** Especialmente sensible en la apreciación subjetiva de la apariencia y el esfuerzo respiratorio.
* **Carga cognitiva y latencia:** En entornos de urgencias y planta pediátrica de alta presión asistencial, el cálculo manual periódico puede retrasar la activación de equipos de respuesta rápida o ingreso en UCIP.

### Solución propuesta:
Un pipeline de aprendizaje supervisado que ingesta de forma continua las constantes vitales, estratifica objetivamente el nivel de riesgo (**Bajo**, **Medio**, **Alto**) y proyecta la información en un cuadro de mando que simula la interfaz de un **monitor de cabecera clínico**.

---

## 🔬 Metodología e Ingeniería de Datos Clínicos

Ante las restricciones de privacidad y sensibilidad en neonatología, se desarrolló un generador de datos sintéticos estocástico (`datos_sinteticos.py`) parametrizado para **población lactante (0 a 11 meses)** ($N = 500$ registros).

### Evolución de la Lógica de Seguridad Fisiológica:
* **Iteración Inicial (Solo límites superiores):** Logró un 91% de exactitud nominal, pero ignoraba eventos clínicos críticos como la **bradicardia** o la **bradipnea extrema** (desaceleraciones que preceden a la parada cardiorrespiratoria en lactantes).
* **Lógica Definitiva con Rangos de Seguridad Bidireccionales:**
  * **FC:** 2 pts si $FC < 90$ o $FC > 160$ bpm; 1 pto si $140 \le FC \le 160$; 0 pts resto.
  * **FR:** 2 pts si $FR < 20$ o $FR > 60$ rpm; 1 pto si $40 \le FR \le 60$; 0 pts resto.
  * **$SpO_2$:** 0 pts si $\ge 95\%$; 1 pto si $90\% \le SpO_2 < 95\%$; 2 pts si $< 90\%$.
  * **Apariencia:** Vinculada fisiológicamente al grado de hipoxemia/distrés (2-3 pts si $SpO_2 < 90\%$ o $FR \le 20$; 0-1 pts en caso contrario).

---

## 🤖 Comparativa de Modelos de Machine Learning

Se evaluaron tres algoritmos supervisados con una partición de datos 80/20 (400 entrenamiento / 100 prueba):

| Algoritmo | Accuracy Global | F1-Score (Alto) | Recall (Alto) | F1-Score (Medio) | Recall (Bajo) | Fortalezas / Limitaciones Clínicas |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Árbol de Decisión (`max_depth=5`)** ⭐ | **80.00%** | **0.83** | **0.77** | **0.81** | **0.69** | **Modelo Seleccionado:** Alta interpretabilidad (*caja blanca*), reglas diagnósticas reproducibles y sin falsos negativos cruzados (nunca clasifica un *Alto* como *Bajo*). |
| **k-NN ($k=5$, StandardScaler)** | 75.00% | 0.71 | 0.62 | 0.76 | 0.77 | Fronteras de decisión no lineales complejas, pero con un *Recall* inaceptable en casos críticos (omite el 38% de riesgos altos). |
| **Gaussian Naive Bayes** | 65.00% | 0.67 | 0.85 | 0.49 | 1.00 | Excelente sensibilidad en extremos, pero colapsa en la clase *Medio* (*Recall* 0.35) debido a la asunción de independencia entre FC, FR y $SpO_2$. |

### Justificación de la Elección del Árbol de Decisión:
El Árbol de Decisión replica el árbol de razonamiento médico: sitúa en el nodo raíz la **Apariencia $\le 1.5$** (mayor ganancia de información según el índice de Gini), bifurcando hacia la **FR** en pacientes estables y hacia la **bradicardia ($FC \le 87.5$)** en pacientes con apariencia comprometida. Clínicamente garantiza que ningún caso en riesgo crítico es infravalorado como de bajo riesgo.

---

## 📊 Dashboards de Soporte a la Decisión (Power BI)

El proyecto incluye dos cuadros de mando interactivos diseñados en Power BI (`dashboards/Dashboards_SAPI.pbix`):

1. **Dashboard 1: Análisis de Correlación y Validación Clínica:**
   * Gráfico de dispersión bidimensional ($FC$ vs $SpO_2$) coloreado por nivel de riesgo predictivo.
   * Matriz de confusión interactiva y métricas de soporte en tiempo real.
   * Guía asistencial: **Rojo** (Intervención inmediata), **Amarillo** (Observación / Reevaluación a 1h), **Verde** (Seguimiento rutinario).
2. **Dashboard 2: Simulación de Monitor de Cabecera:**
   * Simulación visual de un monitor multiparamétrico de paciente pediátrico.
   * Visualización continua de bioseñales (trazados ECG, pletismografía, capnografía).
   * Cálculo dinámico de campos SAPI desglosados (cardiovascular, respiratorio, saturación y apariencia) con alarma cromática del score total.

---

## 📂 Estructura del Proyecto

```text
.
├── src/
│   ├── datos_sinteticos.py         # Script generador de la cohorte sintética de lactantes
│   ├── arbol_decision.py          # Entrenamiento, visualización y evaluación del Árbol de Decisión
│   ├── k-NN.py                    # Clasificador k-NN con estandarización de características
│   └── naive_bayes.py             # Clasificador Gaussian Naive Bayes y matriz de confusión
├── data/
│   ├── datos_sapi_pediatria.csv   # Dataset sintético de entrenamiento/test (N=500)
│   └── resultados_modelo_sapi.csv # Salida con predicciones de riesgo y validación
├── dashboards/
│   └── Dashboards_SAPI.pbix       # Dashboards interactivos en Power BI (Validación + Monitor)
├── docs/
│   └── Memoria_Trabajo_SAD.pdf    # Memoria técnica completa del proyecto
├── .gitignore
└── README.md
```

---

## 🚀 Requisitos e Instalación

### Entorno Python:
Requiere Python 3.8 o superior y las siguientes librerías:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Ejecución de los Modelos:
```bash
# 1. (Opcional) Regenerar el dataset sintético
python src/datos_sinteticos.py

# 2. Entrenar y evaluar el modelo seleccionado (Árbol de Decisión)
python src/arbol_decision.py

# 3. Comparar con k-NN y Naive Bayes
python src/k-NN.py
python src/naive_bayes.py
```

### Visualización en Power BI:
Abre el archivo `dashboards/Dashboards_SAPI.pbix` directamente en **Power BI Desktop** para interactuar con el monitor clínico y los paneles de validación.



