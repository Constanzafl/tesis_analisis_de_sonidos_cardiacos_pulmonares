# Clasificación de Sonidos Cardiopulmonares con Machine Learning

**Tesis:** Maestría en Ciencia de Datos - ITBA  
**Autor:** María Constanza Florio  
**Período:** 2024-2026

---

## Objetivo

Comparar modelos de machine learning clásico vs ensambles para clasificar sonidos cardiopulmonares normales vs patológicos utilizando el dataset HLS-CMDS.

## Dataset

**HLS-CMDS:** Heart and Lung Sounds Dataset from Clinical Manikin using Digital Stethoscope

- **Fuente:** https://data.mendeley.com/datasets/8972jxbpmp/3
- **Paper:** Y. Torabi et al., IEEE Data Descriptions, 2025

| Subset | Muestras | Clases |
|--------|----------|--------|
| Heart Sounds | 50 | 10 |
| Lung Sounds | 50 | 6 |
| Mixed | 145 | Combinaciones |

---

## Estructura del Proyecto

```
TESIS_ANALISIS_SONIDOS/
│
├── data/                          # Datos (NO subir a git)
│   ├── HS/                        # Audios Heart Sounds
│   ├── LS/                        # Audios Lung Sounds
│   ├── Mix/                       # Audios Mixed
│   ├── HS.csv                     # Metadata Heart Sounds
│   ├── LS.csv                     # Metadata Lung Sounds
│   └── Mix.csv                    # Metadata Mixed
│
├── notebooks/                     # Análisis y experimentos
│   ├── 01_eda.py                  # Exploratory Data Analysis
│   ├── 02_feature_engineering.py  # Extracción de features
│   ├── 03_baseline_models.py      # Modelos baseline
│   ├── 04_ensemble_models.py      # Modelos ensemble
│   └── 05_evaluation.py           # Evaluación final
│
├── src/                           # Código reutilizable
│   ├── __init__.py
│   ├── features.py                # Funciones de extracción
│   ├── models.py                  # Funciones de modelado
│   └── utils.py                   # Utilidades
│
├── outputs/                       # Resultados (gráficos, CSVs)
│   ├── figures/
│   └── features/
│
├── models/                        # Modelos entrenados (.pkl)
│
├── docs/                          # Documentación
│   └── rsl/                       # Revisión sistemática
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clonar repositorio
```bash
git clone <tu-repo>
cd TESIS_ANALISIS_SONIDOS
```

### 2. Crear entorno virtual
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Descargar datos
Descargar de https://data.mendeley.com/datasets/8972jxbpmp/3 y colocar en `data/`

---

## Notebooks

| # | Notebook | Descripción |
|---|----------|-------------|
| 01 | `01_eda.py` | Análisis exploratorio completo |
| 02 | `02_feature_engineering.py` | Extracción de MFCCs, spectral features |
| 03 | `03_baseline_models.py` | LogReg, SVM, Decision Tree |
| 04 | `04_ensemble_models.py` | Random Forest, XGBoost, Voting |
| 05 | `05_evaluation.py` | Comparación final, métricas |

---

## Modelos a comparar

1. **Baseline:** Regresión Logística
2. **SVM:** kernels lineal, RBF, polynomial
3. **Árboles:** Decision Tree, Random Forest
4. **Boosting:** LightGBM / XGBoost
5. **Ensemble:** Voting Classifier
6. **Opcional:** MLP básico

---

## Métricas de evaluación

- Accuracy
- Precision, Recall, F1-Score
- AUC-ROC
- Matriz de confusión
- Validación cruzada estratificada (k=5)

---

## Referencias

```bibtex
@article{torabi2025hls,
  author={Torabi, Yasaman and Shirani, Shahram and Reilly, James P.},
  title={Descriptor: Heart and Lung Sounds Dataset Recorded from a Clinical Manikin using Digital Stethoscope (HLS-CMDS)},
  journal={IEEE Data Descriptions},
  year={2025},
  doi={10.1109/IEEEDATA.2025.3566012}
}
```