# tesis_analisis_de_sonidos_cardiacos_pulmonares# Exploración del Dataset HLS-CMDS
## Tesis: Clasificación de Sonidos Cardiopulmonares con Machine Learning

---

## 🚀 Quick Start

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Descargar el dataset

1. Ir a: https://data.mendeley.com/datasets/8972jxbpmp/3
2. Descargar los 3 archivos ZIP:
   - `HS.zip` (Heart Sounds)
   - `LS.zip` (Lung Sounds)
   - `Mix.zip` (Mixed Sounds)
3. Descargar los 3 archivos CSV:
   - `HS.csv`
   - `LS.csv`
   - `Mix.csv`

### 3. Organizar archivos

```
proyecto/
├── exploracion_hls_cmds.py
├── requirements.txt
├── README.md
└── data/
    ├── HS.csv
    ├── LS.csv
    ├── Mix.csv
    ├── HS/
    │   └── (archivos .wav descomprimidos de HS.zip)
    ├── LS/
    │   └── (archivos .wav descomprimidos de LS.zip)
    └── Mix/
        └── (archivos .wav descomprimidos de Mix.zip)
```

### 4. Ejecutar exploración

```bash
python exploracion_hls_cmds.py
```

O en Jupyter Notebook:
```bash
jupyter notebook
# Crear nuevo notebook y copiar/pegar secciones del script
```

---

## 📊 Información del Dataset HLS-CMDS

| Categoría | Cantidad | Descripción |
|-----------|----------|-------------|
| Heart Sounds (HS) | 50 | Sonidos cardíacos puros |
| Lung Sounds (LS) | 50 | Sonidos pulmonares puros |
| Mixed (Mix) | 145 | Sonidos cardiopulmonares mixtos |
| **Total** | **535** | (incluye fuentes separadas de Mix) |

### Tipos de Sonidos Cardíacos (10 clases)
- Normal Heart
- Late Diastolic Murmur
- Mid Systolic Murmur
- Late Systolic Murmur
- Early Systolic Murmur
- Atrial Fibrillation
- Tachycardia
- Atrioventricular Block
- Third Heart Sound (S3)
- Fourth Heart Sound (S4)

### Tipos de Sonidos Pulmonares (6 clases)
- Normal Lung
- Wheezing
- Fine Crackles
- Coarse Crackles
- Rhonchi
- Pleural Rub

### Localizaciones Anatómicas
**Cardíacas:**
- Right Upper Sternal Border
- Left Upper Sternal Border
- Lower Left Sternal Border
- Right Costal Margin
- Left Costal Margin
- Apex

**Pulmonares:**
- Right/Left Upper Anterior
- Right/Left Mid Anterior
- Right/Left Lower Anterior

---

## 🎯 Opciones de Clasificación para la Tesis

### Opción A: Clasificación Binaria 
- **Clases:** Normal vs Patológico
- **Ventajas:** Simple, buen baseline, menos desbalance
- **Métricas:** Accuracy, Precision, Recall, F1, AUC-ROC

### Opción B: Multiclase - Sonidos Cardíacos
- **Clases:** 10 tipos diferentes
- **Ventajas:** Mayor granularidad diagnóstica
- **Desafíos:** Desbalance de clases, más complejidad

### Opción C: Multiclase - Sonidos Pulmonares
- **Clases:** 6 tipos diferentes
- **Ventajas:** Menor cantidad de clases que cardíacos
- **Desafíos:** Menos muestras totales

---

## 📁 Features a Extraer

| Feature | Descripción | Uso típico |
|---------|-------------|------------|
| **MFCCs** | Coeficientes cepstrales | Principal para clasificación de audio |
| **Mel Spectrogram** | Representación tiempo-frecuencia | CNN, visualización |
| **Spectral Centroid** | "Centro de masa" del espectro | Brillo del sonido |
| **Spectral Rolloff** | Frecuencia bajo la cual está el 85% de energía | Caracterización espectral |
| **Zero Crossing Rate** | Tasa de cruces por cero | Ruido vs tonal |
| **RMS Energy** | Energía de la señal | Volumen/intensidad |
| **Chroma Features** | Contenido tonal | Menos usado en biomédico |

---

## 🔧 Modelos a Comparar (según ISL)

1. **Baseline:** Regresión Logística
2. **SVM:** kernels lineal, RBF, polynomial
3. **Árboles:** Decision Tree, Random Forest
4. **Boosting:** LightGBM o XGBoost
5. **Ensemble:** Voting Classifier
6. **Opcional:** MLP básico (scikit-learn)

---

## 📚 Referencias Útiles

- Dataset: https://data.mendeley.com/datasets/8972jxbpmp/3
- Paper del dataset: https://doi.org/10.1109/IEEEDATA.2025.3566012
- librosa documentation: https://librosa.org/doc/
- scikit-learn: https://scikit-learn.org/
- ISL Book: https://www.statlearning.com/
