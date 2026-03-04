# %% [markdown]
# # Exploración Dataset HLS-CMDS
# Tesis: Clasificación de Sonidos Cardiopulmonares con ML

# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import librosa
import librosa.display

plt.style.use('seaborn-v0_8-whitegrid')
DATA_DIR = Path("data")

# %% [markdown]
# ## 1. Carga de datos

# %%
hs = pd.read_csv(DATA_DIR / "HS.csv")
ls = pd.read_csv(DATA_DIR / "LS.csv")
mix = pd.read_csv(DATA_DIR / "Mix.csv")

# %% Ver estructura
hs.head()

# %%
ls.head()

# %%
mix.head()

# %% Columnas disponibles
hs.columns.tolist(), ls.columns.tolist(), mix.columns.tolist()

# %% [markdown]
# ## 2. Análisis exploratorio - Heart Sounds

# %% Resumen estadístico
hs.describe(include='all')

# %% Distribución de clases
# Ajustar nombre de columna según tu CSV
hs_type_col = 'Heart Sound Type'  # MODIFICAR si es diferente
hs[hs_type_col].value_counts()

# %% Visualización
fig, ax = plt.subplots(figsize=(10, 5))
hs[hs_type_col].value_counts().plot(kind='barh', ax=ax)
ax.set_xlabel('Cantidad')
ax.set_title('Distribución - Sonidos Cardíacos')
plt.tight_layout()

# %% Crear etiqueta binaria
hs['is_pathological'] = hs[hs_type_col] != 'Normal'
hs['binary_label'] = hs['is_pathological'].astype(int)
hs[['binary_label', hs_type_col]].value_counts()

# %% [markdown]
# ## 3. Análisis exploratorio - Lung Sounds

# %%
ls_type_col = 'Lung Sound Type'  # MODIFICAR si es diferente
ls[ls_type_col].value_counts()

# %%
fig, ax = plt.subplots(figsize=(10, 5))
ls[ls_type_col].value_counts().plot(kind='barh', ax=ax)
ax.set_xlabel('Cantidad')
ax.set_title('Distribución - Sonidos Pulmonares')
plt.tight_layout()

# %% Etiqueta binaria
ls['is_pathological'] = ls[ls_type_col] != 'Normal'
ls['binary_label'] = ls['is_pathological'].astype(int)

# %% [markdown]
# ## 4. Análisis exploratorio - Mix

# %%
mix.info()

# %%
# Ver todas las columnas del Mix para entender su estructura
mix.head(10)

# %% Distribución de combinaciones (si tiene columnas separadas)
# Ajustar según columnas reales
# mix.groupby(['heart_col', 'lung_col']).size()

# %% [markdown]
# ## 5. Análisis de desbalance

# %%
def imbalance_metrics(series):
    """Calcula métricas de desbalance."""
    counts = series.value_counts()
    total = len(series)
    
    return {
        'n_classes': len(counts),
        'n_samples': total,
        'majority_class': counts.idxmax(),
        'majority_count': counts.max(),
        'minority_class': counts.idxmin(),
        'minority_count': counts.min(),
        'imbalance_ratio': counts.max() / counts.min(),
        'samples_per_class': counts.to_dict()
    }

# %%
imbalance_metrics(hs[hs_type_col])

# %%
imbalance_metrics(ls[ls_type_col])

# %%
# Binario
imbalance_metrics(hs['binary_label'])

# %% [markdown]
# ## 6. Análisis de audio - Cargar ejemplo

# %%
# Ajustar path según estructura
audio_path = DATA_DIR / "HS" / "F_AF_A.wav"  # MODIFICAR
y, sr = librosa.load(audio_path, sr=22050)

# Información básica
len(y), sr, len(y)/sr  # samples, sample_rate, duración en segundos

# %% [markdown]
# ## 7. Visualización de señal de audio

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Waveform
librosa.display.waveshow(y, sr=sr, ax=axes[0,0])
axes[0,0].set_title('Waveform')

# Spectrogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[0,1])
axes[0,1].set_title('Spectrogram')

# Mel spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
S_db = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[1,0])
axes[1,0].set_title('Mel Spectrogram')

# MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[1,1])
axes[1,1].set_title('MFCCs')

plt.tight_layout()

# %% [markdown]
# ## 8. Feature extraction - Un archivo

# %%
def extract_features(y, sr):
    """Extrae features de audio para ML."""
    features = {}
    
    # MFCCs (13 coeficientes) - media y std
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
    
    # Spectral features
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(spec_cent)
    features['spectral_centroid_std'] = np.std(spec_cent)
    
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features['spectral_bandwidth_mean'] = np.mean(spec_bw)
    features['spectral_bandwidth_std'] = np.std(spec_bw)
    
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['spectral_rolloff_mean'] = np.mean(spec_rolloff)
    features['spectral_rolloff_std'] = np.std(spec_rolloff)
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    
    # RMS energy
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    
    return features

# %%
# Probar con un archivo
features = extract_features(y, sr)
pd.Series(features)

# %% [markdown]
# ## 9. Feature extraction - Dataset completo

# %%
def process_dataset(metadata_df, audio_dir, filename_col, label_col):
    """Extrae features de todos los archivos de un dataset."""
    records = []
    
    for _, row in metadata_df.iterrows():
        filepath = audio_dir / row[filename_col]
        if not filepath.exists():
            continue
            
        y, sr = librosa.load(filepath, sr=22050)
        features = extract_features(y, sr)
        features['filename'] = row[filename_col]
        features['label'] = row[label_col]
        records.append(features)
    
    return pd.DataFrame(records)

# %%
# Procesar Heart Sounds (ajustar nombres de columnas)
# hs_features = process_dataset(hs, DATA_DIR / "HS", 'Filename', 'Heart Sound Type')
# hs_features.head()

# %% [markdown]
# ## 10. Análisis de features extraídas

# %%
# Descomentar cuando tengas hs_features
# hs_features.describe()

# %%
# Correlación entre features
# plt.figure(figsize=(12, 10))
# sns.heatmap(hs_features.drop(['filename', 'label'], axis=1).corr(), 
#             cmap='coolwarm', center=0, annot=False)
# plt.title('Correlación entre features')

# %%
# Distribución de features por clase (boxplots)
# feature_cols = [c for c in hs_features.columns if c not in ['filename', 'label']]
# fig, axes = plt.subplots(4, 4, figsize=(16, 12))
# for ax, col in zip(axes.flat, feature_cols[:16]):
#     sns.boxplot(data=hs_features, x='label', y=col, ax=ax)
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
# plt.tight_layout()

# %% [markdown]
# ## 11. Preparación para modelado

# %%
# Separar features y labels
# X = hs_features.drop(['filename', 'label'], axis=1)
# y = hs_features['label']

# Para clasificación binaria
# y_binary = (y != 'Normal').astype(int)

# %%
# Split estratificado
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_binary, test_size=0.2, stratify=y_binary, random_state=42
# )

# %%
# Escalar features
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# ## 12. Resumen para decisiones de modelado

# %%
summary = {
    'heart_sounds': {
        'n_samples': len(hs),
        'n_classes': hs[hs_type_col].nunique(),
        'binary_split': hs['binary_label'].value_counts().to_dict()
    },
    'lung_sounds': {
        'n_samples': len(ls),
        'n_classes': ls[ls_type_col].nunique(),
        'binary_split': ls['binary_label'].value_counts().to_dict()
    },
    'n_features': 32,  # 13 mfcc*2 + 6 spectral
    'recommendations': [
        'Usar clasificación binaria por tamaño del dataset',
        'Validación cruzada estratificada (k=5)',
        'Considerar SMOTE para desbalance',
        'Empezar con Random Forest como baseline'
    ]
}
summary
