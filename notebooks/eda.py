# %% [markdown]
# # Exploratory Data Analysis (EDA) - Dataset HLS-CMDS
# 
# **Tesis:** Clasificación de Sonidos Cardiopulmonares con Machine Learning  
# **Autor:** María Constanza Florio  
# **Programa:** Maestría en Ciencia de Datos - ITBA  
# 
# **Dataset:** HLS-CMDS (Heart and Lung Sounds from Clinical Manikin using Digital Stethoscope)  
# **Fuente:** https://data.mendeley.com/datasets/8972jxbpmp/3
# 
# ---
# 
# ## Índice
# 1. Setup y carga de datos
# 2. Análisis estructural de los datos
# 3. Análisis de distribuciones
# 4. Análisis de desbalance de clases
# 5. Análisis de variables categóricas
# 6. Análisis de señales de audio
# 7. Feature Engineering exploratorio
# 8. Correlación entre features
# 9. Conclusiones y decisiones para modelado

# %% [markdown]
# ---
# ## 1. Setup y carga de datos

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import warnings

import librosa
import librosa.display

from IPython.display import display

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
sns.set_style("whitegrid")

# %%
# Rutas
DATA_DIR = Path("data")
AUDIO_HS = DATA_DIR / "HS"
AUDIO_LS = DATA_DIR / "LS"
AUDIO_MIX = DATA_DIR / "Mix"

# Cargar metadatos
hs = pd.read_csv(DATA_DIR / "HS.csv")
ls = pd.read_csv(DATA_DIR / "LS.csv")
mix = pd.read_csv(DATA_DIR / "Mix.csv")

# %% [markdown]
# ---
# ## 2. Análisis estructural de los datos
# 
# ### 2.1 Dimensiones y tipos de datos

# %%
datasets = {'Heart Sounds': hs, 'Lung Sounds': ls, 'Mixed Sounds': mix}

structure = pd.DataFrame({
    name: {'filas': len(df), 'columnas': len(df.columns), 'columnas_list': df.columns.tolist()}
    for name, df in datasets.items()
}).T
structure

# %%
# Tipos de datos por dataset
for name, df in datasets.items():
    print(f"\n{name}:")
    display(df.dtypes.to_frame('dtype'))

# %% [markdown]
# ### 2.2 Primeras filas de cada dataset

# %%
hs.head(10)

# %%
ls.head(10)

# %%
mix.head(10)

# %% [markdown]
# ### 2.3 Valores nulos y duplicados

# %%
quality_check = pd.DataFrame({
    'Heart Sounds': {
        'nulos_total': hs.isnull().sum().sum(),
        'duplicados': hs.duplicated().sum(),
        'ids_unicos': hs['Heart Sound ID'].nunique()
    },
    'Lung Sounds': {
        'nulos_total': ls.isnull().sum().sum(),
        'duplicados': ls.duplicated().sum(),
        'ids_unicos': ls['Lung Sound ID'].nunique()
    },
    'Mixed Sounds': {
        'nulos_total': mix.isnull().sum().sum(),
        'duplicados': mix.duplicated().sum(),
        'ids_unicos': mix['Mixed Sound ID'].nunique()
    }
}).T
quality_check

# %% [markdown]
# ---
# ## 3. Análisis de distribuciones
# 
# ### 3.1 Distribución de clases - Heart Sounds

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Conteo
hs_counts = hs['Heart Sound Type'].value_counts()
colors = sns.color_palette("husl", len(hs_counts))

# Barplot
ax1 = axes[0]
bars = ax1.barh(hs_counts.index, hs_counts.values, color=colors)
ax1.set_xlabel('Cantidad de muestras')
ax1.set_title(f'Heart Sounds - Distribución de clases (n={len(hs)})')
for bar, val in zip(bars, hs_counts.values):
    ax1.text(val + 0.2, bar.get_y() + bar.get_height()/2, f'{val}', va='center')
ax1.set_xlim(0, max(hs_counts) * 1.2)

# Pie chart
ax2 = axes[1]
ax2.pie(hs_counts.values, labels=hs_counts.index, autopct='%1.1f%%', colors=colors)
ax2.set_title('Proporción de clases')

plt.tight_layout()
plt.savefig('outputs/01_distribucion_heart_sounds.png', dpi=150, bbox_inches='tight')

# %% [markdown]
# ### 3.2 Distribución de clases - Lung Sounds

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ls_counts = ls['Lung Sound Type'].value_counts()
colors = sns.color_palette("husl", len(ls_counts))

ax1 = axes[0]
bars = ax1.barh(ls_counts.index, ls_counts.values, color=colors)
ax1.set_xlabel('Cantidad de muestras')
ax1.set_title(f'Lung Sounds - Distribución de clases (n={len(ls)})')
for bar, val in zip(bars, ls_counts.values):
    ax1.text(val + 0.2, bar.get_y() + bar.get_height()/2, f'{val}', va='center')
ax1.set_xlim(0, max(ls_counts) * 1.2)

ax2 = axes[1]
ax2.pie(ls_counts.values, labels=ls_counts.index, autopct='%1.1f%%', colors=colors)
ax2.set_title('Proporción de clases')

plt.tight_layout()
plt.savefig('outputs/02_distribucion_lung_sounds.png', dpi=150, bbox_inches='tight')

# %% [markdown]
# ### 3.3 Distribución de clases - Mixed Sounds

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heart Sound Type en Mix
mix_hs_counts = mix['Heart Sound Type'].value_counts()
ax1 = axes[0]
mix_hs_counts.plot(kind='barh', ax=ax1, color=sns.color_palette("husl", len(mix_hs_counts)))
ax1.set_title(f'Mix - Heart Sound Types (n={len(mix)})')
ax1.set_xlabel('Cantidad')

# Lung Sound Type en Mix
mix_ls_counts = mix['Lung Sound Type'].value_counts()
ax2 = axes[1]
mix_ls_counts.plot(kind='barh', ax=ax2, color=sns.color_palette("husl", len(mix_ls_counts)))
ax2.set_title(f'Mix - Lung Sound Types (n={len(mix)})')
ax2.set_xlabel('Cantidad')

plt.tight_layout()
plt.savefig('outputs/03_distribucion_mixed_sounds.png', dpi=150, bbox_inches='tight')

# %% [markdown]
# ### 3.4 Matriz de combinaciones en Mixed Sounds

# %%
# Crosstab Heart x Lung en Mix
cross = pd.crosstab(mix['Heart Sound Type'], mix['Lung Sound Type'])

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(cross, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
ax.set_title('Combinaciones Heart Sound x Lung Sound en Mix')
plt.tight_layout()
plt.savefig('outputs/04_heatmap_combinaciones_mix.png', dpi=150, bbox_inches='tight')

# %%
cross

# %% [markdown]
# ---
# ## 4. Análisis de desbalance de clases
# 
# ### 4.1 Métricas de desbalance

# %%
def calculate_imbalance_metrics(series, name):
    """Calcula métricas de desbalance para una serie categórica."""
    counts = series.value_counts()
    total = len(series)
    
    # Entropía normalizada (1 = perfectamente balanceado)
    probs = counts / total
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(counts))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return {
        'dataset': name,
        'n_samples': total,
        'n_classes': len(counts),
        'clase_mayoritaria': counts.idxmax(),
        'n_mayoritaria': counts.max(),
        'clase_minoritaria': counts.idxmin(),
        'n_minoritaria': counts.min(),
        'ratio_imbalance': round(counts.max() / counts.min(), 2),
        'balance_score': round(normalized_entropy, 3)
    }

# %%
imbalance_report = pd.DataFrame([
    calculate_imbalance_metrics(hs['Heart Sound Type'], 'Heart Sounds'),
    calculate_imbalance_metrics(ls['Lung Sound Type'], 'Lung Sounds'),
    calculate_imbalance_metrics(mix['Heart Sound Type'], 'Mix - Heart'),
    calculate_imbalance_metrics(mix['Lung Sound Type'], 'Mix - Lung')
])
imbalance_report.set_index('dataset')

# %% [markdown]
# ### 4.2 Crear etiquetas binarias (Normal vs Patológico)

# %%
# Heart Sounds
hs['binary_label'] = (hs['Heart Sound Type'] != 'Normal').astype(int)
hs['binary_class'] = hs['binary_label'].map({0: 'Normal', 1: 'Patológico'})

# Lung Sounds  
ls['binary_label'] = (ls['Lung Sound Type'] != 'Normal').astype(int)
ls['binary_class'] = ls['binary_label'].map({0: 'Normal', 1: 'Patológico'})

# Mixed - dos etiquetas
mix['heart_binary'] = (mix['Heart Sound Type'] != 'Normal').astype(int)
mix['lung_binary'] = (mix['Lung Sound Type'] != 'Normal').astype(int)

# %%
# Balance binario
binary_balance = pd.DataFrame({
    'Heart Sounds': hs['binary_class'].value_counts(),
    'Lung Sounds': ls['binary_class'].value_counts()
})
binary_balance

# %%
# Visualizar balance binario
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, (name, df, col) in zip(axes, [('Heart Sounds', hs, 'binary_class'), 
                                        ('Lung Sounds', ls, 'binary_class')]):
    counts = df[col].value_counts()
    colors = ['#2ecc71', '#e74c3c']  # Verde=Normal, Rojo=Patológico
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title(f'{name} - Balance Binario')
    ax.set_ylabel('Cantidad')
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.5, f'{v} ({v/len(df)*100:.1f}%)', ha='center')

plt.tight_layout()
plt.savefig('outputs/05_balance_binario.png', dpi=150, bbox_inches='tight')

# %% [markdown]
# ### 4.3 Resumen de desbalance

# %%
imbalance_summary = pd.DataFrame({
    'Clasificación': ['Multiclase HS', 'Multiclase LS', 'Binaria HS', 'Binaria LS'],
    'Clases': [10, 6, 2, 2],
    'Ratio': [4.5, 2.4, 4.56, 3.17],
    'Recomendación': [
        'Agrupar clases o usar SMOTE',
        'Manejable con class_weight',
        'Usar class_weight o SMOTE',
        'Usar class_weight o SMOTE'
    ]
})
imbalance_summary

# %% [markdown]
# ---
# ## 5. Análisis de variables categóricas
# 
# ### 5.1 Distribución por género

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, df) in zip(axes, datasets.items()):
    counts = df['Gender'].value_counts()
    ax.pie(counts.values, labels=['Masculino' if x=='M' else 'Femenino' for x in counts.index],
           autopct='%1.1f%%', colors=['#3498db', '#e91e63'])
    ax.set_title(f'{name}\n(n={len(df)})')

plt.suptitle('Distribución por Género', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('outputs/06_distribucion_genero.png', dpi=150, bbox_inches='tight')

# %% [markdown]
# ### 5.2 Distribución por localización anatómica

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heart Sounds - Localizaciones
hs_loc = hs['Location'].value_counts()
ax1 = axes[0]
hs_loc.plot(kind='bar', ax=ax1, color=sns.color_palette("viridis", len(hs_loc)))
ax1.set_title('Heart Sounds - Localizaciones')
ax1.set_xlabel('Localización')
ax1.set_ylabel('Cantidad')
ax1.tick_params(axis='x', rotation=45)

# Lung Sounds - Localizaciones
ls_loc = ls['Location'].value_counts()
ax2 = axes[1]
ls_loc.plot(kind='bar', ax=ax2, color=sns.color_palette("viridis", len(ls_loc)))
ax2.set_title('Lung Sounds - Localizaciones')
ax2.set_xlabel('Localización')
ax2.set_ylabel('Cantidad')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('outputs/07_distribucion_localizaciones.png', dpi=150, bbox_inches='tight')

# %% [markdown]
# ### 5.3 Relación entre tipo de sonido y localización

# %%
# Heatmap: Tipo de sonido vs Localización para Heart Sounds
cross_hs_loc = pd.crosstab(hs['Heart Sound Type'], hs['Location'])

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cross_hs_loc, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title('Heart Sound Type vs Localización')
plt.tight_layout()
plt.savefig('outputs/08_heatmap_hs_location.png', dpi=150, bbox_inches='tight')

# %%
# Heatmap: Tipo de sonido vs Localización para Lung Sounds
cross_ls_loc = pd.crosstab(ls['Lung Sound Type'], ls['Location'])

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(cross_ls_loc, annot=True, fmt='d', cmap='Greens', ax=ax)
ax.set_title('Lung Sound Type vs Localización')
plt.tight_layout()
plt.savefig('outputs/09_heatmap_ls_location.png', dpi=150, bbox_inches='tight')

# %% [markdown]
# ---
# ## 6. Análisis de señales de audio
# 
# ### 6.1 Cargar y explorar un audio de ejemplo

# %%
# Seleccionar un archivo de ejemplo
sample_file = list(AUDIO_HS.glob("*.wav"))[0]
y, sr = librosa.load(sample_file, sr=22050)

audio_info = {
    'archivo': sample_file.name,
    'sample_rate': sr,
    'duracion_segundos': len(y) / sr,
    'n_samples': len(y),
    'amplitud_max': np.max(np.abs(y)),
    'amplitud_media': np.mean(np.abs(y)),
    'rms': np.sqrt(np.mean(y**2))
}
pd.Series(audio_info)

# %% [markdown]
# ### 6.2 Visualización completa de un audio

# %%
def plot_audio_analysis(filepath, title=None):
    """Genera análisis visual completo de un archivo de audio."""
    y, sr = librosa.load(filepath, sr=22050)
    if title is None:
        title = Path(filepath).stem
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Waveform
    ax1 = axes[0, 0]
    librosa.display.waveshow(y, sr=sr, ax=ax1, color='steelblue')
    ax1.set_title('Forma de Onda')
    ax1.set_xlabel('Tiempo (s)')
    ax1.set_ylabel('Amplitud')
    
    # Spectrogram
    ax2 = axes[0, 1]
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img2 = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax2, cmap='magma')
    ax2.set_title('Espectrograma (STFT)')
    ax2.set_ylim(0, 4000)  # Limitar a frecuencias relevantes
    fig.colorbar(img2, ax=ax2, format='%+2.0f dB')
    
    # Mel Spectrogram
    ax3 = axes[1, 0]
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=2048)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img3 = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax3, cmap='magma', fmax=2048)
    ax3.set_title('Mel Espectrograma')
    fig.colorbar(img3, ax=ax3, format='%+2.0f dB')
    
    # MFCCs
    ax4 = axes[1, 1]
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    img4 = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax4, cmap='coolwarm')
    ax4.set_title('MFCCs (13 coeficientes)')
    ax4.set_ylabel('Coeficiente')
    fig.colorbar(img4, ax=ax4)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

# %%
# Analizar ejemplo de Heart Sound
hs_example = list(AUDIO_HS.glob("*.wav"))[0]
fig = plot_audio_analysis(hs_example, f"Heart Sound: {hs_example.stem}")
plt.savefig('outputs/10_analisis_audio_hs_ejemplo.png', dpi=150, bbox_inches='tight')

# %% [markdown]
# ### 6.3 Comparación Normal vs Patológico

# %%
# Buscar un archivo Normal y uno Patológico
normal_id = hs[hs['Heart Sound Type'] == 'Normal']['Heart Sound ID'].iloc[0]
pathological_id = hs[hs['Heart Sound Type'] != 'Normal']['Heart Sound ID'].iloc[0]
pathological_type = hs[hs['Heart Sound ID'] == pathological_id]['Heart Sound Type'].iloc[0]

normal_file = AUDIO_HS / f"{normal_id}.wav"
pathological_file = AUDIO_HS / f"{pathological_id}.wav"

# %%
fig, axes = plt.subplots(2, 3, figsize=(16, 8))

for row, (filepath, label) in enumerate([(normal_file, 'Normal'), 
                                          (pathological_file, f'Patológico ({pathological_type})')]):
    y, sr = librosa.load(filepath, sr=22050)
    
    # Waveform
    librosa.display.waveshow(y, sr=sr, ax=axes[row, 0])
    axes[row, 0].set_title(f'{label} - Waveform')
    axes[row, 0].set_ylim(-0.05, 0.05)
    
    # Mel Spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=2048)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=axes[row, 1], cmap='magma', fmax=2048)
    axes[row, 1].set_title(f'{label} - Mel Spectrogram')
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[row, 2], cmap='coolwarm')
    axes[row, 2].set_title(f'{label} - MFCCs')

plt.tight_layout()
plt.savefig('outputs/11_comparacion_normal_patologico.png', dpi=150, bbox_inches='tight')

# %% [markdown]
# ### 6.4 Estadísticas de duración de audios

# %%
def get_audio_duration(filepath):
    """Obtiene duración de un archivo de audio."""
    try:
        y, sr = librosa.load(filepath, sr=None)
        return len(y) / sr
    except:
        return None

# %%
# Calcular duraciones para cada dataset
hs_durations = [get_audio_duration(AUDIO_HS / f"{id}.wav") for id in hs['Heart Sound ID']]
ls_durations = [get_audio_duration(AUDIO_LS / f"{id}.wav") for id in ls['Lung Sound ID']]

duration_stats = pd.DataFrame({
    'Heart Sounds': pd.Series(hs_durations).describe(),
    'Lung Sounds': pd.Series(ls_durations).describe()
})
duration_stats

# %% [markdown]
# ---
# ## 7. Feature Engineering exploratorio
# 
# ### 7.1 Función de extracción de features

# %%
def extract_features(y, sr):
    """
    Extrae features de audio para clasificación.
    
    Features extraídas:
    - MFCCs (13) + estadísticas (mean, std, max, min)
    - Delta MFCCs
    - Spectral: centroid, bandwidth, rolloff, contrast, flatness
    - Zero crossing rate
    - RMS energy
    - Tempo
    
    Total: ~100 features
    """
    features = {}
    
    # === MFCCs ===
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
        features[f'mfcc_{i+1}_max'] = np.max(mfccs[i])
        features[f'mfcc_{i+1}_min'] = np.min(mfccs[i])
    
    # === Delta MFCCs ===
    mfcc_delta = librosa.feature.delta(mfccs)
    for i in range(13):
        features[f'mfcc_delta_{i+1}_mean'] = np.mean(mfcc_delta[i])
        features[f'mfcc_delta_{i+1}_std'] = np.std(mfcc_delta[i])
    
    # === Spectral Features ===
    # Centroid
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(spec_cent)
    features['spectral_centroid_std'] = np.std(spec_cent)
    
    # Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features['spectral_bandwidth_mean'] = np.mean(spec_bw)
    features['spectral_bandwidth_std'] = np.std(spec_bw)
    
    # Rolloff
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['spectral_rolloff_mean'] = np.mean(spec_rolloff)
    features['spectral_rolloff_std'] = np.std(spec_rolloff)
    
    # Contrast
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(spec_contrast.shape[0]):
        features[f'spectral_contrast_{i+1}_mean'] = np.mean(spec_contrast[i])
        features[f'spectral_contrast_{i+1}_std'] = np.std(spec_contrast[i])
    
    # Flatness
    spec_flat = librosa.feature.spectral_flatness(y=y)[0]
    features['spectral_flatness_mean'] = np.mean(spec_flat)
    features['spectral_flatness_std'] = np.std(spec_flat)
    
    # === Zero Crossing Rate ===
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    
    # === RMS Energy ===
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    features['rms_max'] = np.max(rms)
    
    # === Tempo ===
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    
    return features

# %%
# Test con un archivo
y_test, sr_test = librosa.load(sample_file, sr=22050)
sample_features = extract_features(y_test, sr_test)
f"Features extraídas: {len(sample_features)}"

# %%
# Ver algunas features
pd.Series(sample_features).head(20)

# %% [markdown]
# ### 7.2 Extraer features de todo el dataset

# %%
from tqdm import tqdm

def process_all_files(metadata_df, audio_dir, id_col, label_col, binary_col=None):
    """Extrae features de todos los archivos de un dataset."""
    records = []
    
    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
        filepath = audio_dir / f"{row[id_col]}.wav"
        
        if not filepath.exists():
            continue
        
        try:
            y, sr = librosa.load(filepath, sr=22050)
            features = extract_features(y, sr)
            features['file_id'] = row[id_col]
            features['label'] = row[label_col]
            if binary_col:
                features['label_binary'] = row[binary_col]
            records.append(features)
        except Exception as e:
            print(f"Error en {filepath}: {e}")
    
    return pd.DataFrame(records)

# %%
# Extraer features - Heart Sounds
hs_features = process_all_files(hs, AUDIO_HS, 'Heart Sound ID', 'Heart Sound Type', 'binary_label')
hs_features.shape

# %%
# Extraer features - Lung Sounds
ls_features = process_all_files(ls, AUDIO_LS, 'Lung Sound ID', 'Lung Sound Type', 'binary_label')
ls_features.shape

# %%
# Guardar features extraídas
hs_features.to_csv('outputs/hs_features.csv', index=False)
ls_features.to_csv('outputs/ls_features.csv', index=False)

# %% [markdown]
# ### 7.3 Estadísticas descriptivas de features

# %%
feature_cols = [c for c in hs_features.columns if c not in ['file_id', 'label', 'label_binary']]
hs_features[feature_cols].describe().T

# %% [markdown]
# ---
# ## 8. Correlación entre features
# 
# ### 8.1 Matriz de correlación

# %%
# Seleccionar subset de features para visualización
mfcc_cols = [c for c in feature_cols if 'mfcc' in c and 'delta' not in c and 'mean' in c]
spectral_cols = [c for c in feature_cols if 'spectral' in c and 'mean' in c]
other_cols = ['zcr_mean', 'rms_mean', 'tempo']

selected_features = mfcc_cols + spectral_cols + other_cols

# %%
fig, ax = plt.subplots(figsize=(14, 12))
corr_matrix = hs_features[selected_features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0, ax=ax)
ax.set_title('Matriz de Correlación - Features Seleccionadas')
plt.tight_layout()
plt.savefig('outputs/12_correlacion_features.png', dpi=150, bbox_inches='tight')

# %% [markdown]
# ### 8.2 Features más correlacionadas con la etiqueta binaria

# %%
# Correlación con label binario
correlations = hs_features[feature_cols + ['label_binary']].corr()['label_binary'].drop('label_binary')
correlations_sorted = correlations.abs().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 8))
correlations_sorted.head(20).plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Top 20 Features más correlacionadas con la etiqueta binaria')
ax.set_xlabel('|Correlación|')
plt.tight_layout()
plt.savefig('outputs/13_top_features_correlacion.png', dpi=150, bbox_inches='tight')

# %%
# Ver correlaciones
correlations_sorted.head(20)

# %% [markdown]
# ### 8.3 Distribución de features por clase

# %%
# Boxplots de top features por clase binaria
top_features = correlations_sorted.head(9).index.tolist()

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for ax, feature in zip(axes.flat, top_features):
    sns.boxplot(data=hs_features, x='label_binary', y=feature, ax=ax, palette=['#2ecc71', '#e74c3c'])
    ax.set_xticklabels(['Normal', 'Patológico'])
    ax.set_title(feature)

plt.tight_layout()
plt.savefig('outputs/14_boxplots_top_features.png', dpi=150, bbox_inches='tight')

# %% [markdown]
# ---
# ## 9. Conclusiones y decisiones para modelado

# %%
conclusions = {
    'Dataset': {
        'Heart Sounds': f'{len(hs)} muestras, {hs["Heart Sound Type"].nunique()} clases',
        'Lung Sounds': f'{len(ls)} muestras, {ls["Lung Sound Type"].nunique()} clases',
        'Mixed': f'{len(mix)} muestras (combinaciones)'
    },
    'Desbalance': {
        'HS Multiclase': 'Ratio 4.5:1 - Requiere balanceo',
        'LS Multiclase': 'Ratio 2.4:1 - Manejable',
        'HS Binario': 'Normal:9 vs Patológico:41',
        'LS Binario': 'Normal:12 vs Patológico:38'
    },
    'Features': {
        'Total extraídas': len(feature_cols),
        'Más discriminativas': correlations_sorted.head(5).index.tolist()
    },
    'Recomendaciones': [
        '1. Comenzar con clasificación binaria (Normal vs Patológico)',
        '2. Usar validación cruzada estratificada (k=5)',
        '3. Aplicar SMOTE o class_weight para desbalance',
        '4. Probar selección de features con importancia RF',
        '5. Normalizar features antes de SVM y LogReg'
    ]
}

# %%
# Mostrar conclusiones
for section, content in conclusions.items():
    print(f"\n{'='*50}")
    print(f" {section}")
    print('='*50)
    if isinstance(content, dict):
        for k, v in content.items():
            print(f"  {k}: {v}")
    elif isinstance(content, list):
        for item in content:
            print(f"  {item}")

# %% [markdown]
# ---
# ## Guardar datasets procesados para modelado

# %%
# Guardar con etiquetas
hs.to_csv('outputs/hs_metadata_processed.csv', index=False)
ls.to_csv('outputs/ls_metadata_processed.csv', index=False)
mix.to_csv('outputs/mix_metadata_processed.csv', index=False)

# %%
# Resumen final
summary = pd.DataFrame({
    'Archivo': ['hs_features.csv', 'ls_features.csv', 'hs_metadata_processed.csv'],
    'Descripción': ['Features extraídas HS', 'Features extraídas LS', 'Metadata con etiquetas binarias'],
    'Filas': [len(hs_features), len(ls_features), len(hs)],
    'Uso': ['Input para modelos ML', 'Input para modelos ML', 'Referencia de etiquetas']
})
summary