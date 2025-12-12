# -*- coding: utf-8 -*-
"""
================================================================================
EXPLORACIÓN INICIAL DEL DATASET HLS-CMDS
Heart and Lung Sounds Dataset from Clinical Manikin using Digital Stethoscope
================================================================================

Autor: María Constanza Florio
Tesis: Maestría en Ciencia de Datos - ITBA
Fecha: Diciembre 2025

INSTRUCCIONES:
1. Descargar el dataset de: https://data.mendeley.com/datasets/8972jxbpmp/3
2. Descomprimir los archivos: HS.zip, LS.zip, Mix.zip
3. Colocar los CSVs (HS.csv, LS.csv, Mix.csv) en la carpeta 'data/'
4. Colocar los audios en subcarpetas: data/HS/, data/LS/, data/Mix/
5. Ejecutar este script o convertirlo a Jupyter notebook

ESTRUCTURA DE CARPETAS ESPERADA:
    proyecto/
    ├── exploracion_hls_cmds.py  (este archivo)
    └── data/
        ├── HS.csv
        ├── LS.csv
        ├── Mix.csv
        ├── HS/
        │   └── (archivos .wav de sonidos cardíacos)
        ├── LS/
        │   └── (archivos .wav de sonidos pulmonares)
        └── Mix/
            └── (archivos .wav de sonidos mixtos)

DEPENDENCIAS:
    pip install pandas numpy matplotlib seaborn librosa scikit-learn
"""

# =============================================================================
# SECCIÓN 1: IMPORTACIONES Y CONFIGURACIÓN
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
sns.set_palette("husl")

# Intentar importar librosa (para procesamiento de audio)
try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
    print("✓ librosa disponible para análisis de audio")
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠ librosa no instalado. Ejecutar: pip install librosa")

# =============================================================================
# SECCIÓN 2: CONFIGURACIÓN DE RUTAS
# =============================================================================

# Modificar esta ruta según tu estructura de carpetas
DATA_DIR = Path("data")

# Rutas a los archivos CSV
HS_CSV = DATA_DIR / "HS.csv"
LS_CSV = DATA_DIR / "LS.csv"
MIX_CSV = DATA_DIR / "Mix.csv"

# Rutas a las carpetas de audio
HS_AUDIO_DIR = DATA_DIR / "HS"
LS_AUDIO_DIR = DATA_DIR / "LS"
MIX_AUDIO_DIR = DATA_DIR / "Mix"

# =============================================================================
# SECCIÓN 3: FUNCIONES AUXILIARES
# =============================================================================

def check_data_exists():
    """Verifica que los archivos de datos existan."""
    files_status = {
        "HS.csv": HS_CSV.exists(),
        "LS.csv": LS_CSV.exists(),
        "Mix.csv": MIX_CSV.exists(),
        "HS/ folder": HS_AUDIO_DIR.exists(),
        "LS/ folder": LS_AUDIO_DIR.exists(),
        "Mix/ folder": MIX_AUDIO_DIR.exists(),
    }
    
    print("\n" + "="*60)
    print("VERIFICACIÓN DE ARCHIVOS")
    print("="*60)
    
    all_exist = True
    for file, exists in files_status.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")
        if not exists:
            all_exist = False
    
    if not all_exist:
        print("\n⚠ ATENCIÓN: Faltan archivos. Descargar de:")
        print("  https://data.mendeley.com/datasets/8972jxbpmp/3")
    
    return all_exist


def load_metadata():
    """Carga los archivos CSV con metadatos."""
    dfs = {}
    
    if HS_CSV.exists():
        dfs['heart'] = pd.read_csv(HS_CSV)
        dfs['heart']['source'] = 'heart'
        print(f"✓ HS.csv cargado: {len(dfs['heart'])} registros")
    
    if LS_CSV.exists():
        dfs['lung'] = pd.read_csv(LS_CSV)
        dfs['lung']['source'] = 'lung'
        print(f"✓ LS.csv cargado: {len(dfs['lung'])} registros")
    
    if MIX_CSV.exists():
        dfs['mix'] = pd.read_csv(MIX_CSV)
        dfs['mix']['source'] = 'mix'
        print(f"✓ Mix.csv cargado: {len(dfs['mix'])} registros")
    
    return dfs


def explore_dataframe(df, name):
    """Explora un DataFrame mostrando información básica."""
    print(f"\n{'='*60}")
    print(f"EXPLORACIÓN: {name}")
    print(f"{'='*60}")
    
    print(f"\n📊 Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
    
    print(f"\n📋 Columnas disponibles:")
    for col in df.columns:
        dtype = df[col].dtype
        n_unique = df[col].nunique()
        print(f"   • {col}: {dtype} ({n_unique} valores únicos)")
    
    print(f"\n🔍 Primeras 5 filas:")
    print(df.head().to_string())
    
    print(f"\n📈 Valores nulos:")
    null_counts = df.isnull().sum()
    if null_counts.sum() == 0:
        print("   No hay valores nulos ✓")
    else:
        print(null_counts[null_counts > 0])
    
    return df


def plot_class_distribution(df, column, title, ax=None):
    """Genera gráfico de distribución de clases."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    counts = df[column].value_counts()
    colors = sns.color_palette("husl", len(counts))
    
    bars = ax.barh(counts.index, counts.values, color=colors)
    ax.set_xlabel('Cantidad de muestras')
    ax.set_title(title)
    
    # Agregar etiquetas con valores
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{count}', va='center', fontsize=9)
    
    ax.set_xlim(0, max(counts.values) * 1.15)
    
    return ax


def create_binary_label(df, sound_type_col):
    """
    Crea etiqueta binaria: Normal (0) vs Patológico (1)
    
    IMPORTANTE: Ajustar según los nombres exactos en tu CSV
    """
    # Nombres esperados para "Normal" (ajustar si es diferente)
    normal_patterns = ['Normal Heart', 'Normal Lung', 'normal', 'Normal']
    
    df = df.copy()
    df['is_pathological'] = ~df[sound_type_col].str.contains('|'.join(normal_patterns), 
                                                              case=False, na=False)
    df['binary_label'] = df['is_pathological'].map({True: 'Patológico', False: 'Normal'})
    
    return df


# =============================================================================
# SECCIÓN 4: ANÁLISIS DE AUDIO (requiere librosa)
# =============================================================================

def load_audio_file(filepath, sr=22050):
    """Carga un archivo de audio."""
    if not LIBROSA_AVAILABLE:
        print("⚠ librosa no disponible")
        return None, None
    
    try:
        y, sr = librosa.load(filepath, sr=sr)
        return y, sr
    except Exception as e:
        print(f"Error cargando {filepath}: {e}")
        return None, None


def plot_audio_analysis(filepath, title="Análisis de Audio"):
    """
    Genera visualización completa de un archivo de audio:
    - Forma de onda
    - Espectrograma
    - Mel espectrograma
    - MFCCs
    """
    if not LIBROSA_AVAILABLE:
        print("⚠ librosa necesario para análisis de audio")
        return
    
    # Cargar audio
    y, sr = load_audio_file(filepath)
    if y is None:
        return
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{title}\nArchivo: {Path(filepath).name}', fontsize=14, fontweight='bold')
    
    # 1. Forma de onda
    ax1 = axes[0, 0]
    librosa.display.waveshow(y, sr=sr, ax=ax1, color='steelblue')
    ax1.set_title('Forma de Onda')
    ax1.set_xlabel('Tiempo (s)')
    ax1.set_ylabel('Amplitud')
    
    # 2. Espectrograma (STFT)
    ax2 = axes[0, 1]
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img2 = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax2, cmap='magma')
    ax2.set_title('Espectrograma (STFT)')
    fig.colorbar(img2, ax=ax2, format='%+2.0f dB')
    
    # 3. Mel Espectrograma
    ax3 = axes[1, 0]
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img3 = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax3, cmap='magma')
    ax3.set_title('Mel Espectrograma')
    fig.colorbar(img3, ax=ax3, format='%+2.0f dB')
    
    # 4. MFCCs
    ax4 = axes[1, 1]
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    img4 = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax4, cmap='coolwarm')
    ax4.set_title('MFCCs (13 coeficientes)')
    ax4.set_ylabel('Coeficiente MFCC')
    fig.colorbar(img4, ax=ax4)
    
    plt.tight_layout()
    plt.savefig(f'analisis_audio_{Path(filepath).stem}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Información del audio
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"\n📊 Información del audio:")
    print(f"   • Duración: {duration:.2f} segundos")
    print(f"   • Sample rate: {sr} Hz")
    print(f"   • Samples totales: {len(y):,}")
    
    return y, sr


def extract_features_summary(y, sr):
    """
    Extrae un resumen de features de audio para análisis exploratorio.
    
    Retorna un diccionario con estadísticas de las features principales.
    """
    if not LIBROSA_AVAILABLE or y is None:
        return {}
    
    features = {}
    
    # MFCCs (13 coeficientes)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
    
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    features['spectral_centroid_std'] = np.std(spectral_centroid)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_rolloff_std'] = np.std(spectral_rolloff)
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    
    # RMS energy
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    
    return features


# =============================================================================
# SECCIÓN 5: EJECUCIÓN PRINCIPAL
# =============================================================================

def main():
    """Función principal de exploración."""
    
    print("\n" + "="*70)
    print("   EXPLORACIÓN INICIAL DEL DATASET HLS-CMDS")
    print("   Tesis: Clasificación de Sonidos Cardiopulmonares con ML")
    print("="*70)
    
    # -----------------------------------------------------------------
    # PASO 1: Verificar que existan los datos
    # -----------------------------------------------------------------
    data_exists = check_data_exists()
    
    if not data_exists:
        print("\n" + "-"*60)
        print("MODO DEMO: Mostrando estructura esperada del análisis")
        print("-"*60)
        print("""
        Una vez descargados los datos, este script realizará:
        
        1. ANÁLISIS DE METADATOS:
           • Exploración de cada CSV (HS, LS, Mix)
           • Distribución de clases de sonidos
           • Análisis de localizaciones anatómicas
           • Identificación de desbalance de clases
        
        2. ANÁLISIS DE AUDIO:
           • Visualización de formas de onda
           • Espectrogramas (STFT)
           • Mel espectrogramas
           • MFCCs (features principales para ML)
        
        3. PREPARACIÓN PARA MODELADO:
           • Creación de etiquetas binarias (Normal vs Patológico)
           • Análisis de viabilidad de clasificación multiclase
           • Recomendaciones de preprocesamiento
        """)
        return
    
    # -----------------------------------------------------------------
    # PASO 2: Cargar metadatos
    # -----------------------------------------------------------------
    print("\n" + "="*60)
    print("CARGANDO METADATOS...")
    print("="*60)
    
    dfs = load_metadata()
    
    # -----------------------------------------------------------------
    # PASO 3: Explorar cada dataset
    # -----------------------------------------------------------------
    for name, df in dfs.items():
        explore_dataframe(df, name.upper())
    
    # -----------------------------------------------------------------
    # PASO 4: Análisis de distribución de clases
    # -----------------------------------------------------------------
    print("\n" + "="*60)
    print("ANÁLISIS DE DISTRIBUCIÓN DE CLASES")
    print("="*60)
    
    # Identificar columna de tipo de sonido (ajustar según CSV real)
    # Nombres posibles: 'Sound Type', 'sound_type', 'Type', 'heart_sound_type', etc.
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Determinar nombres de columnas (ajustar según tu CSV)
    # Esto es un placeholder - ajustar cuando veas los nombres reales
    
    if 'heart' in dfs:
        # Buscar columna que contenga tipo de sonido
        type_cols = [col for col in dfs['heart'].columns if 'type' in col.lower() or 'sound' in col.lower()]
        if type_cols:
            plot_class_distribution(dfs['heart'], type_cols[0], 
                                  'Distribución de Sonidos Cardíacos', axes[0])
        print(f"\n📊 Sonidos cardíacos - Columnas encontradas: {dfs['heart'].columns.tolist()}")
    
    if 'lung' in dfs:
        type_cols = [col for col in dfs['lung'].columns if 'type' in col.lower() or 'sound' in col.lower()]
        if type_cols:
            plot_class_distribution(dfs['lung'], type_cols[0],
                                  'Distribución de Sonidos Pulmonares', axes[1])
        print(f"\n📊 Sonidos pulmonares - Columnas encontradas: {dfs['lung'].columns.tolist()}")
    
    if 'mix' in dfs:
        print(f"\n📊 Sonidos mixtos - Columnas encontradas: {dfs['mix'].columns.tolist()}")
    
    plt.tight_layout()
    plt.savefig('distribucion_clases.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # -----------------------------------------------------------------
    # PASO 5: Análisis de desbalance
    # -----------------------------------------------------------------
    print("\n" + "="*60)
    print("ANÁLISIS DE DESBALANCE DE CLASES")
    print("="*60)
    
    for name, df in dfs.items():
        type_cols = [col for col in df.columns if 'type' in col.lower() or 'sound' in col.lower()]
        if type_cols:
            col = type_cols[0]
            counts = df[col].value_counts()
            ratio = counts.max() / counts.min()
            print(f"\n{name.upper()}:")
            print(f"   Clase mayoritaria: {counts.idxmax()} ({counts.max()} muestras)")
            print(f"   Clase minoritaria: {counts.idxmin()} ({counts.min()} muestras)")
            print(f"   Ratio de desbalance: {ratio:.2f}:1")
    
    # -----------------------------------------------------------------
    # PASO 6: Análisis de audio (si hay archivos)
    # -----------------------------------------------------------------
    if LIBROSA_AVAILABLE:
        print("\n" + "="*60)
        print("ANÁLISIS DE AUDIO DE MUESTRA")
        print("="*60)
        
        # Buscar primer archivo de audio disponible
        for audio_dir in [HS_AUDIO_DIR, LS_AUDIO_DIR, MIX_AUDIO_DIR]:
            if audio_dir.exists():
                wav_files = list(audio_dir.glob("*.wav"))
                if wav_files:
                    print(f"\nAnalizando archivo de ejemplo: {wav_files[0].name}")
                    plot_audio_analysis(wav_files[0], 
                                      title=f"Ejemplo de {audio_dir.name}")
                    break
    
    # -----------------------------------------------------------------
    # PASO 7: Resumen y recomendaciones
    # -----------------------------------------------------------------
    print("\n" + "="*70)
    print("RESUMEN Y RECOMENDACIONES")
    print("="*70)
    
    total_samples = sum(len(df) for df in dfs.values())
    print(f"""
    📊 RESUMEN DEL DATASET:
       • Total de muestras: {total_samples}
       • Sonidos cardíacos: {len(dfs.get('heart', []))} 
       • Sonidos pulmonares: {len(dfs.get('lung', []))}
       • Sonidos mixtos: {len(dfs.get('mix', []))}
    
    🎯 OPCIONES DE CLASIFICACIÓN:
    
       OPCIÓN A - Binaria (Recomendada para empezar):
       • Normal vs Patológico
       • Más simple, mejor para baseline
       • Menos afectada por desbalance extremo
    
       OPCIÓN B - Multiclase cardíaca:
       • 10 clases de sonidos cardíacos
       • Mayor complejidad
       • Requiere manejo de desbalance
    
       OPCIÓN C - Multiclase pulmonar:
       • 6 clases de sonidos pulmonares
       • Complejidad media
       • Clases más balanceadas
    
    📝 PRÓXIMOS PASOS:
       1. Revisar los nombres exactos de columnas en los CSVs
       2. Analizar el desbalance de clases en detalle
       3. Decidir entre clasificación binaria vs multiclase
       4. Comenzar con extracción de features (MFCCs, espectrogramas)
    """)


# =============================================================================
# SECCIÓN 6: FUNCIONES ADICIONALES PARA FEATURE ENGINEERING
# =============================================================================

def extract_all_features(audio_dir, metadata_df, filename_col, label_col, max_files=None):
    """
    Extrae features de todos los archivos de audio.
    
    Parámetros:
    -----------
    audio_dir : Path
        Directorio con archivos de audio
    metadata_df : DataFrame
        DataFrame con metadatos
    filename_col : str
        Nombre de la columna con nombres de archivos
    label_col : str
        Nombre de la columna con etiquetas
    max_files : int, opcional
        Número máximo de archivos a procesar (para testing)
    
    Retorna:
    --------
    DataFrame con features extraídas
    """
    if not LIBROSA_AVAILABLE:
        print("⚠ librosa necesario para extracción de features")
        return None
    
    features_list = []
    
    files_to_process = metadata_df[filename_col].tolist()
    if max_files:
        files_to_process = files_to_process[:max_files]
    
    print(f"Extrayendo features de {len(files_to_process)} archivos...")
    
    for i, filename in enumerate(files_to_process):
        filepath = audio_dir / filename
        
        if not filepath.exists():
            print(f"  ⚠ Archivo no encontrado: {filename}")
            continue
        
        y, sr = load_audio_file(filepath)
        if y is None:
            continue
        
        # Extraer features
        features = extract_features_summary(y, sr)
        features['filename'] = filename
        
        # Agregar label
        label_mask = metadata_df[filename_col] == filename
        if label_mask.any():
            features['label'] = metadata_df.loc[label_mask, label_col].values[0]
        
        features_list.append(features)
        
        if (i + 1) % 10 == 0:
            print(f"  Procesados: {i + 1}/{len(files_to_process)}")
    
    features_df = pd.DataFrame(features_list)
    print(f"✓ Features extraídas: {features_df.shape}")
    
    return features_df


# =============================================================================
# EJECUTAR SCRIPT
# =============================================================================

if __name__ == "__main__":
    main()
    
    print("\n" + "="*70)
    print("Script completado. Archivos generados:")
    print("  • distribucion_clases.png (si los datos están disponibles)")
    print("  • analisis_audio_*.png (si hay archivos de audio)")
    print("="*70)
