# -*- coding: utf-8 -*-
"""
================================================================================
EXPLORACIÓN DETALLADA DEL DATASET HLS-CMDS - PASO A PASO
================================================================================

Autor: María Constanza Florio
Tesis: Maestría en Ciencia de Datos - ITBA

Este notebook está organizado en secciones para ejecutar paso a paso.
Cada sección tiene un objetivo claro y genera outputs específicos.

Para usar en Jupyter: copiar cada sección en una celda separada.
"""

# =============================================================================
# CELDA 1: SETUP E IMPORTACIONES
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from collections import Counter

# Configuración
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Librosa para audio
try:
    import librosa
    import librosa.display
    LIBROSA_OK = True
    print("✓ librosa disponible")
except ImportError:
    LIBROSA_OK = False
    print("⚠ Instalar librosa: pip install librosa")

# Rutas - MODIFICAR según tu estructura
DATA_DIR = Path("data")
print(f"✓ Directorio de datos: {DATA_DIR.absolute()}")

# =============================================================================
# CELDA 2: CARGAR Y EXPLORAR ESTRUCTURA DE CADA CSV
# =============================================================================

print("\n" + "="*70)
print("PARTE 1: ESTRUCTURA DE LOS ARCHIVOS CSV")
print("="*70)

# Cargar CSVs
hs_df = pd.read_csv(DATA_DIR / "HS.csv")
ls_df = pd.read_csv(DATA_DIR / "LS.csv")
mix_df = pd.read_csv(DATA_DIR / "Mix.csv")

print("\n📁 HS.csv (Heart Sounds)")
print(f"   Dimensiones: {hs_df.shape}")
print(f"   Columnas: {hs_df.columns.tolist()}")

print("\n📁 LS.csv (Lung Sounds)")
print(f"   Dimensiones: {ls_df.shape}")
print(f"   Columnas: {ls_df.columns.tolist()}")

print("\n📁 Mix.csv (Mixed Sounds)")
print(f"   Dimensiones: {mix_df.shape}")
print(f"   Columnas: {mix_df.columns.tolist()}")

# =============================================================================
# CELDA 3: EXPLORACIÓN DETALLADA - HEART SOUNDS (HS)
# =============================================================================

print("\n" + "="*70)
print("PARTE 2: ANÁLISIS DETALLADO DE SONIDOS CARDÍACOS (HS)")
print("="*70)

print("\n📊 Primeras filas de HS.csv:")
print(hs_df.head(10).to_string())

print("\n📊 Tipos de datos:")
print(hs_df.dtypes)

print("\n📊 Valores únicos por columna:")
for col in hs_df.columns:
    print(f"\n   {col}:")
    print(f"   {hs_df[col].value_counts().to_dict()}")

# Identificar columna de tipo de sonido
# Buscar columnas que contengan información de patología
hs_type_col = None
for col in hs_df.columns:
    if 'type' in col.lower() or 'sound' in col.lower() or 'label' in col.lower():
        hs_type_col = col
        break

if hs_type_col:
    print(f"\n✓ Columna de tipo de sonido identificada: '{hs_type_col}'")
else:
    print("\n⚠ No se encontró columna de tipo. Revisar manualmente.")
    print("   Columnas disponibles:", hs_df.columns.tolist())

# =============================================================================
# CELDA 4: EXPLORACIÓN DETALLADA - LUNG SOUNDS (LS)
# =============================================================================

print("\n" + "="*70)
print("PARTE 3: ANÁLISIS DETALLADO DE SONIDOS PULMONARES (LS)")
print("="*70)

print("\n📊 Primeras filas de LS.csv:")
print(ls_df.head(10).to_string())

print("\n📊 Valores únicos por columna:")
for col in ls_df.columns:
    print(f"\n   {col}:")
    print(f"   {ls_df[col].value_counts().to_dict()}")

# =============================================================================
# CELDA 5: EXPLORACIÓN DETALLADA - MIXED SOUNDS
# =============================================================================

print("\n" + "="*70)
print("PARTE 4: ANÁLISIS DETALLADO DE SONIDOS MIXTOS (Mix)")
print("="*70)

print("\n📊 Primeras filas de Mix.csv:")
print(mix_df.head(10).to_string())

print("\n📊 Dimensiones:", mix_df.shape)

print("\n📊 Valores únicos por columna:")
for col in mix_df.columns:
    n_unique = mix_df[col].nunique()
    print(f"\n   {col} ({n_unique} valores únicos):")
    if n_unique <= 20:
        print(f"   {mix_df[col].value_counts().to_dict()}")
    else:
        print(f"   Primeros 10: {mix_df[col].value_counts().head(10).to_dict()}")

# El Mix tiene COMBINACIONES de heart + lung sounds
# Buscar columnas separadas para heart_type y lung_type
print("\n📊 Estructura del Mix:")
print("   Este archivo contiene sonidos mixtos (cardíacos + pulmonares simultáneos)")
print("   Debería tener columnas separadas para tipo cardíaco y tipo pulmonar")

# =============================================================================
# CELDA 6: VISUALIZACIÓN - DISTRIBUCIÓN DE CLASES (CORREGIDO)
# =============================================================================

print("\n" + "="*70)
print("PARTE 5: VISUALIZACIÓN DE DISTRIBUCIÓN DE CLASES")
print("="*70)

# Función mejorada para graficar
def plot_distribution(df, column, title, figsize=(10, 6)):
    """Grafica distribución de una columna categórica."""
    fig, ax = plt.subplots(figsize=figsize)
    
    counts = df[column].value_counts().sort_values(ascending=True)
    colors = plt.cm.husl(np.linspace(0.1, 0.9, len(counts)))
    
    bars = ax.barh(range(len(counts)), counts.values, color=colors)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index)
    ax.set_xlabel('Cantidad de muestras')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Etiquetas con valores y porcentaje
    total = counts.sum()
    for i, (bar, count) in enumerate(zip(bars, counts.values)):
        pct = count / total * 100
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{count} ({pct:.1f}%)', va='center', fontsize=10)
    
    ax.set_xlim(0, max(counts.values) * 1.25)
    plt.tight_layout()
    return fig, ax

# Detectar automáticamente las columnas de tipo
# (ajustar según los nombres reales en tu CSV)

# Para HS - buscar columna de tipo de sonido
hs_type_candidates = [col for col in hs_df.columns 
                      if any(x in col.lower() for x in ['type', 'sound', 'label', 'class'])]
print(f"\nColumnas candidatas en HS: {hs_type_candidates}")

# Para LS
ls_type_candidates = [col for col in ls_df.columns 
                      if any(x in col.lower() for x in ['type', 'sound', 'label', 'class'])]
print(f"Columnas candidatas en LS: {ls_type_candidates}")

# Para Mix
mix_candidates = [col for col in mix_df.columns 
                  if any(x in col.lower() for x in ['type', 'sound', 'label', 'class', 'heart', 'lung'])]
print(f"Columnas candidatas en Mix: {mix_candidates}")

# =============================================================================
# CELDA 7: GRÁFICOS DE DISTRIBUCIÓN
# =============================================================================

# NOTA: Ajustar los nombres de columnas según tu CSV real
# Ejecutar celda 3, 4, 5 primero para ver los nombres exactos

# Ejemplo (modificar según tus columnas reales):
try:
    # Intentar con nombres comunes
    hs_type_col = 'Heart Sound Type' if 'Heart Sound Type' in hs_df.columns else hs_df.columns[1]
    ls_type_col = 'Lung Sound Type' if 'Lung Sound Type' in ls_df.columns else ls_df.columns[1]
    
    fig1, ax1 = plot_distribution(hs_df, hs_type_col, 
                                   f'Distribución de Sonidos Cardíacos (n={len(hs_df)})')
    plt.savefig('dist_heart_sounds.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    fig2, ax2 = plot_distribution(ls_df, ls_type_col,
                                   f'Distribución de Sonidos Pulmonares (n={len(ls_df)})')
    plt.savefig('dist_lung_sounds.png', dpi=150, bbox_inches='tight')
    plt.show()
    
except Exception as e:
    print(f"⚠ Error al graficar: {e}")
    print("   Revisar nombres de columnas en celdas anteriores")

# =============================================================================
# CELDA 8: ANÁLISIS ESPECÍFICO DEL MIX
# =============================================================================

print("\n" + "="*70)
print("PARTE 6: ANÁLISIS ESPECÍFICO DE SONIDOS MIXTOS")
print("="*70)

print("\n📊 Estructura completa del Mix:")
print(mix_df.info())

print("\n📊 Todas las columnas del Mix:")
for i, col in enumerate(mix_df.columns):
    print(f"   {i}: '{col}'")

# Ver si tiene columnas separadas para heart y lung
heart_cols = [col for col in mix_df.columns if 'heart' in col.lower()]
lung_cols = [col for col in mix_df.columns if 'lung' in col.lower()]

print(f"\n   Columnas relacionadas con Heart: {heart_cols}")
print(f"   Columnas relacionadas con Lung: {lung_cols}")

# Si tiene ambas columnas, podemos hacer análisis cruzado
if heart_cols and lung_cols:
    print("\n📊 Combinaciones Heart x Lung en Mix:")
    cross_tab = pd.crosstab(mix_df[heart_cols[0]], mix_df[lung_cols[0]])
    print(cross_tab)

# =============================================================================
# CELDA 9: CREAR ETIQUETAS BINARIAS
# =============================================================================

print("\n" + "="*70)
print("PARTE 7: CREAR ETIQUETAS BINARIAS (Normal vs Patológico)")
print("="*70)

def create_binary_labels(df, type_col, normal_values):
    """
    Crea columna binaria: 0=Normal, 1=Patológico
    
    Parámetros:
    -----------
    df : DataFrame
    type_col : str - nombre de la columna con tipos de sonido
    normal_values : list - valores que se consideran "Normal"
    """
    df = df.copy()
    df['is_normal'] = df[type_col].isin(normal_values)
    df['binary_label'] = df['is_normal'].map({True: 'Normal', False: 'Patológico'})
    df['binary_numeric'] = df['is_normal'].map({True: 0, False: 1})
    return df

# Aplicar a HS (ajustar 'Normal' según el valor real en tu CSV)
# Primero verificar el valor exacto de Normal
print("\nValores en columna de tipo HS:")
print(hs_df.iloc[:, 1].value_counts())  # Ajustar índice según tu CSV

# =============================================================================
# CELDA 10: ANÁLISIS DE DESBALANCE DETALLADO
# =============================================================================

print("\n" + "="*70)
print("PARTE 8: ANÁLISIS DE DESBALANCE DETALLADO")
print("="*70)

def analyze_imbalance(df, type_col, name):
    """Análisis detallado de desbalance de clases."""
    counts = df[type_col].value_counts()
    total = len(df)
    
    print(f"\n{'='*50}")
    print(f"📊 {name}")
    print(f"{'='*50}")
    print(f"Total de muestras: {total}")
    print(f"Número de clases: {len(counts)}")
    print(f"\nDistribución:")
    
    for clase, count in counts.items():
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        print(f"   {clase:25s}: {count:3d} ({pct:5.1f}%) {bar}")
    
    # Métricas de desbalance
    ratio = counts.max() / counts.min()
    entropy = -sum((c/total) * np.log2(c/total) for c in counts.values)
    max_entropy = np.log2(len(counts))
    balance_score = entropy / max_entropy  # 1 = perfectamente balanceado
    
    print(f"\nMétricas de desbalance:")
    print(f"   • Ratio max/min: {ratio:.2f}:1")
    print(f"   • Balance score: {balance_score:.2f} (1.0 = perfecto)")
    print(f"   • Clase mayoritaria: {counts.idxmax()} ({counts.max()})")
    print(f"   • Clase minoritaria: {counts.idxmin()} ({counts.min()})")
    
    return counts

# Aplicar a cada dataset (ajustar nombres de columnas)
# hs_counts = analyze_imbalance(hs_df, 'Heart Sound Type', 'SONIDOS CARDÍACOS')
# ls_counts = analyze_imbalance(ls_df, 'Lung Sound Type', 'SONIDOS PULMONARES')

# =============================================================================
# CELDA 11: ANÁLISIS DE LOCALIZACIONES ANATÓMICAS
# =============================================================================

print("\n" + "="*70)
print("PARTE 9: ANÁLISIS DE LOCALIZACIONES ANATÓMICAS")
print("="*70)

# Buscar columnas de localización
loc_cols_hs = [col for col in hs_df.columns if any(x in col.lower() 
               for x in ['location', 'position', 'site', 'landmark'])]
loc_cols_ls = [col for col in ls_df.columns if any(x in col.lower() 
               for x in ['location', 'position', 'site', 'landmark'])]

print(f"Columnas de localización en HS: {loc_cols_hs}")
print(f"Columnas de localización en LS: {loc_cols_ls}")

# Si existen, analizar distribución
if loc_cols_hs:
    print(f"\n📍 Localizaciones en HS ({loc_cols_hs[0]}):")
    print(hs_df[loc_cols_hs[0]].value_counts())

if loc_cols_ls:
    print(f"\n📍 Localizaciones en LS ({loc_cols_ls[0]}):")
    print(ls_df[loc_cols_ls[0]].value_counts())

# =============================================================================
# CELDA 12: ANÁLISIS DE ARCHIVOS DE AUDIO
# =============================================================================

print("\n" + "="*70)
print("PARTE 10: ANÁLISIS DE ARCHIVOS DE AUDIO")
print("="*70)

# Verificar archivos de audio
hs_audio_dir = DATA_DIR / "HS"
ls_audio_dir = DATA_DIR / "LS"
mix_audio_dir = DATA_DIR / "Mix"

for name, audio_dir in [("HS", hs_audio_dir), ("LS", ls_audio_dir), ("Mix", mix_audio_dir)]:
    if audio_dir.exists():
        wav_files = list(audio_dir.glob("*.wav"))
        print(f"\n📁 {name}/")
        print(f"   Archivos .wav encontrados: {len(wav_files)}")
        if wav_files:
            print(f"   Ejemplo: {wav_files[0].name}")
    else:
        print(f"\n⚠ Carpeta {name}/ no encontrada")

# =============================================================================
# CELDA 13: COMPARAR AUDIO NORMAL VS PATOLÓGICO
# =============================================================================

print("\n" + "="*70)
print("PARTE 11: COMPARACIÓN VISUAL - NORMAL VS PATOLÓGICO")
print("="*70)

if LIBROSA_OK:
    def plot_audio_comparison(file1, file2, label1, label2, sr=22050):
        """Compara dos archivos de audio lado a lado."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(f'Comparación: {label1} vs {label2}', fontsize=14, fontweight='bold')
        
        for idx, (filepath, label) in enumerate([(file1, label1), (file2, label2)]):
            y, sr = librosa.load(filepath, sr=sr)
            
            # Forma de onda
            ax_wave = axes[0, idx]
            librosa.display.waveshow(y, sr=sr, ax=ax_wave)
            ax_wave.set_title(f'{label} - Forma de Onda')
            ax_wave.set_xlabel('Tiempo (s)')
            
            # Mel espectrograma
            ax_mel = axes[1, idx]
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', 
                                           y_axis='mel', ax=ax_mel, cmap='magma')
            ax_mel.set_title(f'{label} - Mel Espectrograma')
            fig.colorbar(img, ax=ax_mel, format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig('comparacion_normal_patologico.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Buscar un archivo Normal y uno Patológico para comparar
    # (Ajustar según los archivos disponibles)
    print("Para comparar, ejecutar:")
    print("plot_audio_comparison('data/HS/archivo_normal.wav', 'data/HS/archivo_patologico.wav', 'Normal', 'Patológico')")

# =============================================================================
# CELDA 14: RESUMEN EJECUTIVO
# =============================================================================

print("\n" + "="*70)
print("RESUMEN EJECUTIVO")
print("="*70)

print("""
📊 HALLAZGOS PRINCIPALES:

1. TAMAÑO DEL DATASET:
   • Heart Sounds: 50 muestras (10 clases)
   • Lung Sounds: 50 muestras (6 clases)
   • Mixed: 145 muestras (combinaciones)
   • TOTAL: ~245 muestras únicas
   
   ⚠ Dataset PEQUEÑO - riesgo de overfitting

2. DESBALANCE:
   • HS: ratio 4.5:1 (Normal:9, S4:2)
   • LS: ratio 2.4:1 (Normal:12, Fine Crackles:5)
   
   ✓ Desbalance moderado, manejable con técnicas estándar

3. OPCIONES DE CLASIFICACIÓN:

   OPCIÓN A - Binaria (RECOMENDADA):
   • Normal vs Patológico
   • HS: 9 Normal vs 41 Patológico
   • LS: 12 Normal vs 38 Patológico
   • Más robusto con pocos datos
   
   OPCIÓN B - Multiclase agrupada:
   • Agrupar patologías similares:
     - Murmurs (Early+Mid+Late Systolic + Late Diastolic)
     - Arritmias (AF, Tachycardia, AV Block)
     - Sonidos extra (S3, S4)
   
   OPCIÓN C - Multiclase completa:
   • 10 clases cardíacas / 6 pulmonares
   • ⚠ Muy pocas muestras por clase

4. PRÓXIMOS PASOS:
   □ Decidir enfoque (binario vs multiclase)
   □ Extraer features de audio (MFCCs, espectrogramas)
   □ Implementar validación cruzada estratificada
   □ Considerar data augmentation
""")

# =============================================================================
# CELDA 15: EXPORTAR DATOS PROCESADOS
# =============================================================================

print("\n" + "="*70)
print("PARTE 12: PREPARAR DATOS PARA MODELADO")
print("="*70)

# Combinar datasets con etiquetas
# (Descomentar y ajustar cuando tengas los nombres correctos de columnas)

"""
# Ejemplo de cómo preparar los datos finales:

# 1. Agregar columna de origen
hs_df['source'] = 'heart'
ls_df['source'] = 'lung'

# 2. Crear etiquetas binarias
hs_df = create_binary_labels(hs_df, 'Heart Sound Type', ['Normal'])
ls_df = create_binary_labels(ls_df, 'Lung Sound Type', ['Normal'])

# 3. Estandarizar nombres de columnas
hs_df = hs_df.rename(columns={'Heart Sound Type': 'sound_type'})
ls_df = ls_df.rename(columns={'Lung Sound Type': 'sound_type'})

# 4. Guardar datos procesados
hs_df.to_csv('data/hs_processed.csv', index=False)
ls_df.to_csv('data/ls_processed.csv', index=False)

print("✓ Datos procesados guardados")
"""

print("\n✓ Notebook de exploración completado")
print("   Ejecutar celda por celda para análisis detallado")
