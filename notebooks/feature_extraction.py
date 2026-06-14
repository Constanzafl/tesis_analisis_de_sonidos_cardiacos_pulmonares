# %% [markdown]
# # Extracción de características — versión corregida
#
# Reemplaza las celdas de extracción de features del EDA original.
# Cambios principales respecto de la versión anterior:
#  1. `process_all_files` ya NO descarta archivos en silencio: reporta y
#     devuelve la lista de faltantes/errores (corrige la pérdida de 15/50 LS).
#  2. Diagnóstico previo que compara los IDs del metadato con los .wav en disco.
#  3. `extract_features` robusto ante audios cortos (delta) y valores NaN/inf.
#  4. La feature `tempo` queda como OPCIONAL y marcada (proxy poco válido en
#     sonidos cardíacos). Controlable con INCLUDE_TEMPO.
#  5. Semilla global para reproducibilidad.

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import librosa

RANDOM_STATE = 42
SR = 22050           # frecuencia de muestreo de carga
N_MFCC = 13
INCLUDE_TEMPO = False  # poner True solo si se justifica clínicamente (ver informe 2.3.3)

np.random.seed(RANDOM_STATE)

DATA_DIR = Path("../data")
AUDIO = {"HS": DATA_DIR / "HS", "LS": DATA_DIR / "LS", "Mix": DATA_DIR / "Mix"}
OUT_DIR = Path("../outputs")
OUT_DIR.mkdir(exist_ok=True)


# %%
def extract_features(y, sr=SR, include_tempo=INCLUDE_TEMPO):
    """Convierte una señal de audio en un vector fijo de características.

    Familias: MFCC (13) + delta, espectrales (centroid, bandwidth, rolloff,
    contrast, flatness), ZCR, RMS y (opcional) tempo.
    Devuelve un dict {nombre_feature: valor}. Maneja audios cortos y NaN.
    """
    features = {}

    # --- MFCC (media/std/max/min por coeficiente) ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        c = mfccs[i]
        features[f"mfcc_{i+1}_mean"] = np.mean(c)
        features[f"mfcc_{i+1}_std"] = np.std(c)
        features[f"mfcc_{i+1}_max"] = np.max(c)
        features[f"mfcc_{i+1}_min"] = np.min(c)

    # --- Delta MFCC (width seguro para señales con pocas tramas) ---
    n_frames = mfccs.shape[1]
    width = min(9, n_frames if n_frames % 2 == 1 else n_frames - 1)
    width = max(width, 3)  # delta requiere width impar >= 3
    if n_frames >= 3:
        mfcc_delta = librosa.feature.delta(mfccs, width=width)
    else:
        mfcc_delta = np.zeros_like(mfccs)
    for i in range(N_MFCC):
        features[f"mfcc_delta_{i+1}_mean"] = np.mean(mfcc_delta[i])
        features[f"mfcc_delta_{i+1}_std"] = np.std(mfcc_delta[i])

    # --- Espectrales ---
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features["spectral_centroid_mean"] = np.mean(spec_cent)
    features["spectral_centroid_std"] = np.std(spec_cent)

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features["spectral_bandwidth_mean"] = np.mean(spec_bw)
    features["spectral_bandwidth_std"] = np.std(spec_bw)

    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features["spectral_rolloff_mean"] = np.mean(spec_rolloff)
    features["spectral_rolloff_std"] = np.std(spec_rolloff)

    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(spec_contrast.shape[0]):
        features[f"spectral_contrast_{i+1}_mean"] = np.mean(spec_contrast[i])
        features[f"spectral_contrast_{i+1}_std"] = np.std(spec_contrast[i])

    spec_flat = librosa.feature.spectral_flatness(y=y)[0]
    features["spectral_flatness_mean"] = np.mean(spec_flat)
    features["spectral_flatness_std"] = np.std(spec_flat)

    # --- Temporales / energía ---
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features["zcr_mean"] = np.mean(zcr)
    features["zcr_std"] = np.std(zcr)

    rms = librosa.feature.rms(y=y)[0]
    features["rms_mean"] = np.mean(rms)
    features["rms_std"] = np.std(rms)
    features["rms_max"] = np.max(rms)

    # --- Tempo (OPCIONAL: proxy poco válido en sonidos cardíacos) ---
    if include_tempo:
        tempo = librosa.beat.beat_track(y=y, sr=sr)[0]
        features["tempo"] = float(np.atleast_1d(tempo)[0])

    # Reemplazar posibles NaN/inf por 0.0 (señales degeneradas)
    for k, v in features.items():
        if not np.isfinite(v):
            features[k] = 0.0
    return features


# %%
def check_files(metadata_df, audio_dir, id_col):
    """Diagnóstico: qué IDs del metadato NO tienen su .wav en disco.

    Devuelve (faltantes, en_disco_sin_metadato) para entender la pérdida
    de archivos antes de extraer features.
    """
    audio_dir = Path(audio_dir)
    ids_meta = set(metadata_df[id_col].astype(str))
    ids_disk = {p.stem for p in audio_dir.glob("*.wav")}
    faltantes = sorted(ids_meta - ids_disk)
    sobrantes = sorted(ids_disk - ids_meta)
    print(f"[{audio_dir.name}] metadato={len(ids_meta)} | en disco={len(ids_disk)}"
          f" | faltantes={len(faltantes)} | en disco sin metadato={len(sobrantes)}")
    if faltantes:
        print("  Faltantes (revisar nombres/.wav):", faltantes)
    if sobrantes:
        print("  En disco sin metadato:", sobrantes)
    return faltantes, sobrantes


# %%
def normalize_lung_id(raw_id):
    """Repara los IDs de Lung Sounds para que coincidan con los .wav en disco.

    Problema detectado en HLS-CMDS: el metadato usa abreviaturas distintas a
    las de los archivos para los crackles, más un espacio sobrante en un ID.
      - quita espacios sobrantes (p. ej. ' M_W_LMA' -> 'M_W_LMA')
      - G -> CC  (Coarse Crackles)
      - C -> FC  (Fine Crackles)
    El mapeo es 1:1 y está forzado por los conteos (9 'G' <-> 9 'CC', 5 'C' <-> 5 'FC').
    Documentar como paso de limpieza de datos (informe, sección 3.3).
    """
    parts = str(raw_id).strip().split("_")
    if len(parts) == 3:
        gender, stype, loc = parts
        stype = {"G": "CC", "C": "FC"}.get(stype, stype)
        return f"{gender}_{stype}_{loc}"
    return str(raw_id).strip()


# %%
def process_all_files(metadata_df, audio_dir, id_col, label_col, binary_col=None):
    """Extrae features de todos los archivos. NO descarta en silencio:
    reporta archivos faltantes y errores, y verifica el conteo final.
    """
    audio_dir = Path(audio_dir)
    records, missing, errors = [], [], []

    for _, row in metadata_df.iterrows():
        file_id = str(row[id_col])
        filepath = audio_dir / f"{file_id}.wav"
        if not filepath.exists():
            missing.append(file_id)
            continue
        try:
            y, sr = librosa.load(filepath, sr=SR)
            feats = extract_features(y, sr)
            feats["file_id"] = file_id
            feats["label"] = row[label_col]
            if binary_col is not None:
                feats["label_binary"] = row[binary_col]
            records.append(feats)
        except Exception as e:  # noqa: BLE001 - queremos saber QUÉ falló
            errors.append((file_id, repr(e)))

    df = pd.DataFrame(records)
    n_meta = len(metadata_df)
    print(f"[{audio_dir.name}] procesados={len(df)}/{n_meta}"
          f" | faltantes={len(missing)} | errores={len(errors)}")
    if missing:
        print("  FALTANTES:", missing)
    if errors:
        print("  ERRORES:", errors)
    if len(df) != n_meta:
        print(f"  ⚠ ATENCIÓN: se esperaban {n_meta} filas y se obtuvieron {len(df)}.")
    return df, missing, errors


# %% [markdown]
# ## Uso (descomentar cuando se ejecute con los datos reales)
#
# ```python
# hs = pd.read_csv(DATA_DIR / "HS.csv")
# ls = pd.read_csv(DATA_DIR / "LS.csv")
#
# # 1) Etiqueta binaria: 0 = Normal, 1 = Patológico
# hs["binary_label"] = (hs["Heart Sound Type"] != "Normal").astype(int)
# ls["binary_label"] = (ls["Lung Sound Type"]  != "Normal").astype(int)
#
# # 1.b) Reparar los IDs pulmonares (recupera los 15 archivos por nombre mal escrito)
# ls["Lung Sound ID"] = ls["Lung Sound ID"].map(normalize_lung_id)
#
# # 2) Diagnóstico de archivos ANTES de extraer
# check_files(hs, AUDIO["HS"], "Heart Sound ID")
# check_files(ls, AUDIO["LS"], "Lung Sound ID")   # ahora debe dar faltantes=0
#
# # 3) Extracción (ahora reporta todo lo que se pierde)
# hs_features, hs_missing, hs_errors = process_all_files(
#     hs, AUDIO["HS"], "Heart Sound ID", "Heart Sound Type", "binary_label")
# ls_features, ls_missing, ls_errors = process_all_files(
#     ls, AUDIO["LS"], "Lung Sound ID", "Lung Sound Type", "binary_label")
#
# # 4) Guardar
# hs_features.to_csv(OUT_DIR / "hs_features.csv", index=False)
# ls_features.to_csv(OUT_DIR / "ls_features.csv", index=False)
# ```
