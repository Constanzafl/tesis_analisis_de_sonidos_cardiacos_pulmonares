# %% [markdown]
# # 03 — Robustez ante sonido superpuesto (subset Mix)
#
# El HLS-CMDS incluye 145 grabaciones **mezcladas** (corazón + pulmón) que hasta
# ahora no se usaban. Sirven para responder la pregunta clínicamente relevante:
# *¿el clasificador sobrevive cuando el otro órgano suena encima?*
#
# **Qué NO son**: muestras nuevas. Verifiqué por hash que las componentes de las
# mezclas salen de las mismas 50 grabaciones limpias (20 son bit-idénticas a un
# archivo de HS/), así que usarlas para agrandar el n sería data leakage puro.
#
# **Integridad del metadato** (verificado sobre las 53 componentes cardíacas y 57
# pulmonares que matchean exactamente por hash con un archivo limpio):
#  - el **tipo de sonido coincide siempre** (53/53 y 57/57) -> la etiqueta es confiable;
#  - el **género NO** (32/53 y 48/57) -> la columna Gender de Mix.csv no identifica
#    la grabación fuente, así que acá se agrupa por TIPO, no por género_tipo.
#
# **Protocolo (tipo no visto)**: para cada tipo de sonido se entrena con las
# grabaciones limpias que NO son de ese tipo y se predice ese tipo. Ningún modelo
# evalúa una patología que vio en entrenamiento. Se aplica dos veces con idéntico
# entrenamiento y folds, cambiando solo la señal de test: limpia vs. mezclada.
# La diferencia es la degradación atribuible al sonido superpuesto.
# Los normales son un único tipo (dejarlos afuera eliminaría la clase negativa),
# así que se parten por género: `N_F` / `N_M`.
#
# **Dos variantes de features**: con la ganancia original y con la señal
# normalizada a pico unitario. Las mezclas están mucho más "fuertes" que las
# grabaciones limpias (rms_mean está +10 desvíos por encima), así que sin
# normalizar el modelo enfrenta un desplazamiento de dominio además del ruido.

# %%
import sys
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))
from src.evaluation import RANDOM_STATE, bootstrap_auc_ci, make_pipeline
from src.features import features_from_file, load_metadata, normalize_lung_id
from src.paths import AUDIO, DATA_DIR, OUT_DIR

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_STATE)

BALANCE = "smote"
DROP_COLS = ["file_id", "label", "label_binary"]
VARIANTES = {"ganancia original": False, "pico normalizado": True}


# %%
# ---------- Tablas de features (cacheadas: extraer 195 .wav tarda) ----------
def _suffix(normalize):
    return "_peaknorm" if normalize else ""


def clean_table(tag, normalize):
    """Features de las 50 grabaciones limpias de HS o LS."""
    cache = OUT_DIR / f"{tag}_features{_suffix(normalize)}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    hs, ls = load_metadata()
    meta, id_col, type_col = ((hs, "Heart Sound ID", "Heart Sound Type") if tag == "hs"
                              else (ls, "Lung Sound ID", "Lung Sound Type"))
    df = pd.DataFrame([features_from_file(AUDIO[tag.upper()] / f"{i}.wav", normalize=normalize)
                       for i in meta[id_col]])
    df["file_id"] = meta[id_col].values
    df["label"] = meta[type_col].values
    df["label_binary"] = meta["binary_label"].values
    df.to_csv(cache, index=False)
    return df


def mix_table(normalize):
    """Features de los 145 M####.wav con las etiquetas binarias de ambas tareas."""
    cache = OUT_DIR / f"mix_features{_suffix(normalize)}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    meta = pd.read_csv(DATA_DIR / "Mix.csv")
    df = pd.DataFrame([features_from_file(AUDIO["Mix"] / f"{i}.wav", normalize=normalize)
                       for i in meta["Mixed Sound ID"]])
    df["file_id"] = meta["Mixed Sound ID"].values
    df["hs_type"], df["ls_type"] = meta["Heart Sound Type"].values, meta["Lung Sound Type"].values
    df["gender"] = meta["Gender"].values
    df["hs_label"] = (df.hs_type != "Normal").astype(int)
    df["ls_label"] = (df.ls_type != "Normal").astype(int)
    df.to_csv(cache, index=False)
    return df


m0 = mix_table(False)
print(f"Mezclas: {len(m0)} grabaciones | HS patológico={int(m0.hs_label.sum())}, "
      f"normal={int((m0.hs_label==0).sum())} | LS patológico={int(m0.ls_label.sum())}, "
      f"normal={int((m0.ls_label==0).sum())}")
print("Desplazamiento de dominio limpio -> mezcla (en desvíos de las limpias):")
c0 = clean_table("hs", False)
for f in ["rms_mean", "mfcc_1_mean", "spectral_centroid_mean", "zcr_mean"]:
    print(f"  {f:24s} {(m0[f].mean() - c0[f].mean()) / c0[f].std():+.2f}")


# %%
# ---------- Grupos por tipo de sonido ----------
def type_groups_clean(file_ids):
    """Tipo de sonido de cada grabación limpia; los normales se parten por género."""
    parts = pd.Series(file_ids).astype(str).str.split("_")
    tipo, gen = parts.str[1], parts.str[0]
    return np.where(tipo == "N", "N_" + gen, tipo)


def type_abbrev(meta_csv, type_col, id_col, fix_ids=False):
    """Mapa 'nombre completo del tipo' -> abreviatura usada en los file_id."""
    d = pd.read_csv(DATA_DIR / meta_csv)
    ids = d[id_col].map(normalize_lung_id) if fix_ids else d[id_col]
    return dict(zip(d[type_col], ids.str.split("_").str[1]))


def type_groups_mix(types, genders, abbrev):
    """Mismo criterio de grupos para las mezclas, desde el metadato de Mix.csv."""
    t = pd.Series(types).map(abbrev).values
    return np.where(t == "N", "N_" + pd.Series(genders).values, t)


# %%
# ---------- Protocolo: entrenar sin el tipo, predecir ese tipo ----------
def leave_one_type_out(make_est, X_tr, y_tr, g_tr, X_te, y_te, g_te):
    """Probabilidad out-of-fold de cada muestra de test, con un modelo que nunca
    vio su tipo de sonido. NaN en los tipos que dejarían el entrenamiento con
    una sola clase."""
    p = np.full(len(y_te), np.nan)
    for t in np.unique(g_te):
        tr, te = g_tr != t, g_te == t
        if len(np.unique(y_tr[tr])) < 2:
            continue
        pipe = make_pipeline(make_est(), BALANCE)
        pipe.fit(X_tr[tr], y_tr[tr])
        p[te] = pipe.predict_proba(X_te[te])[:, 1]
    return p


def evaluate_task(tarea, variante, normalize, tag, make_est, label_col, type_col, abbrev):
    """Desempeño sobre audio limpio vs. sobre las mezclas, con idéntico entrenamiento."""
    clean, mix = clean_table(tag, normalize), mix_table(normalize)
    cols = [c for c in clean.columns if c not in DROP_COLS]
    X_c, y_c = clean[cols].values, clean.label_binary.astype(int).values
    g_c = type_groups_clean(clean.file_id)
    X_m, y_m = mix[cols].values, mix[label_col].values     # mismas features, mismo orden
    g_m = type_groups_mix(mix[type_col], mix.gender, abbrev)

    escenarios = {
        "limpio (tipo no visto)": (leave_one_type_out(make_est, X_c, y_c, g_c, X_c, y_c, g_c), y_c),
        "mezcla (tipo no visto)": (leave_one_type_out(make_est, X_c, y_c, g_c, X_m, y_m, g_m), y_m),
    }
    # Cota superior: entrenado con TODAS las limpias (la fuente de cada mezcla
    # estuvo en el entrenamiento), evaluado sobre las mezclas.
    pipe = make_pipeline(make_est(), BALANCE).fit(X_c, y_c)
    escenarios["mezcla (entrenado con todo)"] = (pipe.predict_proba(X_m)[:, 1], y_m)

    rows, curvas = [], {}
    for nombre, (p, y) in escenarios.items():
        ok = ~np.isnan(p)
        lo, hi = bootstrap_auc_ci(y[ok], p[ok])
        tn, fp, fn, tp = confusion_matrix(y[ok], (p[ok] >= 0.5).astype(int), labels=[0, 1]).ravel()
        rows.append({"tarea": tarea, "variante": variante, "escenario": nombre, "n": int(ok.sum()),
                     "AUC": roc_auc_score(y[ok], p[ok]), "IC95_bajo": lo, "IC95_alto": hi,
                     "sensibilidad": tp / (tp + fn), "especificidad": tn / (tn + fp)})
        curvas[nombre] = (y[ok], p[ok])
    return pd.DataFrame(rows), curvas


# %%
# ---------- Ejecución ----------
TAREAS = {
    "HS (cardíaco)": dict(
        tag="hs", label_col="hs_label", type_col="hs_type",
        make_est=lambda: RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        abbrev=type_abbrev("HS.csv", "Heart Sound Type", "Heart Sound ID"),
    ),
    "LS (pulmonar)": dict(
        tag="ls", label_col="ls_label", type_col="ls_type",
        make_est=lambda: LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        abbrev=type_abbrev("LS.csv", "Lung Sound Type", "Lung Sound ID", fix_ids=True),
    ),
}

res, curvas_all = [], {}
for tarea, cfg in TAREAS.items():
    for variante, normalize in VARIANTES.items():
        df, curvas = evaluate_task(tarea, variante, normalize, **cfg)
        res.append(df)
        curvas_all[(tarea, variante)] = curvas

res = pd.concat(res, ignore_index=True)
print("\n===== Robustez ante sonido superpuesto =====")
print(res.round(3).to_string(index=False))
res.to_csv(OUT_DIR / "results_robustez_mix.csv", index=False)


# %%
# ---------- Curvas ROC: limpio vs. mezcla, con y sin normalizar ----------
ESTILO = {"limpio (tipo no visto)": "-", "mezcla (tipo no visto)": "--"}
COLOR = {"ganancia original": "tab:red", "pico normalizado": "tab:blue"}

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, tarea in zip(axes, TAREAS):
    for (t, variante), curvas in curvas_all.items():
        if t != tarea:
            continue
        for nombre, ls in ESTILO.items():
            y, p = curvas[nombre]
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, ls, color=COLOR[variante], lw=1.6,
                    label=f"{nombre.split(' (')[0]} — {variante} — AUC={roc_auc_score(y, p):.2f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="1 - Especificidad", ylabel="Sensibilidad", title=tarea)
    ax.legend(loc="lower right", fontsize=7)
fig.suptitle("Degradación por sonido superpuesto (subset Mix)")
fig.tight_layout()
fig.savefig(OUT_DIR / "17_robustez_mix.png", dpi=150)
