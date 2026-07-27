# %% [markdown]
# # 02 — Modelado: comparación de modelos clásicos y de ensemble
#
# Clasificación binaria (Normal vs. Patológico) de sonidos cardíacos (HS) y
# pulmonares (LS) sobre la matriz de features del EDA.
#
# Principios de diseño (críticos con n pequeño):
#  - TODO el preprocesamiento (escalado + balanceo) va DENTRO de un Pipeline que
#    se ajusta solo con el fold de entrenamiento -> sin data leakage.
#  - Validación cruzada estratificada REPETIDA -> media + desvío (intervalos).
#  - Métricas clínicas: sensibilidad, especificidad, F1-macro, AUC.
#  - Se comparan modelos individuales y de ensemble, más un MLP chico.
#  - Las DOS estrategias de balanceo (SMOTE y class_weight) se evalúan y se
#    guardan en la misma corrida, en un único CSV con la columna `balanceo`.

# %%
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, make_scorer, confusion_matrix

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Modelos de boosting (opcionales: si no están instalados, se omiten)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


# %%
# ---------- Configuración ----------
FEAT_DIR = Path("../outputs")          # dónde quedaron hs_features.csv / ls_features.csv
OUT_DIR = Path("../outputs"); OUT_DIR.mkdir(exist_ok=True)
BALANCE_MODES = ["smote", "class_weight"]   # ambas estrategias en una sola corrida
N_SPLITS, N_REPEATS = 5, 10            # 5-fold x 10 repeticiones = 50 evaluaciones
DROP_COLS = ["file_id", "label", "label_binary"]  # columnas que no son features
TARGET = "label_binary"               # 0 = Normal, 1 = Patológico


# %%
def load_xy(csv_name):
    """Carga un CSV de features y devuelve X (features) e y (etiqueta binaria)."""
    path = FEAT_DIR / csv_name
    if not path.exists():                       # fallback a ../data por si acaso
        path = Path("../data") / csv_name
    df = pd.read_csv(path)
    y = df[TARGET].astype(int).values
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    X = X.select_dtypes(include=[np.number])    # solo numéricas
    return X.values, y, list(X.columns)


# %%
def make_pipeline(estimator, balance, force_smote=False):
    """Arma un Pipeline: escalado -> (SMOTE opcional) -> modelo.
    El escalado y el SMOTE se ajustan SOLO con el train de cada fold.
    """
    steps = [("scaler", StandardScaler())]
    if balance == "smote" or force_smote:
        # k_neighbors chico: la clase minoritaria en cada fold de train es muy pequeña
        steps.append(("smote", SMOTE(k_neighbors=3, random_state=RANDOM_STATE)))
    steps.append(("clf", estimator))
    return ImbPipeline(steps)


def build_models(balance):
    """Devuelve un dict {nombre: pipeline} para la estrategia de balanceo dada.
    Bajo 'class_weight' los modelos que lo soportan usan class_weight='balanced';
    el MLP no lo soporta, así que en ese modo se balancea con SMOTE (excepción
    documentada).
    """
    cw = "balanced" if balance == "class_weight" else None
    models = {}
    models["LogReg"] = make_pipeline(
        LogisticRegression(max_iter=2000, class_weight=cw, random_state=RANDOM_STATE), balance)
    models["SVM-Linear"] = make_pipeline(
        SVC(kernel="linear", probability=True, class_weight=cw, random_state=RANDOM_STATE), balance)
    models["SVM-RBF"] = make_pipeline(
        SVC(kernel="rbf", probability=True, class_weight=cw, random_state=RANDOM_STATE), balance)
    models["DecisionTree"] = make_pipeline(
        DecisionTreeClassifier(class_weight=cw, max_depth=4, random_state=RANDOM_STATE), balance)
    models["RandomForest"] = make_pipeline(
        RandomForestClassifier(n_estimators=300, class_weight=cw, max_depth=None,
                               random_state=RANDOM_STATE), balance)
    if HAS_XGB:
        models["XGBoost"] = make_pipeline(
            XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                          subsample=0.9, eval_metric="logloss", random_state=RANDOM_STATE), balance)
    if HAS_LGBM:
        models["LightGBM"] = make_pipeline(
            LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                           class_weight=cw, random_state=RANDOM_STATE, verbose=-1), balance)
    # Ensemble por votación (soft): combina 3 modelos diversos
    voting = VotingClassifier(estimators=[
        ("lr", LogisticRegression(max_iter=2000, class_weight=cw, random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(n_estimators=300, class_weight=cw, random_state=RANDOM_STATE)),
        ("svm", SVC(kernel="rbf", probability=True, class_weight=cw, random_state=RANDOM_STATE)),
    ], voting="soft")
    models["Voting"] = make_pipeline(voting, balance)
    # MLP chico (perceptrón multicapa) sobre la matriz tabular.
    # No soporta class_weight -> en modo class_weight se balancea con SMOTE.
    models["MLP"] = make_pipeline(
        MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-2, early_stopping=True,
                      max_iter=1000, random_state=RANDOM_STATE),
        balance, force_smote=(balance == "class_weight"))
    return models


# %%
# Métricas: sensibilidad = recall clase 1 (patológico); especificidad = recall clase 0 (normal)
SCORERS = {
    "sensibilidad": make_scorer(recall_score, pos_label=1, zero_division=0),
    "especificidad": make_scorer(recall_score, pos_label=0, zero_division=0),
    "f1_macro": "f1_macro",
    "AUC": "roc_auc",
    "balanced_acc": "balanced_accuracy",
}


def evaluate(X, y, balance, label=""):
    """Evalúa todos los modelos con CV estratificada repetida, para UNA estrategia
    de balanceo. Devuelve un DataFrame con media +/- desvío por métrica.
    """
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    rows = []
    for name, pipe in build_models(balance).items():
        res = cross_validate(pipe, X, y, cv=cv, scoring=SCORERS, n_jobs=-1, error_score="raise")
        row = {"balanceo": balance, "modelo": name}
        for m in SCORERS:
            vals = res[f"test_{m}"]
            row[m] = f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"
            row[f"_{m}_mean"] = np.mean(vals)   # numérico para ordenar
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("_AUC_mean", ascending=False).reset_index(drop=True)
    show = ["modelo"] + list(SCORERS.keys())
    print(f"\n===== {label} (balanceo={balance}, {N_SPLITS}x{N_REPEATS} CV) =====")
    print(df[show].to_string(index=False))
    return df


def evaluate_all(X, y, label=""):
    """Corre la evaluación para las dos estrategias de balanceo y las apila."""
    return pd.concat([evaluate(X, y, b, label) for b in BALANCE_MODES], ignore_index=True)


def confusion_for_best(X, y, best_name, balance, label=""):
    """Matriz de confusión agregada (out-of-fold) del mejor modelo."""
    pipe = build_models(balance)[best_name]
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
    cm = confusion_matrix(y, y_pred)
    print(f"\nMatriz de confusión OOF — {label} — {best_name} (balanceo={balance})")
    print(pd.DataFrame(cm, index=["real:Normal", "real:Patol"],
                       columns=["pred:Normal", "pred:Patol"]).to_string())
    return cm


def save_results(df, csv_name):
    """Guarda el DataFrame de resultados (ambos balanceos) sin las columnas auxiliares."""
    out = df.drop(columns=[c for c in df.columns if c.startswith("_")])
    out.to_csv(OUT_DIR / csv_name, index=False)
    return out


def best_per_balance(df):
    """Mejor modelo por AUC para cada estrategia de balanceo -> {balanceo: modelo}."""
    idx = df.groupby("balanceo")["_AUC_mean"].idxmax()
    return dict(zip(df.loc[idx, "balanceo"], df.loc[idx, "modelo"]))


# %%
# ---------- Ejecución: Sonidos cardíacos (HS) ----------
X_hs, y_hs, feat_names = load_xy("hs_features.csv")
print(f"HS: X={X_hs.shape}, positivos(patológico)={int(y_hs.sum())}, negativos(normal)={int((y_hs==0).sum())}")
res_hs = evaluate_all(X_hs, y_hs, label="Sonidos cardíacos (HS)")
best_hs = best_per_balance(res_hs)
for b, m in best_hs.items():
    confusion_for_best(X_hs, y_hs, m, b, label="HS")
save_results(res_hs, "results_hs.csv")


# %%
# ---------- Ejecución: Sonidos pulmonares (LS) ----------
X_ls, y_ls, _ = load_xy("ls_features.csv")
print(f"LS: X={X_ls.shape}, positivos(patológico)={int(y_ls.sum())}, negativos(normal)={int((y_ls==0).sum())}")
res_ls = evaluate_all(X_ls, y_ls, label="Sonidos pulmonares (LS)")
best_ls = best_per_balance(res_ls)
for b, m in best_ls.items():
    confusion_for_best(X_ls, y_ls, m, b, label="LS")
save_results(res_ls, "results_ls.csv")


# %%
# ---------- LOOCV + ROC agrupada (pooled) — HS ----------
# Con solo 9 casos normales, la especificidad por fold es muy inestable (cada fold
# de test tiene 1-2 normales). LOOCV deja UNA muestra afuera por vez y se juntan
# TODAS las probabilidades out-of-fold en una única curva ROC (pooled): una
# estimación de AUC con menos varianza de partición que el promedio por fold.
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score


def loocv_probas(X, y, balance):
    """Probabilidad out-of-fold (LOOCV) de la clase patológica, por modelo."""
    loo = LeaveOneOut()
    return pd.DataFrame({
        name: cross_val_predict(pipe, X, y, cv=loo, method="predict_proba", n_jobs=-1)[:, 1]
        for name, pipe in build_models(balance).items()
    })


rows, probas = [], {}
for b in BALANCE_MODES:
    P = loocv_probas(X_hs, y_hs, b)
    for name in P.columns:
        probas[f"{b}|{name}"] = P[name].values
        rows.append({"balanceo": b, "modelo": name,
                     "AUC_pooled": roc_auc_score(y_hs, P[name]),
                     "AP_pooled": average_precision_score(y_hs, P[name])})

loocv_hs = pd.DataFrame(rows).sort_values("AUC_pooled", ascending=False).reset_index(drop=True)
print(f"\n===== HS — LOOCV, ROC agrupada (n={len(y_hs)}) =====")
print(loocv_hs.to_string(index=False))
loocv_hs.to_csv(OUT_DIR / "results_hs_loocv.csv", index=False)
pd.DataFrame(probas).assign(y_true=y_hs).to_csv(OUT_DIR / "hs_loocv_probas.csv", index=False)

# Curvas ROC agrupadas de los 3 mejores (modelo + estrategia de balanceo)
fig, ax = plt.subplots(figsize=(5.5, 5))
for _, r in loocv_hs.head(3).iterrows():
    fpr, tpr, _ = roc_curve(y_hs, probas[f"{r.balanceo}|{r.modelo}"])
    ax.plot(fpr, tpr, lw=1.8, label=f"{r.modelo} ({r.balanceo}) — AUC={r.AUC_pooled:.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set(xlabel="1 - Especificidad (FPR)", ylabel="Sensibilidad (TPR)",
       title="HS — ROC agrupada out-of-fold (LOOCV)")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "15_roc_loocv_hs.png", dpi=150)


# %%
# ---------- Importancia de features (interpretación, no es estimación de desempeño) ----------
def top_importances(X, y, feat_names, k=15):
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=RANDOM_STATE)
    rf.fit(StandardScaler().fit_transform(X), y)
    imp = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)
    return imp.head(k)

print("\nTop 15 features (HS) por importancia RF:")
print(top_importances(X_hs, y_hs, feat_names).to_string())


# %% [markdown]
# ## Ajuste de hiperparámetros (patrón de CV anidada — opcional)
#
# Con n pequeño, para elegir hiperparámetros SIN contaminar la estimación de
# desempeño se usa validación cruzada anidada: un CV interno tunea y un CV
# externo estima. Ejemplo para Random Forest (descomentar para usar):
#
# ```python
# from sklearn.model_selection import GridSearchCV, cross_val_score
# grid = {"clf__max_depth": [3, 5, None], "clf__n_estimators": [200, 400]}
# inner = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
# outer = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
# search = GridSearchCV(make_pipeline(RandomForestClassifier(class_weight="balanced",
#              random_state=RANDOM_STATE), "class_weight"), grid, cv=inner,
#              scoring="roc_auc", n_jobs=-1)
# nested_auc = cross_val_score(search, X_hs, y_hs, cv=outer, scoring="roc_auc")
# print("AUC anidada:", nested_auc.mean(), "+/-", nested_auc.std())
# ```
