"""Infraestructura de evaluación compartida por los notebooks de modelado.

Contiene lo que 02_modeling y 03_robustez_mix usan por igual: el pipeline
sin fugas (escalado -> balanceo -> modelo), el agrupamiento por caso del
maniquí y el intervalo de confianza bootstrap del AUC.
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
K_NEIGHBORS = 3   # tope de vecinos de SMOTE (la minoritaria por fold es muy chica)


class SafeSMOTE(SMOTE):
    """SMOTE que ajusta k_neighbors al tamaño real de la clase minoritaria.

    Con k fijo, un fold cuyo train queda con <= k muestras normales hace fallar
    a SMOTE ("n_neighbors <= n_samples_fit"). Pasa siempre en CV agrupada por
    caso: HS tiene solo 2 casos normales, así que dejar uno afuera deja 3
    normales en train.
    """

    def _fit_resample(self, X, y, **kw):
        n_min = int(np.min(np.bincount(y)))
        self.k_neighbors = max(1, min(K_NEIGHBORS, n_min - 1))
        return super()._fit_resample(X, y, **kw)


def make_pipeline(estimator, balance, force_smote=False):
    """Pipeline escalado -> (SMOTE opcional) -> modelo.

    Todo se ajusta SOLO con el fold de entrenamiento: sin data leakage.
    `balance` es "smote" o "class_weight" (en este último caso el modelo ya
    viene con class_weight='balanced' y no se sobremuestrea).
    """
    steps = [("scaler", StandardScaler())]
    if balance == "smote" or force_smote:
        steps.append(("smote", SafeSMOTE(random_state=RANDOM_STATE)))
    steps.append(("clf", estimator))
    return ImbPipeline(steps)


def case_groups(file_ids):
    """Caso del maniquí al que pertenece cada grabación: 'genero_tipo'.

    Las 50 grabaciones NO son 50 sujetos: son género x patología x localización
    sobre un mismo maniquí (HS: 19 casos, LS: 12). Dos localizaciones del mismo
    caso no son independientes, así que deben caer siempre del mismo lado de la
    partición. Se usan como `groups` de StratifiedGroupKFold/LeaveOneGroupOut.
    """
    s = pd.Series(file_ids).astype(str).str.split("_")
    return (s.str[0] + "_" + s.str[1]).values


def bootstrap_auc_ci(y, proba, n_boot=2000, alpha=0.05, random_state=RANDOM_STATE):
    """IC percentil bootstrap del AUC sobre probabilidades out-of-fold.

    Con n=50 el AUC puntual no dice nada sin intervalo. Se remuestrea a nivel
    de muestra, descartando réplicas que quedan con una sola clase.
    """
    y = np.asarray(y)
    proba = np.asarray(proba)
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(y))
    aucs = []
    while len(aucs) < n_boot:
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) == 2:
            aucs.append(roc_auc_score(y[s], proba[s]))
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
