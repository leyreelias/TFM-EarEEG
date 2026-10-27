"""
Clasificacion REST vs OpenMATB con las 11 features originales por canal.
REST_1/2/3 = 0; OPENMATB = 1. No usa epochs N-back, ni para normalizar.
Solo grabaciones que contienen OpenMATB (actualmente 13, participant_02).
Mismos modelos, grids, CV agrupada y control de convergencia que 05_3back_vs_rest.
Por defecto NORMALIZE_PER_GROUP=False: imputacion, seleccion y escalado solo TRAIN.
Si se activa, la normalizacion por grabacion es un analisis offline adicional
que usa los epochs REST/OpenMATB de la propia grabacion de test.
Resultados separados; conserva combinaciones completas al volver a ejecutar.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data_features"
OUTPUT_DIR = ROOT / "results" / "ml_rest_vs_openmatb"
MODEL_DIR = OUTPUT_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

CLASSES = {"REST_1": 0, "REST_2": 0, "REST_3": 0, "OPENMATB": 1}

# Unidad que NUNCA se reparte entre train y test:
#   "recording"   -> session + participant (como antes)
#   "participant" -> solo participant (más estricto; usar si el mismo
#                    participante aparece en varias sesiones)
GROUP_BY = "recording"

N_OUTER = 5          # folds externos (evaluación)
N_INNER = 3          # folds internos (GridSearch)

# True: z-score de cada feature dentro de cada grabación (no usa etiquetas).
# Ojo: en uso online requeriría una ventana de calibración por sesión.
NORMALIZE_PER_GROUP = False

# True: reentrena con TODOS los datos y guarda el modelo final por método/modelo
SAVE_FINAL_MODELS = True

# k de SelectKBest; se prueba en el grid, igual para todos los métodos
K_GRID = [30, 100]


# ============================================================
# MODELOS + GRIDS
# ============================================================

MODELS = {
    "LogisticRegression": {
        "classifier": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "params": {
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__penalty": ["l1", "l2"],
        },
    },
    "LDA_shrinkage": {
        "classifier": LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
        ),
        "params": {},
    },
    "SVM": {
        # Evitar ajustes sin limite; los no convergidos no se puntuan.
        "classifier": SVC(class_weight="balanced", cache_size=512, max_iter=200000),
        "params": [
            {
                "classifier__kernel": ["linear"],
                "classifier__C": [0.1, 1],
            },
            {
                "classifier__kernel": ["rbf"],
                "classifier__C": [1, 10],
                "classifier__gamma": ["scale", 0.01],
            },
        ],
    },
    "RandomForest": {
        "classifier": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,  # el paralelismo lo hace GridSearchCV
        ),
        "params": {
            "classifier__max_depth": [None, 10],
            "classifier__min_samples_leaf": [1, 5],
            "classifier__max_features": ["sqrt"],
        },
    },
}


# ============================================================
# FUNCIONES
# ============================================================

def add_k_to_grid(params):
    """Añade selector__k a un grid (dict o lista de dicts)."""
    if isinstance(params, list):
        return [{**p, "selector__k": K_GRID} for p in params]
    return {**params, "selector__k": K_GRID}


def load_method(path):
    """Carga un CSV de features, añade 'group', transforma y normaliza."""
    df = pd.read_csv(path, low_memory=False)

    if GROUP_BY == "participant":
        df["group"] = df["participant"].astype(str)
    else:
        df["group"] = df["session"].astype(str) + "__" + df["participant"].astype(str)

    # Elegir la cohorte antes de filtrar: no incluir grabaciones sin OpenMATB.
    groups_with_matb = set(df.loc[df["condition"].eq("OPENMATB"), "group"])
    df = df[df["group"].isin(groups_with_matb) & df["condition"].isin(CLASSES)].copy()
    if df.empty:
        raise ValueError(f"No hay datos REST/OpenMATB: {path}")
    if df.duplicated(["session", "participant", "epoch"]).any():
        raise ValueError(f"Epochs duplicados: {path}")

    feature_cols = [c for c in df.columns if "__" in c]

    # Ratios: muy asimétricos y positivos -> log
    ratio_cols = [c for c in feature_cols if c.endswith(("__theta_alpha_ratio", "__theta_beta_ratio"))]
    if ratio_cols:
        df[ratio_cols] = np.log(df[ratio_cols].clip(lower=1e-12))

    # Potencias relativas en (0, 1) -> logit
    rel_cols = [c for c in feature_cols if "__relative_" in c]
    if rel_cols:
        p = df[rel_cols].clip(1e-6, 1 - 1e-6)
        df[rel_cols] = np.log(p / (1 - p))

    # Opcional offline: solo los epochs REST/OpenMATB seleccionados arriba.
    if NORMALIZE_PER_GROUP:
        g = df.groupby("group")[feature_cols]
        mu = g.transform("mean")
        sd = g.transform("std").replace(0, np.nan)
        df[feature_cols] = (df[feature_cols] - mu) / sd

    df = df[df["condition"].isin(CLASSES)].reset_index(drop=True)
    df["y"] = df["condition"].map(CLASSES)

    return df, feature_cols


def make_outer_folds(df):
    """Folds externos a nivel de grupo; se calculan una vez y se reutilizan."""
    cv = StratifiedGroupKFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE)
    groups = df["group"].to_numpy()
    return [
        set(groups[test_idx])
        for _, test_idx in cv.split(df, df["y"], groups)
    ]


def build_search(model_config):
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("selector", SelectKBest(score_func=f_classif)),
        ("scaler", StandardScaler()),
        ("classifier", clone(model_config["classifier"])),
    ])
    inner_cv = StratifiedGroupKFold(n_splits=N_INNER, shuffle=True, random_state=RANDOM_STATE)
    return GridSearchCV(
        estimator=pipeline,
        param_grid=add_k_to_grid(model_config["params"]),
        scoring=score_converged_model,
        cv=inner_cv,
        n_jobs=2,
        verbose=2,
        refit=False,  # El reajuste se valida explicitamente abajo.
    )


def score_converged_model(model, X, y):
    """Descartar candidatos SVM que alcanzan el limite sin converger."""
    classifier = model.named_steps["classifier"]
    if isinstance(classifier, SVC) and classifier.fit_status_ != 0:
        return np.nan
    return balanced_accuracy_score(y, model.predict(X))


def check_convergence(search, X_train, y_train):
    """Elegir por CV interna la primera configuracion que converge en TRAIN.

    Nunca se consultan X_test ni y_test. Si falla el reajuste de la mejor
    configuracion, se prueba la siguiente del mismo grid, sin repetir la CV.
    """
    scores = np.asarray(search.cv_results_["mean_test_score"], dtype=float)
    n_invalid = int((~np.isfinite(scores)).sum())
    if n_invalid:
        print(f"  Candidatos sin puntuacion valida descartados: {n_invalid}", flush=True)
    candidates = np.flatnonzero(np.isfinite(scores))
    candidates = candidates[np.argsort(-scores[candidates], kind="stable")]
    search.refit_rejected_ = []
    for index in candidates:
        params = search.cv_results_["params"][index]
        model = clone(search.estimator).set_params(**params)
        model.fit(X_train, y_train)
        classifier = model.named_steps["classifier"]
        if isinstance(classifier, SVC) and classifier.fit_status_ != 0:
            search.refit_rejected_.append(params)
            print(f"  No converge en TRAIN; se prueba el siguiente candidato: {params}", flush=True)
            continue
        search.best_index_ = int(index)
        search.best_params_ = params
        search.best_score_ = float(scores[index])
        search.best_estimator_ = model
        return
    raise RuntimeError("Ninguna configuracion converge en TRAIN. No se evalua el test.")


def get_scores(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def compute_metrics(y_true, y_pred, y_score):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan,
    }


def save_progress(fold_rows, summary_rows, confusion_rows):
    pd.DataFrame(fold_rows).to_csv(OUTPUT_DIR / "results_per_fold.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "results.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(OUTPUT_DIR / "confusion_matrices.csv", index=False)


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    # No mezclar resultados al reanudar si se cambia la configuracion.
    config = {"classes": CLASSES, "normalize_per_group": NORMALIZE_PER_GROUP,
              "group_by": GROUP_BY, "seed": RANDOM_STATE,
              "outer": N_OUTER, "inner": N_INNER, "k_grid": K_GRID,
              "grids": {name: value["params"] for name, value in MODELS.items()}}
    config_path = OUTPUT_DIR / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Configuracion distinta: usar otra OUTPUT_DIR para no mezclar resultados.")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    files = sorted(INPUT_DIR.glob("features_*.csv"))

    # Folds externos definidos una sola vez con all_mean
    # y reutilizados en los 5 métodos de rereferencia.
    reference_df, _ = load_method(INPUT_DIR / "features_all_mean.csv")
    outer_folds = make_outer_folds(reference_df)
    pd.DataFrame([{"fold": fold, "test_group": group}
                  for fold, groups in enumerate(outer_folds, 1)
                  for group in sorted(groups)]).to_csv(OUTPUT_DIR / "test_groups.csv", index=False)

    print("=" * 80)
    print(f"CV EXTERNA: {N_OUTER} folds, agrupando por '{GROUP_BY}'")
    print("=" * 80)
    for i, test_groups in enumerate(outer_folds, 1):
        print(f"  Fold {i}: {len(test_groups)} grupos en test")

    # Reutilizar resultados completos si se vuelve a lanzar el mismo experimento.
    def read_rows(name):
        path = OUTPUT_DIR / name
        return pd.read_csv(path).to_dict("records") if path.exists() and path.stat().st_size > 2 else []

    fold_rows = read_rows("results_per_fold.csv")
    summary_rows = read_rows("results.csv")
    confusion_rows = read_rows("confusion_matrices.csv")

    for path in files:
        method = path.stem.replace("features_", "")

        print("\n" + "=" * 80)
        print(method)
        print("=" * 80)

        df, feature_cols = load_method(path)
        if set(df["group"]) != set(reference_df["group"]):
            raise ValueError(f"Grabaciones distintas en {method}; revisar datos")
        X_all = df[feature_cols]
        y_all = df["y"].to_numpy()
        groups_all = df["group"].to_numpy()

        print(f"Epochs: {len(df)} | REST: {(y_all == 0).sum()} | "
              f"OPENMATB: {(y_all == 1).sum()} | features: {len(feature_cols)}")

        for model_name, model_config in MODELS.items():

            def same_model(row):
                return row["method"] == method and row["model"] == model_name

            model_path = MODEL_DIR / f"{method}__{model_name}.joblib"
            if (any(same_model(r) for r in summary_rows)
                    and sum(same_model(r) for r in fold_rows) == N_OUTER
                    and any(same_model(r) for r in confusion_rows)
                    and (not SAVE_FINAL_MODELS or model_path.exists())):
                print(f"\n{model_name}: ya completado; se conserva.", flush=True)
                continue
            # Un modelo interrumpido se repite, sin duplicar sus filas.
            fold_rows = [r for r in fold_rows if not same_model(r)]
            summary_rows = [r for r in summary_rows if not same_model(r)]
            confusion_rows = [r for r in confusion_rows if not same_model(r)]

            print(f"\n--- {model_name} ---")

            oof_true, oof_pred = [], []
            model_fold_rows = []

            for fold, test_groups in enumerate(outer_folds, 1):

                test_mask = np.isin(groups_all, list(test_groups))
                train_mask = ~test_mask

                print(f"  Iniciando fold {fold}/{N_OUTER}: GridSearch...", flush=True)
                search = build_search(model_config)
                search.fit(
                    X_all[train_mask],
                    y_all[train_mask],
                    groups=groups_all[train_mask],
                )
                check_convergence(search, X_all[train_mask], y_all[train_mask])

                model = search.best_estimator_
                X_test, y_test = X_all[test_mask], y_all[test_mask]
                y_pred = model.predict(X_test)
                y_score = get_scores(model, X_test)
                metrics = compute_metrics(y_test, y_pred, y_score)
                # Predicciones externas para revisar despues cada grabacion.
                prediction_dir = OUTPUT_DIR / "predictions"
                prediction_dir.mkdir(exist_ok=True)
                predictions = df.loc[test_mask, ["session", "participant", "epoch", "condition"]].copy()
                predictions["y_true"] = y_test
                predictions["y_pred"] = y_pred
                predictions["score"] = y_score
                predictions.to_csv(prediction_dir / f"{method}__{model_name}__fold{fold}.csv", index=False)

                oof_true.append(y_test)
                oof_pred.append(y_pred)

                row = {
                    "method": method,
                    "model": model_name,
                    "fold": fold,
                    "n_test": int(test_mask.sum()),
                    "inner_cv_balanced_accuracy": search.best_score_,
                    "n_refit_nonconverged": len(search.refit_rejected_),
                    **metrics,
                    "best_params": json.dumps(search.best_params_, sort_keys=True),
                }
                model_fold_rows.append(row)
                fold_rows.append(row)
                save_progress(fold_rows, summary_rows, confusion_rows)

                print(f"  Fold {fold}: bal_acc={metrics['balanced_accuracy']:.3f} "
                      f"auc={metrics['roc_auc']:.3f} "
                      f"(inner CV {search.best_score_:.3f})")

            # --- Resumen del modelo (media ± std entre folds) ---
            fold_df = pd.DataFrame(model_fold_rows)
            summary = {"method": method, "model": model_name}
            for m in ["accuracy", "balanced_accuracy", "f1", "roc_auc"]:
                summary[f"{m}_mean"] = fold_df[m].mean()
                summary[f"{m}_std"] = fold_df[m].std()
            summary["inner_cv_balanced_accuracy_mean"] = fold_df["inner_cv_balanced_accuracy"].mean()
            summary_rows.append(summary)

            cm = confusion_matrix(
                np.concatenate(oof_true), np.concatenate(oof_pred), labels=[0, 1]
            )
            confusion_rows.append({
                "method": method,
                "model": model_name,
                "true_rest_pred_rest": cm[0, 0],
                "true_rest_pred_openmatb": cm[0, 1],
                "true_openmatb_pred_rest": cm[1, 0],
                "true_openmatb_pred_openmatb": cm[1, 1],
            })

            print(f"  >> bal_acc {summary['balanced_accuracy_mean']:.3f} "
                  f"± {summary['balanced_accuracy_std']:.3f} | "
                  f"AUC {summary['roc_auc_mean']:.3f} ± {summary['roc_auc_std']:.3f}")

            # --- Modelo final con todos los datos ---
            if SAVE_FINAL_MODELS:
                print("  GridSearch final con todos los datos (sin nueva evaluacion)...", flush=True)
                final_search = build_search(model_config)
                final_search.fit(X_all, y_all, groups=groups_all)
                check_convergence(final_search, X_all, y_all)
                joblib.dump(
                    {
                        "model": final_search.best_estimator_,
                        "feature_cols": feature_cols,
                        "best_params": final_search.best_params_,
                        "normalize_per_group": NORMALIZE_PER_GROUP,
                        "classes": CLASSES,
                        "refit_nonconverged_params": final_search.refit_rejected_,
                    },
                    MODEL_DIR / f"{method}__{model_name}.joblib",
                )

            # Guardar progreso por si se interrumpe
            save_progress(fold_rows, summary_rows, confusion_rows)

    # ========================================================
    # RESULTADOS FINALES
    # ========================================================

    results_df = (
        pd.DataFrame(summary_rows)
        .sort_values("balanced_accuracy_mean", ascending=False)
    )
    results_df.to_csv(OUTPUT_DIR / "results.csv", index=False)

    print("\n" + "=" * 80)
    print("RESULTADOS FINALES (media ± std entre folds externos)")
    print("=" * 80)

    table = results_df[["method", "model"]].copy()
    for m in ["balanced_accuracy", "roc_auc", "f1", "accuracy"]:
        table[m] = results_df.apply(
            lambda r: f"{r[m + '_mean']:.3f} ± {r[m + '_std']:.3f}", axis=1
        )
    print(table.to_string(index=False))

    print(f"\nResultados guardados en: {OUTPUT_DIR}")  
