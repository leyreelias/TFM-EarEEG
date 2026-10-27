from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "data_features"
OUTPUT_DIR = ROOT / "data_features_normalized"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_CONDITION = "eeg_calibration_EO"

# Número mínimo de epochs válidos de baseline para
# considerar fiable el cálculo de esa feature.
MIN_BASELINE_VALID = 5

EPS = 1e-12


# Las potencias logarítmicas se expresarán como cambio
# respecto a la media basal (dB).
LOG_FEATURES = {
    "log_theta",
    "log_alpha",
    "log_beta",
}


# ============================================================
# FUNCIONES
# ============================================================

def feature_type(column_name):
    """
    Devuelve el nombre de la feature a partir de:
    L1__log_alpha -> log_alpha
    """
    return column_name.split("__")[-1]


# ============================================================
# EJECUCIÓN
# ============================================================

summary_rows = []


for input_path in sorted(INPUT_DIR.glob("features_*.csv")):

    method = input_path.stem.replace("features_", "")

    print("\n" + "=" * 70)
    print(method)
    print("=" * 70)

    df = pd.read_csv(input_path)

    feature_cols = [
        c for c in df.columns
        if "__" in c
    ]

    # Copiamos metadata y datos originales.
    # Añadiremos columnas *_norm.
    result = df.copy()

    # Crear columnas normalizadas inicialmente como NaN
    norm_cols = {}

    for col in feature_cols:
        norm_cols[col + "_norm"] = np.full(
            len(df),
            np.nan,
            dtype=float
        )

    # --------------------------------------------------------
    # Normalización independiente por sesión + participante
    # --------------------------------------------------------

    groups = df.groupby(
        ["session", "participant"],
        sort=False
    )

    for (session, participant), idx in groups.groups.items():

        idx = np.asarray(list(idx))

        group = df.loc[idx]

        baseline = group[
            group["condition"] == BASELINE_CONDITION
        ]

        if len(baseline) == 0:

            print(
                f"WARNING {session}/{participant}: "
                "sin baseline EO"
            )

            continue

        n_normalized = 0
        n_failed = 0

        for col in feature_cols:

            baseline_values = pd.to_numeric(
                baseline[col],
                errors="coerce"
            )

            baseline_values = baseline_values[
                np.isfinite(baseline_values)
            ]

            n_valid = len(baseline_values)

            # No hay suficiente baseline válida para
            # esta combinación canal-feature.
            if n_valid < MIN_BASELINE_VALID:

                n_failed += 1

                summary_rows.append({
                    "method": method,
                    "session": session,
                    "participant": participant,
                    "feature": col,
                    "baseline_valid": n_valid,
                    "status": "INSUFFICIENT_BASELINE",
                })

                continue

            values = pd.to_numeric(
                group[col],
                errors="coerce"
            ).to_numpy(dtype=float)

            base_mean = baseline_values.mean()

            feature = feature_type(col)

            # =================================================
            # LOG-POWER:
            # diferencia respecto a baseline en dB
            # =================================================

            if feature in LOG_FEATURES:

                normalized = (
                    values - base_mean
                )

                status = "DELTA_BASELINE"

            # =================================================
            # RESTO:
            # z-score respecto a baseline EO
            # =================================================

            else:

                base_std = baseline_values.std(
                    ddof=1
                )

                if (
                    not np.isfinite(base_std)
                    or base_std <= EPS
                ):

                    n_failed += 1

                    summary_rows.append({
                        "method": method,
                        "session": session,
                        "participant": participant,
                        "feature": col,
                        "baseline_valid": n_valid,
                        "status": "INVALID_BASELINE_STD",
                    })

                    continue

                normalized = (
                    (values - base_mean)
                    / base_std
                )

                status = "ZSCORE_BASELINE"

            norm_cols[col + "_norm"][idx] = normalized

            n_normalized += 1

            summary_rows.append({
                "method": method,
                "session": session,
                "participant": participant,
                "feature": col,
                "baseline_valid": n_valid,
                "status": status,
            })

        print(
            f"  {session}/{participant}: "
            f"baseline epochs={len(baseline)}, "
            f"features normalizadas={n_normalized}, "
            f"sin baseline suficiente={n_failed}"
        )

    # --------------------------------------------------------
    # Añadir columnas normalizadas
    # --------------------------------------------------------

    norm_df = pd.DataFrame(
        norm_cols,
        index=df.index
    )

    result = pd.concat(
        [result, norm_df],
        axis=1
    )

    # --------------------------------------------------------
    # Guardar
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / f"features_{method}_normalized.csv"
    )

    result.to_csv(
        output_path,
        index=False
    )

    print(f"\nGuardado: {output_path}")


# ============================================================
# INFORME DE NORMALIZACIÓN
# ============================================================

summary = pd.DataFrame(summary_rows)

summary_path = (
    OUTPUT_DIR
    / "normalization_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print("\n" + "=" * 70)
print("NORMALIZACIÓN TERMINADA")
print("=" * 70)

print(f"Resumen guardado en: {summary_path}")

if len(summary) > 0:

    print("\nEstados:")
    print(
        summary["status"]
        .value_counts()
        .to_string()
    )