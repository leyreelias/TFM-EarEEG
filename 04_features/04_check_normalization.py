from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data_features_normalized"

BASELINE = "eeg_calibration_EO"

# Tolerancias numéricas.
# Como usamos exactamente los mismos epochs para calcular
# media/std y luego normalizarlos, deberían quedar casi exactos.
MEAN_TOL = 1e-8
STD_TOL = 1e-6

# Estos NO son límites teóricos.
# Solo sirven para detectar valores extremos que merezcan revisión.
Z_EXTREME = 20
DELTA_DB_EXTREME = 50


LOG_FEATURES = {
    "log_theta",
    "log_alpha",
    "log_beta",
}


def base_feature_name(column):
    """
    L1__log_alpha_norm -> log_alpha
    """
    name = column.replace("_norm", "")
    return name.split("__")[-1]


def original_column(norm_column):
    """
    L1__log_alpha_norm -> L1__log_alpha
    """
    return norm_column[:-5]


# ============================================================
# EJECUCIÓN
# ============================================================

global_errors = 0


for path in sorted(
    INPUT_DIR.glob("features_*_normalized.csv")
):

    method = (
        path.stem
        .replace("features_", "")
        .replace("_normalized", "")
    )

    print("\n" + "=" * 75)
    print(method)
    print("=" * 75)

    df = pd.read_csv(path)

    norm_cols = [
        c for c in df.columns
        if c.endswith("_norm")
    ]

    original_cols = [
        original_column(c)
        for c in norm_cols
    ]

    print("Filas:", len(df))
    print("Features normalizadas:", len(norm_cols))

    # ========================================================
    # 1. INFINITOS
    # ========================================================

    Xnorm = df[norm_cols].to_numpy(dtype=float)

    n_inf = np.isinf(Xnorm).sum()

    print("Infinitos:", n_inf)

    if n_inf == 0:
        print("CHECK infinitos: OK")
    else:
        print("CHECK infinitos: ERROR")
        global_errors += 1

    # ========================================================
    # 2. NaN global
    # ========================================================

    nan_percent = (
        df[norm_cols].isna().sum().sum()
        / df[norm_cols].size
        * 100
    )

    print(
        f"% NaN en features normalizadas: "
        f"{nan_percent:.2f}%"
    )

    # ========================================================
    # 3. No crear información donde el original era NaN
    # ========================================================

    invalid_created = 0

    for norm_col, orig_col in zip(
        norm_cols,
        original_cols
    ):

        bad = (
            df[orig_col].isna()
            & df[norm_col].notna()
        )

        invalid_created += bad.sum()

    print(
        "Valores normalizados creados "
        "a partir de original NaN:",
        invalid_created
    )

    if invalid_created == 0:
        print("CHECK propagación NaN: OK")
    else:
        print("CHECK propagación NaN: ERROR")
        global_errors += 1

    # ========================================================
    # 4. Comprobar baseline sesión por sesión
    # ========================================================

    print("\nCHECK BASELINE EO POR SESIÓN")

    problems = []
    no_baseline = []

    groups = df.groupby(
        ["session", "participant"],
        sort=False
    )

    for (session, participant), group in groups:

        baseline = group[
            group["condition"] == BASELINE
        ]

        if len(baseline) == 0:

            no_baseline.append(
                (session, participant, "0 epochs EO")
            )

            continue

        n_checked = 0
        n_failed = 0
        n_unavailable = 0

        for norm_col in norm_cols:

            feature = base_feature_name(norm_col)

            values = pd.to_numeric(
                baseline[norm_col],
                errors="coerce"
            )

            values = values[
                np.isfinite(values)
            ].to_numpy()

            # Esa feature no pudo normalizarse
            # por falta de baseline válida.
            if len(values) == 0:
                n_unavailable += 1
                continue

            n_checked += 1

            mean = np.mean(values)

            # ------------------------------------------------
            # LOG-POWER:
            # baseline debe tener media ~0
            # ------------------------------------------------

            if feature in LOG_FEATURES:

                if abs(mean) > MEAN_TOL:

                    n_failed += 1

                    problems.append(
                        (
                            session,
                            participant,
                            norm_col,
                            "mean",
                            mean
                        )
                    )

            # ------------------------------------------------
            # Z-SCORE:
            # baseline debe tener media ~0
            # y desviación estándar ~1
            # ------------------------------------------------

            else:

                if len(values) < 2:

                    n_unavailable += 1
                    continue

                std = np.std(
                    values,
                    ddof=1
                )

                if (
                    abs(mean) > MEAN_TOL
                    or abs(std - 1) > STD_TOL
                ):

                    n_failed += 1

                    problems.append(
                        (
                            session,
                            participant,
                            norm_col,
                            "mean/std",
                            (mean, std)
                        )
                    )

        print(
            f"  {session}/{participant}: "
            f"EO={len(baseline)}, "
            f"comprobadas={n_checked}, "
            f"no disponibles={n_unavailable}, "
            f"fallos={n_failed}"
        )

    if len(problems) == 0:
        print("\nCHECK centrado baseline: OK")
    else:
        print(
            "\nCHECK centrado baseline: ERROR "
            f"({len(problems)} problemas)"
        )

        global_errors += 1

        for x in problems[:20]:
            print(" ", x)

    if no_baseline:

        print("\nSesiones SIN baseline EO:")
        for x in no_baseline:
            print(" ", x)

    # ========================================================
    # 5. Comprobar valores extremos después de normalizar
    # ========================================================

    print("\nVALORES NORMALIZADOS")

    # --------------------------------------------------------
    # Delta log-power (dB)
    # --------------------------------------------------------

    log_cols = [
        c for c in norm_cols
        if base_feature_name(c) in LOG_FEATURES
    ]

    log_values = (
        df[log_cols]
        .to_numpy(dtype=float)
        .ravel()
    )

    log_values = log_values[
        np.isfinite(log_values)
    ]

    if len(log_values):

        p1, p50, p99 = np.percentile(
            log_values,
            [1, 50, 99]
        )

        max_abs = np.max(
            np.abs(log_values)
        )

        print(
            "Delta log-power:"
            f" p1={p1:.3f},"
            f" mediana={p50:.3f},"
            f" p99={p99:.3f},"
            f" max|x|={max_abs:.3f} dB"
        )

        n_extreme = np.sum(
            np.abs(log_values)
            > DELTA_DB_EXTREME
        )

        print(
            f"  |delta| > {DELTA_DB_EXTREME} dB:",
            n_extreme
        )

    # --------------------------------------------------------
    # Features con z-score
    # --------------------------------------------------------

    z_cols = [
        c for c in norm_cols
        if base_feature_name(c)
        not in LOG_FEATURES
    ]

    z_values = (
        df[z_cols]
        .to_numpy(dtype=float)
        .ravel()
    )

    z_values = z_values[
        np.isfinite(z_values)
    ]

    if len(z_values):

        p1, p50, p99 = np.percentile(
            z_values,
            [1, 50, 99]
        )

        max_abs = np.max(
            np.abs(z_values)
        )

        print(
            "Z-scores:"
            f" p1={p1:.3f},"
            f" mediana={p50:.3f},"
            f" p99={p99:.3f},"
            f" max|z|={max_abs:.3f}"
        )

        n_extreme = np.sum(
            np.abs(z_values)
            > Z_EXTREME
        )

        print(
            f"  |z| > {Z_EXTREME}:",
            n_extreme
        )

    # ========================================================
    # 6. Columnas totalmente NaN
    # ========================================================

    all_nan = [
        c for c in norm_cols
        if df[c].isna().all()
    ]

    print(
        "\nColumnas normalizadas "
        "completamente NaN:",
        len(all_nan)
    )

    if all_nan:
        print(
            "Primeras:",
            all_nan[:20]
        )


print("\n" + "=" * 75)

if global_errors == 0:
    print("CHECK FINAL: OK")
    print(
        "No se han detectado errores "
        "matemáticos en la normalización."
    )
else:
    print(
        f"CHECK FINAL: {global_errors} "
        "bloques con ERROR"
    )

print("=" * 75)