from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "data_features"

EXPECTED = {
    "all_mean": 16 * 11,
    "contralateral_mean": 16 * 11,
    "ipsilateral_mean": 16 * 11,
    "contralateral_bipolar": 64 * 11,
    "ipsilateral_bipolar": 56 * 11,
}


def finite_values(df, cols):
    x = df[cols].to_numpy(dtype=float).ravel()
    return x[np.isfinite(x)]


for path in sorted(FEATURE_DIR.glob("features_*.csv")):

    method = path.stem.replace("features_", "")

    print("\n" + "=" * 70)
    print(method)
    print("=" * 70)

    df = pd.read_csv(path)

    feature_cols = [
        c for c in df.columns
        if "__" in c
    ]

    print("Filas:", len(df))
    print("Features:", len(feature_cols))
    print("Features esperadas:", EXPECTED.get(method))

    if len(feature_cols) == EXPECTED.get(method):
        print("CHECK nº features: OK")
    else:
        print("CHECK nº features: ERROR")

    # ========================================================
    # NaN e infinitos
    # ========================================================

    X = df[feature_cols]

    n_inf = np.isinf(
        X.to_numpy(dtype=float)
    ).sum()

    nan_percent = (
        X.isna().sum().sum()
        / X.size
        * 100
    )

    all_nan = X.columns[
        X.isna().all()
    ].tolist()

    print(f"% NaN en features: {nan_percent:.2f}%")
    print("Infinitos:", n_inf)
    print("Columnas completamente NaN:", len(all_nan))

    if n_inf == 0:
        print("CHECK infinitos: OK")
    else:
        print("CHECK infinitos: ERROR")

    if len(all_nan) == 0:
        print("CHECK columnas todo NaN: OK")
    else:
        print("CHECK columnas todo NaN: ERROR")
        print(all_nan)

    # ========================================================
    # Features relativas: deben estar entre 0 y 1
    # ========================================================

    for feature in [
        "relative_theta",
        "relative_alpha",
        "relative_beta",
    ]:

        cols = [
            c for c in feature_cols
            if c.endswith("__" + feature)
        ]

        values = finite_values(df, cols)

        if len(values) == 0:
            print(f"{feature}: SIN DATOS")
            continue

        vmin = np.min(values)
        vmax = np.max(values)

        ok = np.all(
            (values >= 0)
            & (values <= 1)
        )

        print(
            f"{feature:22s}"
            f" min={vmin:.6f}"
            f" max={vmax:.6f}"
            f" -> {'OK' if ok else 'ERROR'}"
        )

    # ========================================================
    # Ratios: deben ser >= 0
    # ========================================================

    for feature in [
        "theta_alpha_ratio",
        "theta_beta_ratio",
    ]:

        cols = [
            c for c in feature_cols
            if c.endswith("__" + feature)
        ]

        values = finite_values(df, cols)

        if len(values) == 0:
            print(f"{feature}: SIN DATOS")
            continue

        vmin = np.min(values)
        vmax = np.max(values)

        ok = np.all(values >= 0)

        print(
            f"{feature:22s}"
            f" min={vmin:.6f}"
            f" max={vmax:.6f}"
            f" -> {'OK' if ok else 'ERROR'}"
        )

    # ========================================================
    # Hjorth: deben ser >= 0
    # ========================================================

    for feature in [
        "hjorth_mobility",
        "hjorth_complexity",
    ]:

        cols = [
            c for c in feature_cols
            if c.endswith("__" + feature)
        ]

        values = finite_values(df, cols)

        if len(values) == 0:
            print(f"{feature}: SIN DATOS")
            continue

        vmin = np.min(values)
        vmax = np.max(values)

        ok = np.all(values >= 0)

        print(
            f"{feature:22s}"
            f" min={vmin:.6f}"
            f" max={vmax:.6f}"
            f" -> {'OK' if ok else 'ERROR'}"
        )

    # ========================================================
    # Entropía espectral normalizada: 0 - 1
    # ========================================================

    feature = "spectral_entropy"

    cols = [
        c for c in feature_cols
        if c.endswith("__" + feature)
    ]

    values = finite_values(df, cols)

    if len(values) > 0:

        vmin = np.min(values)
        vmax = np.max(values)

        ok = np.all(
            (values >= 0)
            & (values <= 1)
        )

        print(
            f"{feature:22s}"
            f" min={vmin:.6f}"
            f" max={vmax:.6f}"
            f" -> {'OK' if ok else 'ERROR'}"
        )

    # ========================================================
    # Log-power
    # No tienen un rango teórico fijo
    # Solo comprobamos que sean finitos
    # ========================================================

    for feature in [
        "log_theta",
        "log_alpha",
        "log_beta",
    ]:

        cols = [
            c for c in feature_cols
            if c.endswith("__" + feature)
        ]

        values = finite_values(df, cols)

        if len(values) == 0:
            print(f"{feature}: SIN DATOS")
            continue

        print(
            f"{feature:22s}"
            f" min={np.min(values):.3f}"
            f" median={np.median(values):.3f}"
            f" max={np.max(values):.3f}"
            f" -> OK"
        )

    # ========================================================
    # Condiciones
    # ========================================================

    print("\nCondiciones:")
    print(
        df["condition"]
        .value_counts(dropna=False)
        .to_string()
    )