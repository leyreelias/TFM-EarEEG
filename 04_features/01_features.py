from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.integrate import trapezoid


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "data_epoched"
OUTPUT_DIR = ROOT / "data_features"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 125.0

BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}


# ============================================================
# FUNCIONES
# ============================================================

def load_epochs(npz_path):

    z = np.load(npz_path, allow_pickle=True)

    # Busca automáticamente el array 3D:
    # epochs x channels x samples
    key = None

    for k in z.files:
        if np.asarray(z[k]).ndim == 3:
            key = k
            break

    if key is None:
        raise ValueError(
            f"No encuentro array 3D en {npz_path}. "
            f"Keys: {z.files}"
        )

    epochs = np.asarray(z[key], dtype=float)

    # Intentar recuperar nombres de canales
    channel_names = None

    for k in ["channel_names", "ch_names", "channels"]:
        if k in z.files:
            channel_names = [
                str(x) for x in z[k].tolist()
            ]
            break

    if channel_names is None:
        channel_names = [
            f"ch_{i+1}"
            for i in range(epochs.shape[1])
        ]

    return epochs, channel_names



def bandpower(freqs, psd, low, high):

    mask = (
        (freqs >= low)
        & (freqs < high)
    )

    if mask.sum() < 2:
        return np.nan

    return trapezoid(
        psd[mask],
        freqs[mask]
    )


def hjorth(x):

    dx = np.diff(x)
    ddx = np.diff(dx)

    var_x = np.var(x)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)

    if var_x <= 0 or var_dx <= 0:
        return np.nan, np.nan

    mobility = np.sqrt(
        var_dx / var_x
    )

    mobility_dx = np.sqrt(
        var_ddx / var_dx
    )

    complexity = (
        mobility_dx / mobility
        if mobility > 0
        else np.nan
    )

    return mobility, complexity


def extract_features(x):

    x = np.asarray(x, dtype=float)

    # No interpolamos señal.
    # Si este canal contiene algún NaN en el epoch,
    # sus features quedan como NaN.
    if not np.all(np.isfinite(x)):
        return {
            name: np.nan
            for name in FEATURE_NAMES
        }

    x = x - np.mean(x)

    # Welch: ventanas de 2 s, 50 % overlap
    freqs, psd = welch(
        x,
        fs=FS,
        nperseg=250,
        noverlap=125,
        detrend="constant",
    )

    total = bandpower(
        freqs, psd, 1, 45
    )

    powers = {
        name: bandpower(
            freqs, psd, low, high
        )
        for name, (low, high)
        in BANDS.items()
    }

    eps = 1e-20

    # Log power en dB
    log_theta = 10 * np.log10(
        powers["theta"] + eps
    )

    log_alpha = 10 * np.log10(
        powers["alpha"] + eps
    )

    log_beta = 10 * np.log10(
        powers["beta"] + eps
    )

    relative_theta = (
        powers["theta"] / (total + eps)
    )

    relative_alpha = (
        powers["alpha"] / (total + eps)
    )

    relative_beta = (
        powers["beta"] / (total + eps)
    )

    theta_alpha_ratio = (
        powers["theta"]
        / (powers["alpha"] + eps)
    )

    theta_beta_ratio = (
        powers["theta"]
        / (powers["beta"] + eps)
    )

    mobility, complexity = hjorth(x)

    # Entropía espectral 1–45 Hz
    mask = (
        (freqs >= 1)
        & (freqs <= 45)
    )

    p = psd[mask].copy()
    p = p / (p.sum() + eps)

    spectral_entropy = (
        -np.sum(
            p * np.log2(p + eps)
        )
        / np.log2(len(p))
    )

    return {
        "log_theta": log_theta,
        "log_alpha": log_alpha,
        "log_beta": log_beta,
        "relative_theta": relative_theta,
        "relative_alpha": relative_alpha,
        "relative_beta": relative_beta,
        "theta_alpha_ratio": theta_alpha_ratio,
        "theta_beta_ratio": theta_beta_ratio,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
        "spectral_entropy": spectral_entropy,
    }


FEATURE_NAMES = [
    "log_theta",
    "log_alpha",
    "log_beta",
    "relative_theta",
    "relative_alpha",
    "relative_beta",
    "theta_alpha_ratio",
    "theta_beta_ratio",
    "hjorth_mobility",
    "hjorth_complexity",
    "spectral_entropy",
]


def safe_name(x):

    return re.sub(
        r"[^A-Za-z0-9_]+",
        "_",
        str(x)
    )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    methods = sorted(
        p.name
        for p in INPUT_DIR.iterdir()
        if p.is_dir()
    )

    for method in methods:

        rows = []

        files = sorted(
            (INPUT_DIR / method).glob(
                "session_*/participant_*/epochs.npz"
            )
        )

        print(f"\n{method}")

        for npz_path in files:

            session = npz_path.parents[1].name
            participant = npz_path.parent.name

            info_path = (
                npz_path.parent
                / "epoch_info_labeled.csv"
            )

            if not info_path.exists():
                print(
                    f"  Saltado {session}/{participant}: "
                    "falta epoch_info_labeled.csv"
                )
                continue

            epochs, channels = load_epochs(
                npz_path
            )

            info = pd.read_csv(
                info_path
            )

            # El .npz contiene solo los epochs aceptados.
            # Filtramos la metadata para quedarnos con los mismos.
            info = (
                info[info["keep"] == True]
                .reset_index(drop=True)
            )


            if len(info) != len(epochs):
                raise ValueError(
                    f"{session}/{participant}: "
                    f"{len(epochs)} epochs pero "
                    f"{len(info)} filas metadata"
                )

            for e in range(len(epochs)):

                # Metadata
                row = {
                    "method": method,
                    "session": session,
                    "participant": participant,
                }

                for col in [
                    "epoch",
                    "start_sample",
                    "end_sample",
                    "start_sec",
                    "end_sec",
                    "nan_percent",
                    "condition",
                    "performance",
                    "event_id",
                    "event_sec",
                ]:
                    if col in info.columns:
                        row[col] = info.iloc[e][col]

                # Features por canal
                for c, channel in enumerate(channels):

                    features = extract_features(
                        epochs[e, c]
                    )

                    ch = safe_name(channel)

                    for feature, value in features.items():

                        row[
                            f"{ch}__{feature}"
                        ] = value

                rows.append(row)

            print(
                f"  {session}/{participant}: "
                f"{len(epochs)} epochs"
            )

        result = pd.DataFrame(rows)

        output = (
            OUTPUT_DIR
            / f"features_{method}.csv"
        )

        result.to_csv(
            output,
            index=False
        )

        print(
            f"  Guardado: {output}"
        )

    print("\nFEATURE EXTRACTION TERMINADA")