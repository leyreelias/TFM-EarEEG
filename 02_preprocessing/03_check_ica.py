from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

BEFORE_DIR = ROOT / "data_filtered"
AFTER_DIR = ROOT / "data_ica_clean"

RESULTS_DIR = ROOT / "results" / "results_02" / "ica"

EEG_FILE = "obci_eeg1_data.csv"
FS = 125

CHANNELS = [f"chann_{i}" for i in range(1, 17)]


# ============================================================
# FUNCIONES
# ============================================================

def longest_segment(mask):

    changes = np.diff(
        np.r_[False, mask, False].astype(int)
    )

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    if len(starts) == 0:
        return None

    lengths = ends - starts
    i = np.argmax(lengths)

    return starts[i], ends[i]


def bandpower(x, low, high):

    freqs, psd = welch(
        x,
        fs=FS,
        nperseg=min(500, len(x)),
        axis=0,
    )

    mask = (
        (freqs >= low)
        & (freqs < high)
    )

    return np.trapezoid(
        psd[mask],
        freqs[mask],
        axis=0,
    )


# ============================================================
# ICA SUMMARY
# ============================================================

summary_path = (
    RESULTS_DIR
    / "01_ica_summary.csv"
)

ica_summary = (
    pd.read_csv(summary_path)
    if summary_path.exists()
    else pd.DataFrame()
)


# ============================================================
# COMPARACIÓN
# ============================================================

rows = []

for after_path in sorted(
    AFTER_DIR.glob(
        "session_*/participant_*/obci_eeg1_data.csv"
    )
):

    session = after_path.parents[1].name
    participant = after_path.parent.name

    before_path = (
        BEFORE_DIR
        / session
        / participant
        / EEG_FILE
    )

    before = (
        pd.read_csv(before_path)[CHANNELS]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )

    after = (
        pd.read_csv(after_path)[CHANNELS]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )

    # --------------------------------------------------------
    # NaN
    # --------------------------------------------------------

    nan_before = 100 * np.mean(~np.isfinite(before))
    nan_after = 100 * np.mean(~np.isfinite(after))

    # Canales no permanentemente perdidos
    usable = (
        np.any(np.isfinite(before), axis=0)
        & np.any(np.isfinite(after), axis=0)
    )

    b = before[:, usable]
    a = after[:, usable]

    # --------------------------------------------------------
    # Solo muestras completas en ambos
    # --------------------------------------------------------

    complete = (
        np.all(np.isfinite(b), axis=1)
        & np.all(np.isfinite(a), axis=1)
    )

    segment = longest_segment(complete)

    if segment is None:
        continue

    start, end = segment

    b = b[start:end]
    a = a[start:end]

    # --------------------------------------------------------
    # Aplicar CAR al BEFORE
    # para compararlo justamente con ICA
    # --------------------------------------------------------

    b = b - np.mean(
        b,
        axis=1,
        keepdims=True,
    )

    # --------------------------------------------------------
    # Amplitud
    # --------------------------------------------------------

    rms_before = np.median(
        np.sqrt(np.mean(b ** 2, axis=0))
    )

    rms_after = np.median(
        np.sqrt(np.mean(a ** 2, axis=0))
    )

    range_before = np.median(
        np.percentile(b, 95, axis=0)
        - np.percentile(b, 5, axis=0)
    )

    range_after = np.median(
        np.percentile(a, 95, axis=0)
        - np.percentile(a, 5, axis=0)
    )

    # --------------------------------------------------------
    # PSD
    # --------------------------------------------------------

    total_before = bandpower(b, 1, 45)
    total_after = bandpower(a, 1, 45)

    low_before = bandpower(b, 1, 4)
    low_after = bandpower(a, 1, 4)

    high_before = bandpower(b, 30, 45)
    high_after = bandpower(a, 30, 45)

    low_rel_before = np.median(
        low_before / (total_before + 1e-12)
    )

    low_rel_after = np.median(
        low_after / (total_after + 1e-12)
    )

    high_rel_before = np.median(
        high_before / (total_before + 1e-12)
    )

    high_rel_after = np.median(
        high_after / (total_after + 1e-12)
    )

    # --------------------------------------------------------
    # Cambio total introducido por ICA
    # --------------------------------------------------------

    mean_abs_change = np.mean(
        np.abs(a - b)
    )

    # --------------------------------------------------------
    # Nº componentes eliminados
    # --------------------------------------------------------

    n_excluded = np.nan

    if not ica_summary.empty:

        r = ica_summary[
            (ica_summary["session"] == session)
            & (
                ica_summary["participant"]
                == participant
            )
        ]

        if len(r):
            n_excluded = int(
                r.iloc[0]["n_excluded"]
            )

    rows.append({
        "session": session,
        "participant": participant,
        "n_excluded": n_excluded,

        "segment_sec":
            (end - start) / FS,

        "nan_before":
            nan_before,

        "nan_after":
            nan_after,

        "rms_before":
            rms_before,

        "rms_after":
            rms_after,

        "robust_range_before":
            range_before,

        "robust_range_after":
            range_after,

        "low_1_4_before":
            low_rel_before,

        "low_1_4_after":
            low_rel_after,

        "high_30_45_before":
            high_rel_before,

        "high_30_45_after":
            high_rel_after,

        "mean_abs_change":
            mean_abs_change,
    })

    print(
        f"{session}/{participant} | "
        f"IC eliminados: {n_excluded} | "
        f"NaN {nan_before:.1f}->{nan_after:.1f}% | "
        f"RMS {rms_before:.2f}->{rms_after:.2f} | "
        f"HF {high_rel_before:.3f}->{high_rel_after:.3f} | "
        f"cambio {mean_abs_change:.3f}"
    )


# ============================================================
# GUARDAR
# ============================================================

results = pd.DataFrame(rows)

results.to_csv(
    RESULTS_DIR
    / "02_ica_pre_post_check.csv",
    index=False,
)

print("\nCOMPROBACIÓN ICA TERMINADA")