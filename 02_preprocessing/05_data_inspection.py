from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mne.time_frequency import psd_array_welch


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "data_rereferenced"

RESULTS_DIR = (
    ROOT
    / "results"
    / "results_02"
    / "reref_inspection"
)

FS = 125

EEG_FILE = "obci_eeg1_data.csv"

MIN_PSD_SEC = 4.0

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FUNCIONES
# ============================================================

def longest_true_run(mask):
    """
    Devuelve inicio y final del segmento True
    continuo más largo.
    """

    best_start = 0
    best_end = 0

    start = None

    for i, value in enumerate(mask):

        if value and start is None:
            start = i

        if (
            start is not None
            and (
                not value
                or i == len(mask) - 1
            )
        ):

            end = (
                i + 1
                if value and i == len(mask) - 1
                else i
            )

            if (
                end - start
                > best_end - best_start
            ):
                best_start = start
                best_end = end

            start = None

    return best_start, best_end


def robust_range(x):
    """
    Rango robusto = percentil 95 - percentil 5.
    """

    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return (
        np.percentile(x, 95)
        - np.percentile(x, 5)
    )


def spectral_metrics(segment):
    """
    PSD Welch usando MNE.

    segment:
        muestras x canales
    """

    if len(segment) < FS * MIN_PSD_SEC:
        return np.nan, np.nan, np.nan, np.nan

    x = segment.T

    psd, freqs = psd_array_welch(
        x,
        sfreq=FS,
        fmin=1,
        fmax=45,
        n_fft=256,
        n_per_seg=256,
        n_overlap=128,
        average="median",
        verbose=False,
    )

    def power(low, high):

        mask = (
            (freqs >= low)
            & (freqs < high)
        )

        return np.trapezoid(
            psd[:, mask],
            freqs[mask],
            axis=1,
        )

    total = power(1, 45)

    theta = power(4, 8)
    alpha = power(8, 13)
    beta = power(13, 30)
    high = power(30, 45)

    theta_rel = np.median(
        theta / (total + 1e-12)
    )

    alpha_rel = np.median(
        alpha / (total + 1e-12)
    )

    beta_rel = np.median(
        beta / (total + 1e-12)
    )

    hf_ratio = np.median(
        high / (total + 1e-12)
    )

    return (
        theta_rel,
        alpha_rel,
        beta_rel,
        hf_ratio,
    )


# ============================================================
# EJECUCIÓN
# ============================================================

summary = []
channel_results = []


for eeg_path in sorted(
    INPUT_DIR.glob(
        "*/session_*/participant_*/obci_eeg1_data.csv"
    )
):

    method = eeg_path.parents[2].name
    session = eeg_path.parents[1].name
    participant = eeg_path.parent.name

    print(
        f"\n{method} - "
        f"{session}/{participant}"
    )

    df = pd.read_csv(eeg_path)

    eeg_cols = [
        c for c in df.columns
        if c.startswith("chann_")
        or c.startswith("bp_")
    ]

    data = (
        df[eeg_cols]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(dtype=float)
    )

    # ========================================================
    # NaN
    # ========================================================

    nan_percent = (
        100
        * np.mean(~np.isfinite(data))
    )

    channel_nan = (
        100
        * np.mean(
            ~np.isfinite(data),
            axis=0,
        )
    )

    fully_nan = (
        channel_nan == 100
    )

    usable_cols = (
        ~fully_nan
    )

    usable_data = data[
        :,
        usable_cols
    ]

    # ========================================================
    # MÉTRICAS POR CANAL
    # ========================================================

    channel_ranges = []

    for i, ch in enumerate(eeg_cols):

        rr = robust_range(
            data[:, i]
        )

        channel_ranges.append(rr)

        channel_results.append({
            "method": method,
            "session": session,
            "participant": participant,
            "channel": ch,
            "nan_percent":
                channel_nan[i],
            "robust_range":
                rr,
            "fully_nan":
                fully_nan[i],
        })

    median_range = np.nanmedian(
        channel_ranges
    )

    # ========================================================
    # SEGMENTO COMPLETO MÁS LARGO
    # ========================================================

    if usable_data.shape[1] > 0:

        complete_rows = np.all(
            np.isfinite(usable_data),
            axis=1,
        )

        start, end = (
            longest_true_run(
                complete_rows
            )
        )

        longest_samples = (
            end - start
        )

        longest_sec = (
            longest_samples / FS
        )

    else:

        start = 0
        end = 0
        longest_sec = 0

    # ========================================================
    # RANGO / RANK EMPÍRICO
    # ========================================================

    empirical_rank = np.nan

    if (
        end - start
        >= FS * 2
        and usable_data.shape[1] > 0
    ):

        segment = (
            usable_data[
                start:end
            ]
        )

        centered = (
            segment
            - np.mean(
                segment,
                axis=0,
            )
        )

        empirical_rank = (
            np.linalg.matrix_rank(
                centered
            )
        )

    # ========================================================
    # PSD
    # ========================================================

    theta_rel = np.nan
    alpha_rel = np.nan
    beta_rel = np.nan
    hf_ratio = np.nan

    if (
        end - start
        >= FS * MIN_PSD_SEC
    ):

        (
            theta_rel,
            alpha_rel,
            beta_rel,
            hf_ratio,
        ) = spectral_metrics(
            usable_data[
                start:end
            ]
        )

    # ========================================================
    # GUARDAR
    # ========================================================

    summary.append({

        "method":
            method,

        "session":
            session,

        "participant":
            participant,

        "n_channels":
            len(eeg_cols),

        "n_usable_channels":
            int(
                usable_cols.sum()
            ),

        "n_fully_nan_channels":
            int(
                fully_nan.sum()
            ),

        "nan_percent":
            nan_percent,

        "median_channel_nan_percent":
            np.median(
                channel_nan
            ),

        "max_channel_nan_percent":
            np.max(
                channel_nan
            ),

        "median_robust_range":
            median_range,

        "longest_complete_sec":
            longest_sec,

        "empirical_rank":
            empirical_rank,

        "relative_theta":
            theta_rel,

        "relative_alpha":
            alpha_rel,

        "relative_beta":
            beta_rel,

        "high_frequency_ratio_30_45":
            hf_ratio,
    })

    print(
        f"  NaN: {nan_percent:.1f}% | "
        f"canales útiles: "
        f"{usable_cols.sum()}/{len(eeg_cols)} | "
        f"segmento limpio: "
        f"{longest_sec:.1f} s | "
        f"rank: {empirical_rank}"
    )


# ============================================================
# CSV
# ============================================================

summary = pd.DataFrame(summary)

channels = pd.DataFrame(
    channel_results
)

summary.to_csv(
    RESULTS_DIR
    / "01_reref_inspection_summary.csv",
    index=False,
)

channels.to_csv(
    RESULTS_DIR
    / "02_reref_channel_inspection.csv",
    index=False,
)


# ============================================================
# RESUMEN POR MÉTODO
# ============================================================

method_summary = (
    summary
    .groupby("method")
    .agg({

        "nan_percent":
            "median",

        "longest_complete_sec":
            "median",

        "empirical_rank":
            "median",

        "high_frequency_ratio_30_45":
            "median",

    })
    .reset_index()
)

method_summary.to_csv(
    RESULTS_DIR
    / "03_method_summary.csv",
    index=False,
)


# ============================================================
# FIGURA AUTOMÁTICA
# ============================================================

methods = summary[
    "method"
].unique()

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 8),
)

# ------------------------------------------------------------
# NaN
# ------------------------------------------------------------

data_plot = [
    summary.loc[
        summary["method"] == m,
        "nan_percent",
    ].dropna()
    for m in methods
]

axes[0, 0].boxplot(
    data_plot,
    labels=methods,
)

axes[0, 0].set_title(
    "% NaN"
)

axes[0, 0].tick_params(
    axis="x",
    rotation=45,
)

# ------------------------------------------------------------
# Segmento limpio
# ------------------------------------------------------------

data_plot = [
    summary.loc[
        summary["method"] == m,
        "longest_complete_sec",
    ].dropna()
    for m in methods
]

axes[0, 1].boxplot(
    data_plot,
    labels=methods,
)

axes[0, 1].set_title(
    "Segmento continuo más largo"
)

axes[0, 1].set_ylabel(
    "segundos"
)

axes[0, 1].tick_params(
    axis="x",
    rotation=45,
)

# ------------------------------------------------------------
# Rank
# ------------------------------------------------------------

data_plot = [
    summary.loc[
        summary["method"] == m,
        "empirical_rank",
    ].dropna()
    for m in methods
]

axes[1, 0].boxplot(
    data_plot,
    labels=methods,
)

axes[1, 0].set_title(
    "Rango empírico"
)

axes[1, 0].tick_params(
    axis="x",
    rotation=45,
)

# ------------------------------------------------------------
# Alta frecuencia
# ------------------------------------------------------------

data_plot = [
    summary.loc[
        summary["method"] == m,
        "high_frequency_ratio_30_45",
    ].dropna()
    for m in methods
]

axes[1, 1].boxplot(
    data_plot,
    labels=methods,
)

axes[1, 1].set_title(
    "Potencia relativa 30-45 Hz"
)

axes[1, 1].tick_params(
    axis="x",
    rotation=45,
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    / "04_reref_inspection.png",
    dpi=200,
)

plt.close()


print(
    "\nINSPECCIÓN AUTOMÁTICA TERMINADA"
)

print("\nResumen por método:\n")

print(
    method_summary.to_string(
        index=False
    )
)