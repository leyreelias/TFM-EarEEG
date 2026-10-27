from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.signal import welch, butter, sosfiltfilt


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "data_trimmed"
OUTPUT_DIR = ROOT / "data_cleaned"
RESULTS_DIR = ROOT / "results" / "results_02"

EEG_FILE = "obci_eeg1_data.csv"

AUX_FILES = [
    "obci_eeg2_data.csv",
    "obci_eeg3_data.csv",
    "psychopymarkers_data.csv",
]

FS = 125.0

WINDOW_SEC = 4
STEP_SEC = 2

WINDOW = int(WINDOW_SEC * FS)   # 500 muestras
STEP = int(STEP_SEC * FS)       # 250 muestras


# ============================================================
# UMBRALES
# ============================================================

MAX_NAN_PERCENT = 5.0
MAX_FLATLINE_SEC = 1.0

ROBUST_Z_LIMIT = 5.0
EXTREME_Z_LIMIT = 8.0
MIN_CORRELATION = 0.40

PERMANENT_BAD_FRACTION = 0.40

BAD_WINDOW_CHANNEL_FRACTION = 0.25

MAX_PERMANENT_BAD_CHANNELS_SESSION = 4
MAX_BAD_WINDOW_FRACTION_SESSION = 0.40


# ============================================================
# MAPEO EAR-EEG
# ============================================================

LABELS = {
    "chann_1": "L1",
    "chann_2": "L2",
    "chann_3": "L4",
    "chann_4": "L5",
    "chann_5": "L7",
    "chann_6": "L8",
    "chann_7": "L9",
    "chann_8": "L10",

    "chann_9": "R1",
    "chann_10": "R2",
    "chann_11": "R4",
    "chann_12": "R5",
    "chann_13": "R7",
    "chann_14": "R8",
    "chann_15": "R9",
    "chann_16": "R10",
}


def get_channels(df):
    return [
        c for c in df.columns
        if c.startswith("chann_")
    ]


def same_ear(ch1, ch2):
    return LABELS[ch1][0] == LABELS[ch2][0]


# ============================================================
# MÉTRICAS
# ============================================================

def robust_z(values):

    x = np.asarray(values, dtype=float)

    median = np.nanmedian(x)

    q25, q75 = np.nanpercentile(
        x,
        [25, 75],
    )

    # Robust SD utilizada por PREP
    robust_sd = 0.7413 * (q75 - q25)

    if (
        not np.isfinite(robust_sd)
        or robust_sd == 0
    ):
        return np.zeros(len(x))

    return (x - median) / robust_sd


def flatline_seconds(x):

    x = np.asarray(x, dtype=float)

    longest = 0
    current = 0

    for i in range(1, len(x)):

        if (
            np.isfinite(x[i])
            and np.isfinite(x[i - 1])
            and x[i] == x[i - 1]
        ):
            current += 1
            longest = max(
                longest,
                current + 1,
            )

        else:
            current = 0

    return longest / FS


def analyse_window(window, channels):

    data = (
        window[channels]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(dtype=float)
    )

    # --------------------------------------------------------
    # NaNs y flatline
    # --------------------------------------------------------

    nan_percent = (
        100
        * np.mean(~np.isfinite(data), axis=0)
    )

    flatline = np.array([
        flatline_seconds(data[:, i])
        for i in range(len(channels))
    ])

    # --------------------------------------------------------
    # Amplitud robusta
    # --------------------------------------------------------

    q25 = np.nanpercentile(
        data,
        25,
        axis=0,
    )

    q75 = np.nanpercentile(
        data,
        75,
        axis=0,
    )

    robust_amplitude = (
        0.7413
        * (q75 - q25)
    )

    amplitude_z = robust_z(
        robust_amplitude
    )

    # --------------------------------------------------------
    # Copia sin NaNs para correlación y PSD
    # --------------------------------------------------------

    clean = (
        pd.DataFrame(data)
        .interpolate(limit_direction="both")
        .fillna(0)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Correlación 1-30 Hz
    # --------------------------------------------------------

    sos = butter(
        4,
        [1, 30],
        btype="bandpass",
        fs=FS,
        output="sos",
    )

    filtered = sosfiltfilt(
        sos,
        clean,
        axis=0,
    )

    corr = np.abs(
        np.corrcoef(
            filtered,
            rowvar=False,
        )
    )

    correlations = []

    for i, ch in enumerate(channels):

        indexes = [
            j
            for j, other in enumerate(channels)
            if (
                i != j
                and same_ear(ch, other)
            )
        ]

        values = corr[i, indexes]

        values = values[
            np.isfinite(values)
        ]

        correlations.append(
            np.max(values)
            if len(values)
            else 0
        )

    correlations = np.asarray(
        correlations
    )

    # --------------------------------------------------------
    # Ruido alta frecuencia
    # --------------------------------------------------------

    freqs, psd = welch(
        clean,
        fs=FS,
        nperseg=int(2 * FS),
        axis=0,
    )

    total = (
        (freqs >= 1)
        & (freqs < 45)
    )

    high = (
        (freqs >= 30)
        & (freqs < 45)
    )

    total_power = np.trapezoid(
        psd[total],
        freqs[total],
        axis=0,
    )

    high_power = np.trapezoid(
        psd[high],
        freqs[high],
        axis=0,
    )

    hf_ratio = np.divide(
        high_power,
        total_power,
        out=np.zeros_like(high_power),
        where=total_power > 0,
    )

    hf_z = robust_z(
        hf_ratio
    )

    # --------------------------------------------------------
    # Decisión
    # --------------------------------------------------------

    bad_nan = (
        nan_percent
        > MAX_NAN_PERCENT
    )

    bad_flatline = (
        flatline
        >= MAX_FLATLINE_SEC
    )

    bad_amplitude = (
        np.abs(amplitude_z)
        > ROBUST_Z_LIMIT
    )

    bad_correlation = (
        correlations
        < MIN_CORRELATION
    )

    bad_hf = (
        hf_z
        > ROBUST_Z_LIMIT
    )

    # --------------------------------------------------------
    # Decisión final
    # --------------------------------------------------------

    # Criterios graves:
    # uno solo basta para rechazar el canal en esa ventana
    severe = (
        bad_nan
        | bad_flatline
    )

    # Criterios débiles:
    # necesitamos al menos 2 simultáneamente
    n_weak = (
        bad_amplitude.astype(int)
        + bad_correlation.astype(int)
        + bad_hf.astype(int)
    )

    # Casos extremadamente anómalos:
    # pueden rechazarse aunque solo falle un criterio
    very_bad_amplitude = (
        np.abs(amplitude_z)
        > EXTREME_Z_LIMIT
    )

    very_bad_hf = (
        hf_z
        > EXTREME_Z_LIMIT
    )

    # Decisión final del canal en esta ventana
    bad = (
        severe
        | very_bad_amplitude
        | very_bad_hf
        | (n_weak >= 2)
    )

    return pd.DataFrame({
        "channel": channels,
        "channel_label": [
            LABELS[c]
            for c in channels
        ],

        "nan_percent": nan_percent,
        "flatline_sec": flatline,

        "amplitude_z":
            amplitude_z,

        "max_same_ear_correlation":
            correlations,

        "hf_noise_z":
            hf_z,

        "bad_nan":
            bad_nan,

        "bad_flatline":
            bad_flatline,

        "bad_amplitude":
            bad_amplitude,

        "bad_correlation":
            bad_correlation,

        "bad_high_frequency":
            bad_hf,
        
        "very_bad_amplitude":
            very_bad_amplitude,

        "very_bad_high_frequency":
            very_bad_hf,
        
        "n_weak_criteria": n_weak,

        "bad_channel":
            bad,
    })


# ============================================================
# PROCESAR UNA SESIÓN
# ============================================================

def process_session(folder):

    session = folder.parent.name
    participant = folder.name

    eeg = pd.read_csv(
        folder / EEG_FILE
    )

    channels = get_channels(eeg)

    if len(channels) != 16:
        raise ValueError(
            f"{session}/{participant}: "
            f"se esperaban 16 canales, "
            f"hay {len(channels)}"
        )

    channel_window_rows = []
    window_rows = []

    masks = {
        ch: np.zeros(
            len(eeg),
            dtype=bool,
        )
        for ch in channels
    }

    global_mask = np.zeros(
        len(eeg),
        dtype=bool,
    )

    window_index = 0

    # --------------------------------------------------------
    # Ventanas
    # --------------------------------------------------------

    for start in range(
        0,
        len(eeg) - WINDOW + 1,
        STEP,
    ):

        end = start + WINDOW

        quality = analyse_window(
            eeg.iloc[start:end],
            channels,
        )

        quality["window_index"] = (
            window_index
        )

        quality["start_sample"] = start
        quality["end_sample"] = end

        n_bad = int(
            quality["bad_channel"].sum()
        )

        bad_window = (
            n_bad
            >= BAD_WINDOW_CHANNEL_FRACTION
            * len(channels)
        )

        window_rows.append({
            "session": session,
            "participant": participant,
            "window_index": window_index,
            "start_sample": start,
            "end_sample": end,
            "n_bad_channels": n_bad,
            "bad_window": bad_window,
        })

        for _, row in quality.iterrows():

            ch = row["channel"]

            if row["bad_channel"]:
                masks[ch][start:end] = True

            channel_window_rows.append({
                "session": session,
                "participant": participant,
                **row.to_dict(),
            })

        if bad_window:
            global_mask[start:end] = True

        window_index += 1

    channel_window = pd.DataFrame(
        channel_window_rows
    )

    windows = pd.DataFrame(
        window_rows
    )

    # --------------------------------------------------------
    # Canal permanente
    # --------------------------------------------------------

    channel_summary = (
        channel_window
        .groupby(
            [
                "session",
                "participant",
                "channel",
                "channel_label",
            ]
        )["bad_channel"]
        .mean()
        .reset_index(
            name="bad_fraction"
        )
    )

    channel_summary[
        "permanent_bad"
    ] = (
        channel_summary["bad_fraction"]
        >= PERMANENT_BAD_FRACTION
    )

    permanent_bad = (
        channel_summary.loc[
            channel_summary[
                "permanent_bad"
            ],
            "channel",
        ]
        .tolist()
    )

    # --------------------------------------------------------
    # Motivos de rechazo de canales permanentes
    # --------------------------------------------------------

    for ch in permanent_bad:

        ch_data = channel_window[
            channel_window["channel"] == ch
        ]

        print(
            f"    {LABELS[ch]} -> "
            f"NaN: {100 * ch_data['bad_nan'].mean():.1f}% | "
            f"flatline: {100 * ch_data['bad_flatline'].mean():.1f}% | "
            f"amplitud: {100 * ch_data['bad_amplitude'].mean():.1f}% | "
            f"correlación: {100 * ch_data['bad_correlation'].mean():.1f}% | "
            f"HF: {100 * ch_data['bad_high_frequency'].mean():.1f}%"
        )

    # --------------------------------------------------------
    # Decisión sesión
    # --------------------------------------------------------

    bad_window_fraction = (
        windows["bad_window"].mean()
        if len(windows)
        else 1.0
    )

    reject_session = (
        len(permanent_bad)
        >= MAX_PERMANENT_BAD_CHANNELS_SESSION
        or
        bad_window_fraction
        >= MAX_BAD_WINDOW_FRACTION_SESSION
    )

    summary = {
        "session":
            session,

        "participant":
            participant,

        "n_permanent_bad":
            len(permanent_bad),

        "permanent_bad_channels":
            ",".join(
                LABELS[ch]
                for ch in permanent_bad
            ),

        "bad_window_percent":
            100 * bad_window_fraction,

        "session_rejected":
            reject_session,
    }

    # --------------------------------------------------------
    # LIMPIEZA REAL
    # --------------------------------------------------------

    if not reject_session:

        clean_eeg = eeg.copy()

        for ch in channels:

            # Canal permanentemente malo:
            # se invalida completo
            if ch in permanent_bad:

                clean_eeg[ch] = np.nan

            else:

                # Segmentos malos de ese canal
                mask = (
                    masks[ch]
                    | global_mask
                )

                clean_eeg.loc[
                    mask,
                    ch,
                ] = np.nan

        out = (
            OUTPUT_DIR
            / session
            / participant
        )

        out.mkdir(
            parents=True,
            exist_ok=True,
        )

        clean_eeg.to_csv(
            out / EEG_FILE,
            index=False,
        )

        # Copiamos el resto de archivos
        for filename in AUX_FILES:

            src = folder / filename

            if src.exists():
                shutil.copy2(
                    src,
                    out / filename,
                )

    return (
        channel_window,
        windows,
        channel_summary,
        summary,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT_DIR.exists():
        shutil.rmtree(
            OUTPUT_DIR
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_channels = []
    all_windows = []
    all_channel_summary = []
    all_sessions = []

    folders = sorted(
        INPUT_DIR.glob(
            "session_*/participant_*"
        )
    )

    for folder in folders:

        if not (
            folder / EEG_FILE
        ).exists():
            continue

        print(
            f"\nProcesando "
            f"{folder.parent.name}/"
            f"{folder.name}"
        )

        (
            channel_window,
            windows,
            channel_summary,
            summary,
        ) = process_session(folder)

        all_channels.append(
            channel_window
        )

        all_windows.append(
            windows
        )

        all_channel_summary.append(
            channel_summary
        )

        all_sessions.append(
            summary
        )

        print(
            f"  Canales permanentes malos: "
            f"{summary['n_permanent_bad']}"
        )

        print(
            f"  Ventanas malas: "
            f"{summary['bad_window_percent']:.1f}%"
        )

        print(
            f"  Sesión rechazada: "
            f"{summary['session_rejected']}"
        )

    # --------------------------------------------------------
    # Resultados
    # --------------------------------------------------------

    pd.concat(
        all_channels,
        ignore_index=True,
    ).to_csv(
        RESULTS_DIR
        / "01_channel_window_rejection.csv",
        index=False,
    )

    pd.concat(
        all_windows,
        ignore_index=True,
    ).to_csv(
        RESULTS_DIR
        / "02_window_rejection.csv",
        index=False,
    )

    pd.concat(
        all_channel_summary,
        ignore_index=True,
    ).to_csv(
        RESULTS_DIR
        / "03_channel_rejection.csv",
        index=False,
    )

    session_df = pd.DataFrame(
        all_sessions
    )

    session_df.to_csv(
        RESULTS_DIR
        / "04_session_rejection.csv",
        index=False,
    )

    print(
        "\nLIMPIEZA TERMINADA\n"
    )

    print(
        session_df.to_string(
            index=False
        )
    )