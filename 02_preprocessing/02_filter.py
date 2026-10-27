from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import mne


"""
02 - Filtrado EEG

Entrada:
    data_cleaned

Salida:
    data_filtered

Proceso:
    - Frecuencia de muestreo fija: 125 Hz.
    - Mantiene los NaN generados durante la limpieza.
    - Filtra por separado cada tramo continuo válido.
    - Notch a 50 Hz.
    - Band-pass 0.5 - 45 Hz.
    - Tramos válidos menores de 10 s se descartan.
"""


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT_DIR / "data_cleaned"
OUTPUT_DIR = ROOT_DIR / "data_filtered"
RESULTS_DIR = ROOT_DIR / "results" / "results_02"

EEG_FILE = "obci_eeg1_data.csv"

AUX_FILES = [
    "obci_eeg2_data.csv",
    "obci_eeg3_data.csv",
    "psychopymarkers_data.csv",
]

FS = 125.0

NOTCH_FREQ = 50.0
LOW_FREQ = 0.5
HIGH_FREQ = 45.0

MIN_SEGMENT_SEC = 10.0
MIN_SEGMENT_SAMPLES = int(
    MIN_SEGMENT_SEC * FS
)


# ============================================================
# FUNCIONES
# ============================================================

def get_channels(eeg):

    channels = [
        c for c in eeg.columns
        if c.startswith("chann_")
    ]

    if len(channels) != 16:
        raise ValueError(
            f"Se esperaban 16 canales y hay "
            f"{len(channels)}"
        )

    return channels


def get_valid_segments(x):
    """
    Encuentra tramos consecutivos sin NaN.
    """

    valid = np.isfinite(x)

    padded = np.r_[
        False,
        valid,
        False,
    ]

    changes = np.diff(
        padded.astype(int)
    )

    starts = np.where(
        changes == 1
    )[0]

    ends = np.where(
        changes == -1
    )[0]

    return list(
        zip(starts, ends)
    )


def filter_segment(x):
    """
    Aplica notch + band-pass a un tramo válido.
    """

    data = x[np.newaxis, :]

    # Notch 50 Hz
    data = mne.filter.notch_filter(
        data,
        Fs=FS,
        freqs=[NOTCH_FREQ],
        verbose=False,
    )

    # Band-pass 0.5 - 45 Hz
    data = mne.filter.filter_data(
        data,
        sfreq=FS,
        l_freq=LOW_FREQ,
        h_freq=HIGH_FREQ,
        verbose=False,
    )

    return data[0]


def filter_channel(x):
    """
    Filtra cada tramo continuo de un canal.

    Los NaN originales se conservan.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    output = np.full(
        len(x),
        np.nan,
    )

    segments = get_valid_segments(x)

    n_filtered = 0
    n_short = 0

    filtered_samples = 0
    discarded_samples = 0

    for start, end in segments:

        n_samples = end - start

        # No filtramos segmentos demasiado cortos
        if n_samples < MIN_SEGMENT_SAMPLES:

            n_short += 1

            discarded_samples += (
                n_samples
            )

            continue

        segment = x[start:end]

        output[start:end] = (
            filter_segment(segment)
        )

        n_filtered += 1

        filtered_samples += (
            n_samples
        )

    return (
        output,
        len(segments),
        n_filtered,
        n_short,
        filtered_samples,
        discarded_samples,
    )


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    # Borrar resultados anteriores
    if OUTPUT_DIR.exists():
        shutil.rmtree(
            OUTPUT_DIR
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = []

    participant_dirs = sorted(
        INPUT_DIR.glob(
            "session_*/participant_*"
        )
    )

    for participant_dir in participant_dirs:

        eeg_path = (
            participant_dir
            / EEG_FILE
        )

        if not eeg_path.exists():
            continue

        session = (
            participant_dir.parent.name
        )

        participant = (
            participant_dir.name
        )

        print(
            f"\nProcesando "
            f"{session}/{participant}"
        )

        eeg = pd.read_csv(
            eeg_path
        )

        channels = get_channels(
            eeg
        )

        eeg_filtered = eeg.copy()

        total_segments = 0
        total_filtered = 0
        total_short = 0

        filtered_samples = 0
        discarded_samples = 0

        permanent_bad_channels = []

        # ====================================================
        # FILTRAR CANAL POR CANAL
        # ====================================================

        for channel in channels:

            x = pd.to_numeric(
                eeg[channel],
                errors="coerce",
            ).to_numpy(dtype=float)

            # Canal completamente rechazado por el script 01
            if not np.isfinite(x).any():

                eeg_filtered[
                    channel
                ] = np.nan

                permanent_bad_channels.append(
                    channel
                )

                continue

            (
                filtered,
                n_segments,
                n_filtered,
                n_short,
                n_filtered_samples,
                n_discarded_samples,
            ) = filter_channel(x)

            eeg_filtered[
                channel
            ] = filtered

            total_segments += (
                n_segments
            )

            total_filtered += (
                n_filtered
            )

            total_short += (
                n_short
            )

            filtered_samples += (
                n_filtered_samples
            )

            discarded_samples += (
                n_discarded_samples
            )

        # ====================================================
        # GUARDAR
        # ====================================================

        output_dir = (
            OUTPUT_DIR
            / session
            / participant
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        eeg_filtered.to_csv(
            output_dir / EEG_FILE,
            index=False,
        )

        # Copiar ACC, BandPower y markers
        for filename in AUX_FILES:

            source = (
                participant_dir
                / filename
            )

            if source.exists():

                shutil.copy2(
                    source,
                    output_dir / filename,
                )

        # ====================================================
        # RESUMEN
        # ====================================================

        input_data = (
            eeg[channels]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .to_numpy(dtype=float)
        )

        output_data = (
            eeg_filtered[channels]
            .to_numpy(dtype=float)
        )

        nan_input = (
            100
            * np.mean(
                ~np.isfinite(input_data)
            )
        )

        nan_output = (
            100
            * np.mean(
                ~np.isfinite(output_data)
            )
        )

        summary.append({

            "session":
                session,

            "participant":
                participant,

            "fs_used":
                FS,

            "n_channels":
                len(channels),

            "n_permanent_bad_channels":
                len(
                    permanent_bad_channels
                ),

            "permanent_bad_channels":
                ",".join(
                    permanent_bad_channels
                ),

            "n_valid_segments":
                total_segments,

            "n_filtered_segments":
                total_filtered,

            "n_short_segments":
                total_short,

            "filtered_seconds":
                filtered_samples / FS,

            "discarded_short_seconds":
                discarded_samples / FS,

            "nan_percent_before_filter":
                nan_input,

            "nan_percent_after_filter":
                nan_output,
        })

        print(
            f"  fs: {FS:.1f} Hz"
        )

        print(
            f"  Canales completamente NaN: "
            f"{len(permanent_bad_channels)}"
        )

        print(
            f"  Segmentos filtrados: "
            f"{total_filtered}"
        )

        print(
            f"  Segmentos cortos descartados: "
            f"{total_short}"
        )

        print(
            f"  NaN antes: {nan_input:.1f}% | "
            f"después: {nan_output:.1f}%"
        )

    # ========================================================
    # RESUMEN FINAL
    # ========================================================

    summary_df = pd.DataFrame(
        summary
    )

    summary_df.to_csv(
        RESULTS_DIR
        / "05_filter_summary.csv",
        index=False,
    )

    print(
        "\nFILTRADO TERMINADO\n"
    )

    if len(summary_df) > 0:

        print(
            summary_df.to_string(
                index=False
            )
        )

    else:

        print(
            "No se encontraron datos "
            "en data_cleaned."
        )