from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt


# Configuración

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data_trimmed"
RESULTS_DIR = ROOT_DIR / "results" / "results_01"
PSD_DIR = RESULTS_DIR / "plots_psd"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PSD_DIR.mkdir(parents=True, exist_ok=True)

EEG_FILE = "obci_eeg1_data.csv"

FS_MIN = 120
FS_MAX = 130
LARGE_GAP_S = 0.2


# PSD

def save_psd_plot(eeg, channels, fs, session, participant):

    psds = []
    freqs = None

    nperseg = min(int(4 * fs), len(eeg))

    for channel in channels:

        signal = pd.to_numeric(
            eeg[channel],
            errors="coerce"
        ).interpolate(limit_direction="both")

        if signal.isna().any() or signal.std() == 0:
            continue

        f, psd = welch(
            signal - signal.median(),
            fs=fs,
            nperseg=nperseg
        )

        freqs = f
        psds.append(psd)

    if not psds:
        return

    median_psd = np.median(
        np.array(psds),
        axis=0
    )

    mask = (freqs >= 1) & (freqs <= 60)

    plt.figure(figsize=(8, 5))
    plt.semilogy(
        freqs[mask],
        median_psd[mask]
    )

    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("PSD")
    plt.title(f"{session} - {participant}")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        PSD_DIR / f"{session}_{participant}_psd.png",
        dpi=150
    )

    plt.close()


# Ejecución

if __name__ == "__main__":

    results = []

    for participant_dir in sorted(
        DATA_DIR.glob("session_*/participant_*")
    ):

        session = participant_dir.parent.name
        participant = participant_dir.name

        print(f"Analizando {session}/{participant}")

        eeg_path = participant_dir / EEG_FILE

        if not eeg_path.exists():
            print("  No hay archivo EEG")
            continue

        eeg = pd.read_csv(eeg_path)

        channels = [
            column
            for column in eeg.columns
            if column.startswith("chann_")
        ]

        timestamps = pd.to_numeric(
            eeg["timestamp"],
            errors="coerce"
        ).dropna().to_numpy()

        duration = timestamps[-1] - timestamps[0]

        fs = (
            (len(timestamps) - 1) / duration
            if duration > 0
            else np.nan
        )

        gaps = np.diff(timestamps)
        large_gaps = gaps[gaps > LARGE_GAP_S]

        eeg_data = eeg[channels].apply(
            pd.to_numeric,
            errors="coerce"
        )

        nan_percent = (
            eeg_data.isna().sum().sum()
            / eeg_data.size
            * 100
        )

        flat_channels = []

        for channel in channels:

            signal = eeg_data[channel].dropna()

            if len(signal) == 0 or signal.std() < 1e-12:
                flat_channels.append(channel)

        problems = []

        if not FS_MIN <= fs <= FS_MAX:
            problems.append("frecuencia de muestreo")

        if len(large_gaps) > 0:
            problems.append("gaps grandes")

        if nan_percent > 0:
            problems.append("NaNs")

        if flat_channels:
            problems.append("canales planos")

        results.append({
            "session": session,
            "participant": participant,
            "status": "OK" if not problems else "REVIEW",
            "duration_min": duration / 60,
            "n_samples": len(eeg),
            "n_channels": len(channels),
            "fs_effective_hz": fs,
            "n_large_gaps": len(large_gaps),
            "max_gap_s": (
                large_gaps.max()
                if len(large_gaps) > 0
                else 0
            ),
            "nan_percent": nan_percent,
            "n_flat_channels": len(flat_channels),
            "flat_channels": ", ".join(flat_channels),
            "problems": "; ".join(problems),
        })

        save_psd_plot(
            eeg,
            channels,
            fs,
            session,
            participant
        )

    results_df = pd.DataFrame(results)

    output_path = (
        RESULTS_DIR
        / "03_eeg_quality_summary.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\nRESULTADOS")
    print(results_df.to_string(index=False))

    print(f"\nResumen: {output_path}")
    print(f"PSD: {PSD_DIR}")