from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "data_rereferenced"
RESULTS_DIR = ROOT / "results" / "results_02"

FS = 125
WINDOW_SEC = 4
STEP_SEC = 2

MAX_NAN_EPOCH_PERCENT = 20.0   # <-- UMBRAL A CAMBIAR

EEG_FILE = "obci_eeg1_data.csv"

WINDOW = int(WINDOW_SEC * FS)
STEP = int(STEP_SEC * FS)


# ============================================================
# EJECUCIÓN
# ============================================================

results = []

for eeg_path in INPUT_DIR.glob(
    "*/session_*/participant_*/obci_eeg1_data.csv"
):

    method = eeg_path.parents[2].name
    session = eeg_path.parents[1].name
    participant = eeg_path.parent.name

    df = pd.read_csv(eeg_path)

    # Solo columnas EEG
    eeg_cols = [
        c for c in df.columns
        if c.startswith("chann_")
        or c.startswith("bp_")
    ]

    data = df[eeg_cols].to_numpy(dtype=float)

    n_epochs = 0
    n_kept = 0

    for start in range(
        0,
        len(data) - WINDOW + 1,
        STEP,
    ):

        epoch = data[
            start:start + WINDOW
        ]

        nan_percent = (
            100
            * np.mean(~np.isfinite(epoch))
        )

        keep = (
            nan_percent
            <= MAX_NAN_EPOCH_PERCENT
        )

        results.append({
            "method": method,
            "session": session,
            "participant": participant,
            "start_sample": start,
            "end_sample": start + WINDOW,
            "nan_percent": nan_percent,
            "keep": keep,
        })

        n_epochs += 1
        n_kept += int(keep)

    print(
        f"{method:25s} "
        f"{session}/{participant}: "
        f"{n_kept}/{n_epochs} épocas válidas "
        f"({100*n_kept/n_epochs:.1f}%)"
    )


# ============================================================
# GUARDAR
# ============================================================

results = pd.DataFrame(results)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

results.to_csv(
    RESULTS_DIR / "07_epoch_quality.csv",
    index=False,
)

print("\nTERMINADO - umbral 20")