from pathlib import Path
import shutil

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "data_rereferenced"
OUTPUT_DIR = ROOT / "data_epoched"
RESULTS_DIR = ROOT / "results" / "results_03"

FS = 125
WINDOW_SEC = 4
STEP_SEC = 2

MAX_NAN_EPOCH_PERCENT = 20.0

EEG_FILE = "obci_eeg1_data.csv"

WINDOW = int(WINDOW_SEC * FS)   # 500 muestras
STEP = int(STEP_SEC * FS)       # 250 muestras


# ============================================================
# PREPARAR SALIDA
# ============================================================

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir(parents=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

summary = []


# ============================================================
# RECORRER LOS 5 MÉTODOS
# ============================================================

for eeg_path in sorted(
    INPUT_DIR.glob(
        "*/session_*/participant_*/obci_eeg1_data.csv"
    )
):

    method = eeg_path.parents[2].name
    session = eeg_path.parents[1].name
    participant = eeg_path.parent.name

    print(
        f"\n{method} - {session}/{participant}"
    )

    df = pd.read_csv(eeg_path)

    # --------------------------------------------------------
    # Detectar canales EEG
    # --------------------------------------------------------

    eeg_cols = [
        c for c in df.columns
        if c.startswith("chann_")
        or c.startswith("bp_")
    ]

    data = (
        df[eeg_cols]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )

    kept_epochs = []
    epoch_info = []

    epoch_id = 0

    # ========================================================
    # EPOCHING
    # ========================================================

    for start in range(
        0,
        len(data) - WINDOW + 1,
        STEP,
    ):

        end = start + WINDOW

        epoch = data[start:end]

        nan_percent = (
            100
            * np.mean(~np.isfinite(epoch))
        )

        keep = (
            nan_percent
            <= MAX_NAN_EPOCH_PERCENT
        )

        epoch_info.append({
            "epoch": epoch_id,
            "start_sample": start,
            "end_sample": end,
            "start_sec": start / FS,
            "end_sec": end / FS,
            "nan_percent": nan_percent,
            "keep": keep,
        })

        if keep:

            # Guardar como:
            # canales x muestras
            kept_epochs.append(
                epoch.T
            )

        epoch_id += 1


    # ========================================================
    # GUARDAR
    # ========================================================

    output_dir = (
        OUTPUT_DIR
        / method
        / session
        / participant
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    info = pd.DataFrame(epoch_info)

    info.to_csv(
        output_dir / "epoch_info.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Guardar épocas aceptadas
    # --------------------------------------------------------

    if kept_epochs:

        epochs = np.stack(
            kept_epochs
        ).astype(np.float32)

    else:

        epochs = np.empty(
            (
                0,
                len(eeg_cols),
                WINDOW,
            ),
            dtype=np.float32,
        )

    np.savez_compressed(
        output_dir / "epochs.npz",
        epochs=epochs,
        channels=np.array(eeg_cols),
    )


    # ========================================================
    # RESUMEN
    # ========================================================

    n_total = len(info)
    n_kept = int(info["keep"].sum())
    n_rejected = n_total - n_kept

    kept_percent = (
        100 * n_kept / n_total
        if n_total
        else np.nan
    )

    summary.append({
        "method": method,
        "session": session,
        "participant": participant,
        "n_channels": len(eeg_cols),
        "n_total_epochs": n_total,
        "n_kept_epochs": n_kept,
        "n_rejected_epochs": n_rejected,
        "kept_percent": kept_percent,
    })

    print(
        f"  Canales: {len(eeg_cols)} | "
        f"Épocas: {n_kept}/{n_total} "
        f"({kept_percent:.1f}%)"
    )


# ============================================================
# RESUMEN GLOBAL
# ============================================================

summary = pd.DataFrame(summary)

summary.to_csv(
    RESULTS_DIR / "01_epoch_summary.csv",
    index=False,
)

print(
    "\nEPOCHING TERMINADO\n"
)

print(
    summary.to_string(index=False)
)