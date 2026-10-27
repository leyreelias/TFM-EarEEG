from pathlib import Path
import pandas as pd
import numpy as np

# Configuración

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results" / "results_01"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EEG_FILE = "obci_eeg1_data.csv"
MARKERS_FILE = "psychopymarkers_data.csv"

EXPECTED_COMMON_MARKERS = [
    "eeg_calibration_started",
    "eeg_calibration_end",
    "EXPERIMENT_START",
    "EXPERIMENT_END",
    "1_BACK_START",
    "1_BACK_END",
    "2_BACK_START",
    "2_BACK_END",
    "3_BACK_START",
    "3_BACK_END",
]

EXPECTED_OPENMATB_MARKERS = [
    "PHASE_NO_OPENMATB_START",
    "PHASE_NO_OPENMATB_END",
    "PHASE_OPENMATB_START",
    "PHASE_OPENMATB_END",
    "OPENMATB_START",
    "OPENMATB_END",
]


# Comprobar una sesión

def check_session(participant_dir):

    session = participant_dir.parent.name
    participant = participant_dir.name

    task = "nback" if participant in ["participant_01", "participant_1"] else "nback_openmatb"

    eeg_path = participant_dir / EEG_FILE
    markers_path = participant_dir / MARKERS_FILE

    if not eeg_path.exists() or not markers_path.exists():
        return {
            "session": session,
            "participant": participant,
            "task": task,
            "status": "ERROR",
            "problems": "Falta EEG o markers",
        }

    eeg = pd.read_csv(eeg_path)
    markers = pd.read_csv(markers_path)

    if "timestamp" not in eeg.columns:
        raise ValueError(f"{session}/{participant}: EEG sin timestamp")

    if "timestamp" not in markers.columns or "marker" not in markers.columns:
        raise ValueError(f"{session}/{participant}: formato de markers incorrecto")

    eeg_ts = pd.to_numeric(eeg["timestamp"], errors="coerce").dropna().to_numpy()
    marker_ts = pd.to_numeric(markers["timestamp"], errors="coerce").dropna().to_numpy()

    eeg_start, eeg_end = eeg_ts.min(), eeg_ts.max()
    marker_start, marker_end = marker_ts.min(), marker_ts.max()

    eeg_diff = np.diff(eeg_ts)
    marker_diff = np.diff(marker_ts)

    markers_before_eeg = int(np.sum(marker_ts < eeg_start))
    markers_after_eeg = int(np.sum(marker_ts > eeg_end))

    eeg_out_of_order = int(np.sum(eeg_diff < 0))
    markers_out_of_order = int(np.sum(marker_diff < 0))

    eeg_duplicate_ts = int(np.sum(eeg_diff == 0))
    marker_duplicate_ts = int(np.sum(marker_diff == 0))

    existing_markers = set(markers["marker"].dropna().astype(str))

    expected_markers = EXPECTED_COMMON_MARKERS.copy()

    if task == "nback_openmatb":
        expected_markers += EXPECTED_OPENMATB_MARKERS

    missing_markers = [
        marker for marker in expected_markers
        if marker not in existing_markers
    ]

    problems = []

    if markers_before_eeg or markers_after_eeg:
        problems.append("markers fuera del EEG")

    if eeg_out_of_order:
        problems.append("EEG desordenado")

    if markers_out_of_order:
        problems.append("markers desordenados")

    if missing_markers:
        problems.append("faltan markers")

    return {
        "session": session,
        "participant": participant,
        "task": task,
        "status": "OK" if not problems else "REVIEW",
        "eeg_duration_min": (eeg_end - eeg_start) / 60,
        "marker_duration_min": (marker_end - marker_start) / 60,
        "markers_before_eeg": markers_before_eeg,
        "markers_after_eeg": markers_after_eeg,
        "eeg_out_of_order": eeg_out_of_order,
        "markers_out_of_order": markers_out_of_order,
        "eeg_duplicate_timestamps": eeg_duplicate_ts,
        "marker_duplicate_timestamps": marker_duplicate_ts,
        "missing_markers": ", ".join(missing_markers),
        "problems": "; ".join(problems),
    }


# Ejecución

if __name__ == "__main__":

    results = []

    for participant_dir in sorted(DATA_DIR.glob("session_*/participant_*")):
        print(f"Comprobando {participant_dir.parent.name}/{participant_dir.name}")
        results.append(check_session(participant_dir))

    results_df = pd.DataFrame(results)

    output_path = RESULTS_DIR / "01_timestamp_marker_check.csv"
    results_df.to_csv(output_path, index=False)

    print("\nRESULTADOS")
    print(results_df.to_string(index=False))
    print(f"\nGuardado en: {output_path}")