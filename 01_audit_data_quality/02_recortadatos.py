from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
TRIMMED_DIR = ROOT_DIR / "data_trimmed"
RESULTS_DIR = ROOT_DIR / "results" / "results_01"

TRIMMED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FILES_TO_TRIM = [
    "obci_eeg1_data.csv",
    "obci_eeg2_data.csv",
    "obci_eeg3_data.csv",
]

MARKERS_FILE = "psychopymarkers_data.csv"


# ============================================================
# FUNCIÓN DE RECORTE
# ============================================================

def trim_dataframe(df, start_ts, end_ts):

    if "timestamp" not in df.columns:
        raise ValueError(
            "El archivo no tiene columna 'timestamp'."
        )

    timestamps = pd.to_numeric(
        df["timestamp"],
        errors="coerce"
    )

    return df.loc[
        timestamps.between(start_ts, end_ts)
    ].copy()


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    summary = []

    for participant_dir in sorted(
        DATA_DIR.glob("session_*/participant_*")
    ):

        session = participant_dir.parent.name
        participant = participant_dir.name

        print(f"\nProcesando {session}/{participant}")

        markers_path = participant_dir / MARKERS_FILE

        if not markers_path.exists():
            print("  Saltado: faltan markers")
            continue

        # ----------------------------------------------------
        # EL CSV DE MARKERS DEFINE EL INICIO Y EL FINAL
        # ----------------------------------------------------

        markers = pd.read_csv(markers_path)

        marker_ts = pd.to_numeric(
            markers["timestamp"],
            errors="coerce"
        ).dropna()

        if marker_ts.empty:
            print("  Saltado: no hay timestamps válidos")
            continue

        start_ts = marker_ts.min()
        end_ts = marker_ts.max()

        print(f"  Inicio prueba: {start_ts:.6f}")
        print(f"  Final prueba:  {end_ts:.6f}")

        output_dir = (
            TRIMMED_DIR
            / session
            / participant
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        result = {
            "session": session,
            "participant": participant,
            "start_ts": start_ts,
            "end_ts": end_ts,
        }

        # ----------------------------------------------------
        # RECORTAR LOS CSV DE DATOS
        # ----------------------------------------------------

        for filename in FILES_TO_TRIM:

            input_path = participant_dir / filename

            if not input_path.exists():
                result[f"{filename}_rows"] = "missing"
                continue

            df = pd.read_csv(input_path)

            trimmed = trim_dataframe(
                df,
                start_ts,
                end_ts
            )

            trimmed.to_csv(
                output_dir / filename,
                index=False
            )

            result[f"{filename}_rows_before"] = len(df)
            result[f"{filename}_rows_after"] = len(trimmed)

        # ----------------------------------------------------
        # LOS MARKERS SE COPIAN TAL CUAL
        # ----------------------------------------------------

        markers.to_csv(
            output_dir / MARKERS_FILE,
            index=False
        )

        summary.append(result)

    # ========================================================
    # RESUMEN
    # ========================================================

    summary_df = pd.DataFrame(summary)

    output_summary = (
        RESULTS_DIR
        / "02_trimmed_data_summary.csv"
    )

    summary_df.to_csv(
        output_summary,
        index=False
    )

    print("\nRECORTE TERMINADO")
    print(summary_df.to_string(index=False))

    print(f"\nDatos recortados: {TRIMMED_DIR}")
    print(f"Resumen: {output_summary}")