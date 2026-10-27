from pathlib import Path
import shutil

import numpy as np
import pandas as pd


"""
03 - Re-referenciación Ear-EEG

Genera cinco métodos:

1. all_mean
2. contralateral_mean
3. ipsilateral_mean
4. contralateral_bipolar
5. ipsilateral_bipolar

Entrada:
    data_filtered

Salida:
    data_rereferenced/

Los NaN se consideran datos no válidos y nunca participan
en el cálculo de la referencia.
"""


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "data_ica_clean"

OUTPUT_DIR = ROOT / "data_rereferenced"

RESULTS_DIR = (
    ROOT / "results" / "results_02"
)

QC_DIR = RESULTS_DIR

EEG_FILE = "obci_eeg1_data.csv"

AUX_FILES = [
    "obci_eeg2_data.csv",
    "obci_eeg3_data.csv",
    "psychopymarkers_data.csv",
]


# ============================================================
# CANALES
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

LEFT = [
    "chann_1",
    "chann_2",
    "chann_3",
    "chann_4",
    "chann_5",
    "chann_6",
    "chann_7",
    "chann_8",
]

RIGHT = [
    "chann_9",
    "chann_10",
    "chann_11",
    "chann_12",
    "chann_13",
    "chann_14",
    "chann_15",
    "chann_16",
]

CHANNELS = LEFT + RIGHT


# ============================================================
# NÚMERO MÍNIMO DE ELECTRODOS PARA CALCULAR MEDIA
# ============================================================

# CAR:
# como máximo permitimos que falten 4 de 16
MIN_ALL_REFERENCE_CHANNELS = 12

# Referencia de un oído:
# como máximo permitimos que falten 2 de 8
MIN_EAR_REFERENCE_CHANNELS = 6


METHODS = [
    "all_mean",
    "contralateral_mean",
    "ipsilateral_mean",
    "contralateral_bipolar",
    "ipsilateral_bipolar",
]


# ============================================================
# UTILIDADES
# ============================================================

def safe_mean_reference(
    data,
    minimum_channels,
):
    """
    Calcula la media ignorando NaN.

    Si no hay suficientes canales válidos,
    devuelve NaN para esa muestra.
    """

    valid = np.isfinite(data)

    count = valid.sum(axis=1)

    total = np.nansum(
        data,
        axis=1,
    )

    reference = np.full(
        len(data),
        np.nan,
        dtype=float,
    )

    good = (
        count >= minimum_channels
    )

    reference[good] = (
        total[good]
        / count[good]
    )

    return reference


def get_permanent_bad(
    channel_qc,
    session,
    participant,
):
    """
    Obtiene los canales permanentemente malos
    detectados por el script 01.
    """

    if channel_qc.empty:
        return []

    rows = channel_qc[
        (channel_qc["session"] == session)
        & (
            channel_qc["participant"]
            == participant
        )
    ]

    rows = rows[
        rows["permanent_bad"] == True
    ]

    return (
        rows["channel"]
        .astype(str)
        .tolist()
    )


def is_session_rejected(
    session_qc,
    session,
    participant,
):

    if session_qc.empty:
        return False

    rows = session_qc[
        (session_qc["session"] == session)
        & (
            session_qc["participant"]
            == participant
        )
    ]

    if len(rows) == 0:
        return False

    value = rows.iloc[0][
        "session_rejected"
    ]

    if isinstance(value, str):

        return (
            value.lower()
            == "true"
        )

    return bool(value)


# ============================================================
# 1. ALL-MEAN
# ============================================================

def rereference_all_mean(data):

    reference = safe_mean_reference(
        data,
        MIN_ALL_REFERENCE_CHANNELS,
    )

    output = (
        data
        - reference[:, None]
    )

    return output


# ============================================================
# 2. CONTRALATERAL-MEAN
# ============================================================

def rereference_contralateral_mean(
    data,
):

    output = np.full_like(
        data,
        np.nan,
    )

    left_data = data[:, :8]
    right_data = data[:, 8:]

    # Para canales izquierdos:
    # referencia = media oído derecho
    right_reference = (
        safe_mean_reference(
            right_data,
            MIN_EAR_REFERENCE_CHANNELS,
        )
    )

    output[:, :8] = (
        left_data
        - right_reference[:, None]
    )

    # Para canales derechos:
    # referencia = media oído izquierdo
    left_reference = (
        safe_mean_reference(
            left_data,
            MIN_EAR_REFERENCE_CHANNELS,
        )
    )

    output[:, 8:] = (
        right_data
        - left_reference[:, None]
    )

    return output


# ============================================================
# 3. IPSILATERAL-MEAN
# ============================================================

def rereference_ipsilateral_mean(
    data,
):

    output = np.full_like(
        data,
        np.nan,
    )

    left_data = data[:, :8]
    right_data = data[:, 8:]

    left_reference = (
        safe_mean_reference(
            left_data,
            MIN_EAR_REFERENCE_CHANNELS,
        )
    )

    right_reference = (
        safe_mean_reference(
            right_data,
            MIN_EAR_REFERENCE_CHANNELS,
        )
    )

    output[:, :8] = (
        left_data
        - left_reference[:, None]
    )

    output[:, 8:] = (
        right_data
        - right_reference[:, None]
    )

    return output


# ============================================================
# 4. CONTRALATERAL-BIPOLAR
# ============================================================

def rereference_contralateral_bipolar(
    data,
):

    output = {}
    columns = {}

    for i, left_ch in enumerate(LEFT):

        for j, right_ch in enumerate(RIGHT):

            name = (
                f"bp_"
                f"{LABELS[left_ch]}_"
                f"{LABELS[right_ch]}"
            )

            left_signal = data[:, i]

            right_signal = data[
                :,
                j + 8
            ]

            bipolar = (
                left_signal
                - right_signal
            )

            output[name] = bipolar

            columns[name] = (
                f"{LABELS[left_ch]}"
                f"-"
                f"{LABELS[right_ch]}"
            )

    return (
        pd.DataFrame(output),
        columns,
    )


# ============================================================
# 5. IPSILATERAL-BIPOLAR
# ============================================================

def rereference_ipsilateral_bipolar(
    data,
):

    output = {}
    columns = {}

    # --------------------------------------------------------
    # Oído izquierdo
    # --------------------------------------------------------

    for i in range(len(LEFT)):

        for j in range(
            i + 1,
            len(LEFT),
        ):

            ch1 = LEFT[i]
            ch2 = LEFT[j]

            name = (
                f"bp_"
                f"{LABELS[ch1]}_"
                f"{LABELS[ch2]}"
            )

            output[name] = (
                data[:, i]
                - data[:, j]
            )

            columns[name] = (
                f"{LABELS[ch1]}"
                f"-"
                f"{LABELS[ch2]}"
            )

    # --------------------------------------------------------
    # Oído derecho
    # --------------------------------------------------------

    for i in range(len(RIGHT)):

        for j in range(
            i + 1,
            len(RIGHT),
        ):

            ch1 = RIGHT[i]
            ch2 = RIGHT[j]

            idx1 = i + 8
            idx2 = j + 8

            name = (
                f"bp_"
                f"{LABELS[ch1]}_"
                f"{LABELS[ch2]}"
            )

            output[name] = (
                data[:, idx1]
                - data[:, idx2]
            )

            columns[name] = (
                f"{LABELS[ch1]}"
                f"-"
                f"{LABELS[ch2]}"
            )

    return (
        pd.DataFrame(output),
        columns,
    )


# ============================================================
# GUARDAR MÉTODO
# ============================================================

def save_output(
    method,
    eeg_original,
    eeg_reref,
    participant_dir,
    session,
    participant,
):

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

    # Conservar timestamp y cualquier otra
    # columna que no sea EEG
    metadata_columns = [
        c for c in eeg_original.columns
        if c not in CHANNELS
    ]

    output = eeg_original[
        metadata_columns
    ].copy()

    for column in eeg_reref.columns:

        output[column] = (
            eeg_reref[column]
            .to_numpy()
        )

    output.to_csv(
        output_dir / EEG_FILE,
        index=False,
    )

    # Copiar archivos auxiliares
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


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Resultados QC anteriores
    # --------------------------------------------------------

    channel_qc_path = (
        QC_DIR
        / "03_channel_rejection.csv"
    )

    session_qc_path = (
        QC_DIR
        / "04_session_rejection.csv"
    )

    channel_qc = (
        pd.read_csv(channel_qc_path)
        if channel_qc_path.exists()
        else pd.DataFrame()
    )

    session_qc = (
        pd.read_csv(session_qc_path)
        if session_qc_path.exists()
        else pd.DataFrame()
    )

    # --------------------------------------------------------
    # Limpiar salida anterior
    # --------------------------------------------------------

    if OUTPUT_DIR.exists():

        shutil.rmtree(
            OUTPUT_DIR
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = []

    participant_dirs = sorted(
        INPUT_DIR.glob(
            "session_*/participant_*"
        )
    )

    # ========================================================
    # SESIONES
    # ========================================================

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

        # ----------------------------------------------------
        # ¿Sesión rechazada?
        # ----------------------------------------------------

        if is_session_rejected(
            session_qc,
            session,
            participant,
        ):

            print(
                "  Sesión rechazada por QC. "
                "No se re-referencia."
            )

            continue

        eeg = pd.read_csv(
            eeg_path
        )

        missing = [
            ch for ch in CHANNELS
            if ch not in eeg.columns
        ]

        if missing:

            raise ValueError(
                f"Faltan canales en "
                f"{session}/{participant}: "
                f"{missing}"
            )

        # ----------------------------------------------------
        # Datos EEG
        # ----------------------------------------------------

        data = (
            eeg[CHANNELS]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .to_numpy(dtype=float)
        )

        # ----------------------------------------------------
        # Canales permanentemente malos
        # ----------------------------------------------------

        permanent_bad = (
            get_permanent_bad(
                channel_qc,
                session,
                participant,
            )
        )

        # Los hacemos NaN explícitamente
        for ch in permanent_bad:

            if ch in CHANNELS:

                idx = CHANNELS.index(ch)

                data[:, idx] = np.nan

        print(
            f"  Canales permanentes excluidos: "
            f"{len(permanent_bad)}"
        )

        # ====================================================
        # 1. ALL-MEAN
        # ====================================================

        all_mean = (
            rereference_all_mean(
                data
            )
        )

        all_mean_df = pd.DataFrame(
            all_mean,
            columns=CHANNELS,
        )

        save_output(
            "all_mean",
            eeg,
            all_mean_df,
            participant_dir,
            session,
            participant,
        )

        # ====================================================
        # 2. CONTRALATERAL-MEAN
        # ====================================================

        contra_mean = (
            rereference_contralateral_mean(
                data
            )
        )

        contra_mean_df = pd.DataFrame(
            contra_mean,
            columns=CHANNELS,
        )

        save_output(
            "contralateral_mean",
            eeg,
            contra_mean_df,
            participant_dir,
            session,
            participant,
        )

        # ====================================================
        # 3. IPSILATERAL-MEAN
        # ====================================================

        ipsi_mean = (
            rereference_ipsilateral_mean(
                data
            )
        )

        ipsi_mean_df = pd.DataFrame(
            ipsi_mean,
            columns=CHANNELS,
        )

        save_output(
            "ipsilateral_mean",
            eeg,
            ipsi_mean_df,
            participant_dir,
            session,
            participant,
        )

        # ====================================================
        # 4. CONTRALATERAL-BIPOLAR
        # ====================================================

        contra_bp_df, _ = (
            rereference_contralateral_bipolar(
                data
            )
        )

        save_output(
            "contralateral_bipolar",
            eeg,
            contra_bp_df,
            participant_dir,
            session,
            participant,
        )

        # ====================================================
        # 5. IPSILATERAL-BIPOLAR
        # ====================================================

        ipsi_bp_df, _ = (
            rereference_ipsilateral_bipolar(
                data
            )
        )

        save_output(
            "ipsilateral_bipolar",
            eeg,
            ipsi_bp_df,
            participant_dir,
            session,
            participant,
        )

        # ====================================================
        # RESUMEN
        # ====================================================

        method_outputs = {
            "all_mean":
                all_mean_df,

            "contralateral_mean":
                contra_mean_df,

            "ipsilateral_mean":
                ipsi_mean_df,

            "contralateral_bipolar":
                contra_bp_df,

            "ipsilateral_bipolar":
                ipsi_bp_df,
        }

        for method, df in method_outputs.items():

            values = df.to_numpy(
                dtype=float
            )

            nan_percent = (
                100
                * np.mean(
                    ~np.isfinite(values)
                )
            )

            summary.append({

                "session":
                    session,

                "participant":
                    participant,

                "method":
                    method,

                "n_output_channels":
                    df.shape[1],

                "n_permanent_bad":
                    len(permanent_bad),

                "permanent_bad_channels":
                    ",".join(
                        LABELS.get(ch, ch)
                        for ch in permanent_bad
                    ),

                "nan_percent":
                    nan_percent,
            })

            print(
                f"  {method}: "
                f"{df.shape[1]} canales | "
                f"NaN {nan_percent:.1f}%"
            )

    # ========================================================
    # GUARDAR RESUMEN
    # ========================================================

    summary_df = pd.DataFrame(
        summary
    )

    summary_df.to_csv(
        RESULTS_DIR
        / "06_rereference_summary.csv",
        index=False,
    )

    print(
        "\nRE-REFERENCIACIÓN TERMINADA\n"
    )

    if len(summary_df):

        print(
            summary_df.to_string(
                index=False
            )
        )