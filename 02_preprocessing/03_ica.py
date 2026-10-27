from pathlib import Path
import shutil

import mne
import numpy as np
import pandas as pd

from eeg_positions import get_elec_coords
from mne.preprocessing import ICA
from mne_icalabel.iclabel import iclabel_label_components


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "data_filtered"
OUTPUT_DIR = ROOT / "data_ica_clean"
RESULTS_DIR = ROOT / "results" / "results_02" / "ica"

EEG_FILE = "obci_eeg1_data.csv"

AUX_FILES = [
    "obci_eeg2_data.csv",
    "obci_eeg3_data.csv",
    "psychopymarkers_data.csv",
]

FS = 125.0

# Para ajustar ICA
ICA_HIGHPASS = 1.0
MIN_FIT_SEGMENT_SEC = 10.0

# ICLabel
ARTIFACT_THRESHOLD = 0.80

ARTIFACT_CLASSES = {
    "muscle artifact",
    "eye blink",
    "heart beat",
    "line noise",
    "channel noise",
}

ICLABEL_CLASSES = [
    "brain",
    "muscle artifact",
    "eye blink",
    "heart beat",
    "line noise",
    "channel noise",
    "other",
]

RANDOM_STATE = 97


# ============================================================
# CANALES
# ============================================================

CHANNELS = [
    f"chann_{i}"
    for i in range(1, 17)
]

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


# ============================================================
# EQUIVALENCIA CON LA NOMENCLATURA cEEGrid CLÁSICA
# ============================================================

# En el conector de 10 posiciones:
#
# 1  -> 01
# 2  -> 02
# 3  -> 03
# 4  -> 04
# 5  -> 04a
# 6  -> 04b
# 7  -> 05
# 8  -> 06
# 9  -> 07
# 10 -> 08
#
# L3/R3 no se adquirieron.
# L6 = REF y R6 = GND.

CEEGRID_POSITION = {
    "L1":  "ceegrid_L01",
    "L2":  "ceegrid_L02",
    "L4":  "ceegrid_L04",
    "L5":  "ceegrid_L04a",
    "L7":  "ceegrid_L05",
    "L8":  "ceegrid_L06",
    "L9":  "ceegrid_L07",
    "L10": "ceegrid_L08",

    "R1":  "ceegrid_R01",
    "R2":  "ceegrid_R02",
    "R4":  "ceegrid_R04",
    "R5":  "ceegrid_R04a",
    "R7":  "ceegrid_R05",
    "R8":  "ceegrid_R06",
    "R9":  "ceegrid_R07",
    "R10": "ceegrid_R08",
}


# ============================================================
# FUNCIONES
# ============================================================

def get_segments(mask):
    """Segmentos consecutivos donde mask == True."""

    changes = np.diff(
        np.r_[False, mask, False].astype(int)
    )

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    return list(zip(starts, ends))


def make_montage(channel_names):
    """
    Montaje cEEGrid idealizado para ICLabel.
    """

    ceegrid_names = [
        CEEGRID_POSITION[ch]
        for ch in channel_names
    ]

    montage = get_elec_coords(
        system="ceegrid",
        elec_names=ceegrid_names,
        as_mne_montage=True,
    )

    montage.rename_channels({
        old: new
        for old, new in zip(
            ceegrid_names,
            channel_names,
        )
    })

    return montage


def make_raw(
    data_uv,
    channel_names,
    montage,
    highpass=None,
):
    """
    data_uv:
        muestras x canales, en microvoltios.

    MNE necesita voltios.
    """

    info = mne.create_info(
        ch_names=channel_names,
        sfreq=FS,
        ch_types="eeg",
    )

    raw = mne.io.RawArray(
        data_uv.T * 1e-6,
        info,
        verbose=False,
    )

    raw.set_montage(
        montage,
        on_missing="raise",
        verbose=False,
    )

    # ICLabel está diseñado para average reference.
    raw.set_eeg_reference(
        "average",
        projection=False,
        verbose=False,
    )

    if highpass is not None:

        raw.filter(
            l_freq=highpass,
            h_freq=None,
            verbose=False,
        )

    return raw


# ============================================================
# PREPARAR SALIDA
# ============================================================

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

summary = []


# ============================================================
# SESIONES
# ============================================================

for eeg_path in sorted(
    INPUT_DIR.glob(
        "session_*/participant_*/obci_eeg1_data.csv"
    )
):

    session = eeg_path.parents[1].name
    participant = eeg_path.parent.name

    print(
        f"\n{session}/{participant}"
    )

    df = pd.read_csv(eeg_path)

    data = (
        df[CHANNELS]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(dtype=float)
    )

    original_nan = ~np.isfinite(data)

    # ========================================================
    # CANALES PERMANENTEMENTE MALOS
    # ========================================================

    usable = np.any(
        np.isfinite(data),
        axis=0,
    )

    good_cols = [
        ch
        for ch, good
        in zip(CHANNELS, usable)
        if good
    ]

    good_labels = [
        LABELS[ch]
        for ch in good_cols
    ]

    bad_cols = [
        ch
        for ch, good
        in zip(CHANNELS, usable)
        if not good
    ]

    good_idx = np.where(usable)[0]

    good_data = data[:, usable]

    print(
        f"  Canales útiles: "
        f"{len(good_cols)}/16"
    )

    # ========================================================
    # SEGMENTOS SIN NaN PARA AJUSTAR ICA
    # ========================================================

    complete = np.all(
        np.isfinite(good_data),
        axis=1,
    )

    segments = get_segments(
        complete
    )

    min_samples = int(
        MIN_FIT_SEGMENT_SEC * FS
    )

    fit_segments = [
        (start, end)
        for start, end in segments
        if end - start >= min_samples
    ]

    if not fit_segments:

        print(
            "  ERROR: no hay segmentos "
            "suficientemente largos."
        )
        continue

    fit_seconds = sum(
        end - start
        for start, end in fit_segments
    ) / FS

    print(
        f"  Datos válidos para ICA: "
        f"{fit_seconds:.1f} s"
    )

    # ========================================================
    # MONTAGE
    # ========================================================

    montage = make_montage(
        good_labels
    )

    # ========================================================
    # CREAR DATOS PARA ICA
    # ========================================================

    fit_raws = []

    for start, end in fit_segments:

        raw_segment = make_raw(
            good_data[start:end],
            good_labels,
            montage,
            highpass=ICA_HIGHPASS,
        )

        fit_raws.append(
            raw_segment
        )

    raw_fit = mne.concatenate_raws(
        fit_raws,
        verbose=False,
    )

    # ========================================================
    # RANK
    # ========================================================

    rank = int(
        mne.compute_rank(
            raw_fit,
            rank=None,
            tol="auto",
            verbose=False,
        )["eeg"]
    )

    print(
        f"  Rank ICA: {rank}"
    )

    # ========================================================
    # ICA EXTENDED INFOMAX
    # ========================================================

    ica = ICA(
        n_components=rank,
        method="infomax",
        fit_params={
            "extended": True
        },
        random_state=RANDOM_STATE,
        max_iter="auto",
    )

    ica.fit(
        raw_fit,
        verbose=False,
    )

    print(
        f"  Componentes ICA: "
        f"{ica.n_components_}"
    )

    # ========================================================
    # ICLABEL
    # ========================================================

    probabilities = (
        iclabel_label_components(
            raw_fit,
            ica,
            inplace=True,
        )
    )

    component_rows = []
    excluded = []

    for i in range(
        probabilities.shape[0]
    ):

        probs = probabilities[i]

        winner = int(
            np.argmax(probs)
        )

        label = (
            ICLABEL_CLASSES[winner]
        )

        probability = float(
            probs[winner]
        )

        remove = (
            label in ARTIFACT_CLASSES
            and probability
            >= ARTIFACT_THRESHOLD
        )

        if remove:
            excluded.append(i)

        row = {
            "component":
                f"ICA{i:03d}",

            "predicted_label":
                label,

            "predicted_probability":
                probability,

            "excluded":
                remove,
        }

        for name, value in zip(
            ICLABEL_CLASSES,
            probs,
        ):

            clean_name = (
                name
                .replace(" ", "_")
            )

            row[
                f"prob_{clean_name}"
            ] = float(value)

        component_rows.append(row)

    # ========================================================
    # MOSTRAR DECISIÓN
    # ========================================================

    if excluded:

        print(
            f"  Componentes eliminados: "
            f"{len(excluded)}"
        )

        for i in excluded:

            row = component_rows[i]

            print(
                f"    ICA{i:03d}: "
                f"{row['predicted_label']} "
                f"({100 * row['predicted_probability']:.1f}%)"
            )

    else:

        print(
            "  Componentes eliminados: 0"
        )

    # ========================================================
    # APLICAR ICA
    # ========================================================

    # ICA no acepta NaN.
    #
    # Para poder aplicar la matriz ICA a toda la grabación,
    # rellenamos NaN SOLO de forma temporal.
    #
    # Después restauramos exactamente los NaN originales.

    filled = (
        pd.DataFrame(good_data)
        .interpolate(
            method="linear",
            axis=0,
            limit_direction="both",
        )
        .to_numpy()
    )

    if not np.all(
        np.isfinite(filled)
    ):

        raise ValueError(
            f"No se pudieron rellenar NaN "
            f"para {session}/{participant}"
        )

    raw_apply = make_raw(
        filled,
        good_labels,
        montage,
        highpass=None,
    )

    ica.apply(
        raw_apply,
        exclude=excluded,
        verbose=False,
    )

    # Volver a microvoltios
    cleaned_good = (
        raw_apply
        .get_data()
        .T
        * 1e6
    )

    # Restaurar NaN originales
    good_nan_mask = (
        original_nan[:, usable]
    )

    cleaned_good[
        good_nan_mask
    ] = np.nan

    # ========================================================
    # RECONSTRUIR 16 CANALES
    # ========================================================

    cleaned = np.full_like(
        data,
        np.nan,
    )

    cleaned[:, good_idx] = (
        cleaned_good
    )

    # ========================================================
    # GUARDAR EEG
    # ========================================================

    output_dir = (
        OUTPUT_DIR
        / session
        / participant
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df = df.copy()

    output_df[CHANNELS] = (
        cleaned
    )

    output_df.to_csv(
        output_dir / EEG_FILE,
        index=False,
    )

    # ========================================================
    # COPIAR AUXILIARES
    # ========================================================

    for filename in AUX_FILES:

        source = (
            eeg_path.parent
            / filename
        )

        if source.exists():

            shutil.copy2(
                source,
                output_dir / filename,
            )

    # ========================================================
    # GUARDAR RESULTADOS ICA
    # ========================================================

    result_dir = (
        RESULTS_DIR
        / session
        / participant
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        component_rows
    ).to_csv(
        result_dir
        / "ica_components.csv",
        index=False,
    )

    ica.save(
        result_dir / "ica.fif",
        overwrite=True,
    )

    nan_before = (
        100
        * np.mean(
            ~np.isfinite(data)
        )
    )

    nan_after = (
        100
        * np.mean(
            ~np.isfinite(cleaned)
        )
    )

    summary.append({
        "session":
            session,

        "participant":
            participant,

        "n_good_channels":
            len(good_cols),

        "bad_channels":
            ",".join(
                LABELS[ch]
                for ch in bad_cols
            ),

        "fit_seconds":
            fit_seconds,

        "rank":
            rank,

        "n_components":
            ica.n_components_,

        "n_excluded":
            len(excluded),

        "excluded_components":
            ",".join(
                f"ICA{i:03d}"
                for i in excluded
            ),

        "nan_percent_before":
            nan_before,

        "nan_percent_after":
            nan_after,
    })


# ============================================================
# RESUMEN
# ============================================================

summary = pd.DataFrame(
    summary
)

summary.to_csv(
    RESULTS_DIR
    / "01_ica_summary.csv",
    index=False,
)

print(
    "\nICA + ICLABEL TERMINADO\n"
)

print(
    summary.to_string(
        index=False
    )
)