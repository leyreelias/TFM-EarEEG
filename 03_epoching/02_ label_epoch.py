"""Etiqueta las epocas y vincula sus bloques BEDFORD en una sola ejecucion."""
from pathlib import Path
import pandas as pd
import re


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

EPOCH_DIR = ROOT / "data_epoched"
DATA_DIR = ROOT / "data_trimmed"

FS = 125.0

TASKS = [
    "1_BACK",
    "2_BACK",
    "3_BACK",
    "OPENMATB",
]

RESTS = ["REST_1", "REST_2", "REST_3"]

PERFORMANCE = [
    "Correct",
    "Wrong_Pressed",
    "Wrong_Not_Pressed",
]


# ============================================================
# FUNCIONES
# ============================================================

def marker_time(markers, name):

    rows = markers[
        markers["marker"].astype(str).str.strip() == name
    ]

    if rows.empty:
        return None

    return float(
        pd.to_numeric(
            rows["timestamp"],
            errors="coerce"
        ).dropna().iloc[0]
    )


def get_task_intervals(markers, eeg_start):

    intervals = []

    # --------------------------------------------------------
    # CALIBRACIÓN
    # --------------------------------------------------------

    cal_start = marker_time(
        markers,
        "eeg_calibration_started"
    )

    if cal_start is not None:

        t0 = cal_start - eeg_start

        # 0-60 s: ojos abiertos
        intervals.append(
            ("eeg_calibration_EO", t0, t0 + 60)
        )

        # 60-120 s: ojos cerrados
        intervals.append(
            ("eeg_calibration_EC", t0 + 60, t0 + 120)
        )

    # --------------------------------------------------------
    # TAREAS Y DESCANSOS
    # Puede haber VARIOS bloques de la misma tarea
    # --------------------------------------------------------

    for task in TASKS + RESTS:

        start_name = f"{task}_START"
        end_name = f"{task}_END"

        rows = markers[
            markers["marker"].isin([
                start_name,
                end_name
            ])
        ].copy()

        rows["timestamp"] = pd.to_numeric(
            rows["timestamp"],
            errors="coerce"
        )

        rows = (
            rows
            .dropna(subset=["timestamp"])
            .sort_values("timestamp")
        )

        current_start = None

        for _, row in rows.iterrows():

            marker = row["marker"]
            t = float(row["timestamp"]) - eeg_start

            if marker == start_name:

                current_start = t

            elif (
                marker == end_name
                and current_start is not None
                and t > current_start
            ):

                intervals.append(
                    (
                        task,
                        current_start,
                        t
                    )
                )

                current_start = None

    return intervals


def get_performance_events(markers, eeg_start):

    events = []

    # Cada tarea lleva su propia numeración
    counters = {
        task: 0
        for task in TASKS
    }

    intervals = get_task_intervals(
        markers,
        eeg_start
    )

    perf_rows = markers[
        markers["marker"].isin(PERFORMANCE)
    ].copy()

    perf_rows["timestamp"] = pd.to_numeric(
        perf_rows["timestamp"],
        errors="coerce"
    )

    perf_rows = (
        perf_rows
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
    )

    for _, row in perf_rows.iterrows():

        event_sec = (
            float(row["timestamp"])
            - eeg_start
        )

        # Averiguar dentro de qué tarea ocurrió
        condition = None

        for cond, start, end in intervals:

            if cond in TASKS and start <= event_sec <= end:
                condition = cond
                break

        # Si no ocurrió dentro de una tarea, ignoramos
        if condition is None:
            continue

        counters[condition] += 1

        events.append({
            "condition": condition,
            "performance": row["marker"],
            "event_id": counters[condition],
            "event_sec": event_sec,
        })

    return events


# ============================================================
# EJECUCIÓN
# ============================================================

def recording_blocks(marker_path):
    markers = pd.read_csv(marker_path)
    markers["timestamp"] = pd.to_numeric(markers.timestamp, errors="raise")
    markers["marker"] = markers.marker.astype(str).str.strip()
    markers = markers.sort_values("timestamp", kind="stable")
    eeg_start = float(pd.read_csv(marker_path.parent / "obci_eeg1_data.csv",
                                 usecols=["timestamp"], nrows=1).timestamp.iloc[0])
    intervals = sorted((c, a, b) for c, a, b in get_task_intervals(markers, eeg_start)
                       if c in ["1_BACK", "2_BACK", "3_BACK"])
    intervals.sort(key=lambda x: x[1])
    bedford = markers[markers.marker.str.fullmatch(r"[123]_BEDFORD_\d+")]
    starts = markers[markers.marker.isin([f"{n}_BACK_START" for n in [1, 2, 3]] + ["OPENMATB_START"])]
    used, rows = set(), []
    for block_id, (condition, start, end) in enumerate(intervals, 1):
        after = starts.loc[starts.timestamp >= eeg_start + end, "timestamp"]
        next_start = after.min() if len(after) else float("inf")
        # La valoracion se emite despues del END, a menudo durante REST.
        ratings = bedford[(bedford.timestamp >= eeg_start + end)
                          & (bedford.timestamp < next_start)]
        if len(ratings) > 1:
            raise ValueError(f"Varias valoraciones para un bloque: {marker_path}, {block_id}")
        score, marker, rating_sec = pd.NA, "", float("nan")
        if len(ratings):
            idx = ratings.index[0]
            marker = ratings.iloc[0].marker
            match = re.fullmatch(r"([123])_BEDFORD_(\d+)", marker)
            if int(match[1]) != int(condition[0]) or idx in used:
                raise ValueError(f"BEDFORD no coincide con el bloque: {marker_path}, {marker}")
            used.add(idx)
            value = int(match[2])
            # Conservar el marker original, pero no convertir 0 en carga baja.
            score = value if 1 <= value <= 10 else pd.NA
            rating_sec = float(ratings.iloc[0].timestamp) - eeg_start
        rows.append(dict(session=marker_path.parents[1].name,
                         participant=marker_path.parent.name,
                         bedford_block_id=block_id, condition=condition,
                         start_sec=start, end_sec=end, bedford_score=score,
                         bedford_valid=pd.notna(score), bedford_marker=marker,
                         bedford_sec=rating_sec))
    if len(used) != len(bedford):
        raise ValueError(f"Hay markers BEDFORD sin bloque: {marker_path}")
    return rows


def link_epochs(info, blocks):
    ids = pd.Series(pd.NA, index=info.index, dtype="Int64")
    for block in blocks.itertuples(index=False):
        mask = (info.condition.eq(block.condition)
                & info.start_sec.ge(block.start_sec)
                & info.end_sec.le(block.end_sec))
        if ids[mask].notna().any():
            raise ValueError("Un epoch pertenece a varios bloques BEDFORD")
        ids.loc[mask] = block.bedford_block_id
    # Incluir tambien rechazados; keep sigue indicando cuales estan en el NPZ.
    info = info.copy()
    info["bedford_block_id"] = ids
    return info



def main():

    rows = []
    for path in sorted(DATA_DIR.glob("session_*/participant_*/psychopymarkers_data.csv")):
        rows.extend(recording_blocks(path))
    blocks = pd.DataFrame(rows)
    if blocks.empty:
        raise ValueError("No se encontraron bloques N-back en data_trimmed")
    blocks["bedford_score"] = pd.array(blocks.bedford_score, dtype="Int64")
    if blocks.duplicated(["session", "participant", "bedford_block_id"]).any():
        raise ValueError("Hay identificadores de bloque BEDFORD duplicados")
    updates = []

    n_files = 0

    for info_path in EPOCH_DIR.glob(
        "*/session_*/participant_*/epoch_info.csv"
    ):

        session = info_path.parents[1].name
        participant = info_path.parent.name

        raw_dir = (
            DATA_DIR
            / session
            / participant
        )

        eeg = pd.read_csv(
            raw_dir / "obci_eeg1_data.csv"
        )

        markers = pd.read_csv(
            raw_dir / "psychopymarkers_data.csv"
        )

        epochs = pd.read_csv(info_path)

        # ----------------------------------------------------
        # Inicio real del EEG trimmed
        # ----------------------------------------------------

        eeg_start = float(
            pd.to_numeric(
                eeg["timestamp"],
                errors="coerce"
            ).dropna().iloc[0]
        )

        # ----------------------------------------------------
        # Tiempo de los epochs
        # ----------------------------------------------------

        if "start_sec" not in epochs.columns:

            if "start_sample" in epochs.columns:

                epochs["start_sec"] = (
                    epochs["start_sample"] / FS
                )

                epochs["end_sec"] = (
                    epochs["end_sample"] / FS
                )

            else:
                # Epochs 4 s, paso 2 s
                epochs["start_sec"] = (
                    epochs["epoch_index"] * 2.0
                )

                epochs["end_sec"] = (
                    epochs["start_sec"] + 4.0
                )

        # ----------------------------------------------------
        # CONDICIÓN
        # ----------------------------------------------------

        intervals = get_task_intervals(
            markers,
            eeg_start
        )

        def get_condition(row):

            matches = []

            for cond, start, end in intervals:

                # Epoch entero dentro del bloque
                if (
                    row["start_sec"] >= start
                    and row["end_sec"] <= end
                ):
                    matches.append(cond)

            if len(matches) == 1:
                return matches[0]

            if len(matches) > 1:
                return "+".join(matches)

            return ""

        epochs["condition"] = epochs.apply(
            get_condition,
            axis=1
        )

        # ----------------------------------------------------
        # PERFORMANCE
        # ----------------------------------------------------

        events = get_performance_events(
            markers,
            eeg_start
        )

        epochs["performance"] = ""
        epochs["event_id"] = ""
        epochs["event_sec"] = ""

        for event in events:

            mask = (
                (epochs["start_sec"] <= event["event_sec"])
                & (epochs["end_sec"] > event["event_sec"])
                & (epochs["condition"] == event["condition"])
            )

            # Puede aparecer en dos epochs por el solapamiento.
            # Conserva el MISMO event_id porque es el mismo evento.
            for idx in epochs.index[mask]:

                if epochs.at[idx, "performance"] == "":

                    epochs.at[
                        idx, "performance"
                    ] = event["performance"]

                    epochs.at[
                        idx, "event_id"
                    ] = str(event["event_id"])

                    epochs.at[
                        idx, "event_sec"
                    ] = f"{event['event_sec']:.3f}"

                else:
                    # Por si excepcionalmente caen dos eventos
                    # dentro del mismo epoch.
                    epochs.at[
                        idx, "performance"
                    ] += ";" + event["performance"]

                    epochs.at[
                        idx, "event_id"
                    ] += ";" + str(event["event_id"])

                    epochs.at[
                        idx, "event_sec"
                    ] += ";" + f"{event['event_sec']:.3f}"

        # ----------------------------------------------------
        # GUARDAR
        # ----------------------------------------------------

        output = (
            info_path.parent
            / "epoch_info_labeled.csv"
        )

        sub = blocks[(blocks.session == session)
                     & (blocks.participant == participant)]
        if sub.empty:
            raise ValueError(f"No hay bloques para {info_path}")
        epochs = link_epochs(epochs, sub)
        updates.append((output, epochs))

        n_files += 1

    # Validar todos los registros antes de escribir las salidas.
    EPOCH_DIR.mkdir(parents=True, exist_ok=True)
    blocks.to_csv(EPOCH_DIR / "bedford_blocks.csv", index=False)
    for output, epochs in updates:
        epochs.to_csv(output, index=False)
    print(
        f"\nETIQUETADO TERMINADO: "
        f"{n_files} archivos procesados"
    )

    print(f"BEDFORD: {len(blocks)} bloques, {blocks.bedford_valid.sum()} puntuaciones validas.")


if __name__ == "__main__":
    main()
