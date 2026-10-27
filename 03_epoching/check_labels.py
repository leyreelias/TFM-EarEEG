from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

for participant in ["participant_01", "participant_02"]:

    f = (
        ROOT / "data_epoched" / "all_mean"
        / "session_01" / participant
        / "epoch_info_labeled.csv"
    )

    df = pd.read_csv(f)

    print(f"\n=== {participant} ===")

    print("\nCondiciones:")
    print(df["condition"].value_counts(dropna=False))

    print("\nPerformance:")
    print(df["performance"].value_counts(dropna=False))

    print("\nPrimeros eventos:")
    print(
        df.loc[
            df["performance"].notna(),
            [
                "start_sec", "end_sec",
                "condition", "performance",
                "event_id", "event_sec"
            ]
        ].head(20).to_string(index=False)
    )

    print("\nCalibración alrededor del cambio EO/EC:")
    print(
        df.loc[
            (df["start_sec"] >= 54)
            & (df["start_sec"] <= 66),
            ["start_sec", "end_sec", "condition"]
        ].to_string(index=False)
    )

markers = pd.read_csv(
    ROOT / "data_trimmed" / "session_01"
    / "participant_02" / "psychopymarkers_data.csv"
)

names = [
    "OPENMATB_START",
    "OPENMATB_END",
    "PHASE_OPENMATB_START",
    "PHASE_OPENMATB_END",
    "PHASE_NO_OPENMATB_START",
    "PHASE_NO_OPENMATB_END",
]

x = markers[markers["marker"].isin(names)][
    ["timestamp", "marker"]
].copy()

x["relative_sec"] = (
    x["timestamp"] - markers["timestamp"].min()
)

print(x.to_string(index=False))