"""Compara calibracion EO/EC con las features existentes de las 5 rereferencias.

EO = primeros 60 s; EC = siguientes 60 s, segun las etiquetas existentes.
Una observacion emparejada es una grabacion, nunca un epoch independiente.
Mediana temporal por canal/condicion y media entre canales validos en ambas.
No normaliza respecto a EO, no imputa ni interpola, no entrena clasificadores.
Analisis descriptivo: los IDs no se interpretan automaticamente como personas.
Como EO precede siempre a EC, condicion y orden temporal no se separan.

Motivacion: aumento de alpha con ojos cerrados observado en cEEGrid:
Debener et al. (2015), https://doi.org/10.1038/srep16743
Se conserva nuestra banda alpha 8-13 Hz; no se cambia la extraccion.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data_features"
OUTPUT = ROOT / "results/calibration_eo_ec"
EO, EC = "eeg_calibration_EO", "eeg_calibration_EC"
MIN_VALID_EPOCHS = 10  # Por canal y condicion; no basta el total del bloque.
KEY = ["session", "participant"]


def analyze_method(path):
    method = path.stem.replace("features_", "")
    df = pd.read_csv(path, low_memory=False)
    df = df[df.condition.isin([EO, EC])].copy()
    if df.duplicated(KEY + ["epoch"]).any():
        raise ValueError(f"Epochs duplicados en {method}")
    cols = [c for c in df if "__" in c]
    rows = []
    for (session, participant), group in df.groupby(KEY):
        a = group.loc[group.condition.eq(EO), cols].replace([np.inf, -np.inf], np.nan)
        b = group.loc[group.condition.eq(EC), cols].replace([np.inf, -np.inf], np.nan)
        na, nb = a.count(), b.count()
        ma, mb = a.median(), b.median()
        for col in cols:
            channel, feature = col.split("__", 1)
            valid = na[col] >= MIN_VALID_EPOCHS and nb[col] >= MIN_VALID_EPOCHS
            rows.append(dict(method=method, session=session, participant=participant,
                             channel=channel, feature=feature,
                             n_epochs_eo=len(a), n_epochs_ec=len(b),
                             n_valid_eo=int(na[col]), n_valid_ec=int(nb[col]),
                             usable=valid, eo=ma[col], ec=mb[col],
                             difference_ec_minus_eo=mb[col]-ma[col] if valid else np.nan))
    channels = pd.DataFrame(rows)
    if channels.empty:
        raise ValueError(f"Sin calibracion en {method}")
    # Mismos canales en EO y EC: evita diferencias debidas a cobertura distinta.
    paired = channels[channels.usable].groupby(["method"] + KEY + ["feature"]).agg(
        eo=("eo", "mean"), ec=("ec", "mean"),
        difference_ec_minus_eo=("difference_ec_minus_eo", "mean"),
        n_channels=("channel", "nunique"),
    ).reset_index()
    folder = OUTPUT / method
    folder.mkdir(parents=True, exist_ok=True)
    channels.to_csv(folder / "channel_pairs.csv", index=False)
    paired.to_csv(folder / "recording_pairs.csv", index=False)

    # Localizacion descriptiva del efecto alpha por canal/derivacion.
    alpha = channels[channels.feature.eq("log_alpha") & channels.usable]
    if not alpha.empty:
        delta = alpha.groupby("channel").difference_ec_minus_eo.median().sort_values()
        fig, ax = plt.subplots(figsize=(8, max(4, 0.22 * len(delta))))
        ax.barh(delta.index, delta.values)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set(title=f"{method}: alpha por canal", xlabel="Mediana entre grabaciones de EC - EO (dB)")
        ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout()
        fig.savefig(folder / "alpha_channels.png", dpi=130)
        plt.close(fig)
    print(f"{method}: {len(df)} epochs de calibracion; "
          f"{len(paired[paired.feature.eq('log_alpha')])} pares alpha utilizables.", flush=True)
    return paired


def summarize(pairs):
    return pairs.groupby(["method", "participant", "feature"]).agg(
        n_recordings=("difference_ec_minus_eo", "count"),
        median_difference=("difference_ec_minus_eo", "median"),
        mean_difference=("difference_ec_minus_eo", "mean"),
        n_ec_higher=("difference_ec_minus_eo", lambda x: int((x > 0).sum())),
    ).reset_index()


def main():
    paths = sorted(INPUT.glob("features_*.csv"))
    if len(paths) != 5:
        raise ValueError("Se esperaban los cinco CSV de features")
    pairs = pd.concat([analyze_method(p) for p in paths], ignore_index=True)
    summarize(pairs).to_csv(OUTPUT / "summary.csv", index=False)
    # Comparacion entre metodos con las mismas grabaciones, por cada feature.
    common_rows = []
    for feature, sub in pairs.groupby("feature"):
        methods = [p.stem.replace("features_", "") for p in paths]
        common = set.intersection(*(set(sub[sub.method.eq(m)].set_index(KEY).index) for m in methods))
        common_rows.append(sub[sub.set_index(KEY).index.isin(common)])
    common = pd.concat(common_rows, ignore_index=True)
    common.to_csv(OUTPUT / "recording_pairs_common.csv", index=False)
    summary = summarize(common)
    summary.to_csv(OUTPUT / "summary_common.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), squeeze=False)
    for ax, (method, group) in zip(axes.flat, pairs[pairs.feature.eq("log_alpha")].groupby("method")):
        for row in group.itertuples(index=False):
            color = "tab:blue" if row.participant == "participant_01" else "tab:orange"
            ax.plot([0, 1], [row.eo, row.ec], color=color, alpha=0.45, marker=".")
        ax.set(title=method, xticks=[0, 1], xticklabels=["EO", "EC"],
               ylabel="Alpha (dB); media de medianas por canal")
    axes.flat[-1].axis("off")
    fig.suptitle("Calibracion: cada linea es una grabacion (azul: participant_01; naranja: participant_02)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT / "alpha_eo_ec.png", dpi=150)
    plt.close(fig)
    print("\nALPHA: diferencias EC - EO, sobre grabaciones comunes:")
    print(summary[summary.feature.eq("log_alpha")].round(3).to_string(index=False))
    print("Positivo: mayor alpha en ojos cerrados. Descriptivo; no son pruebas de significacion.")
    print(f"Resultados: {OUTPUT}")


if __name__ == "__main__":
    main()
