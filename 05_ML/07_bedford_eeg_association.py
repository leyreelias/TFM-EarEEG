"""Asociacion descriptiva EEG-BEDFORD, sin N-back como predictor.

Construye tablas de bloques para las cinco rereferencias desde features existentes.
Una fila = un bloque. Resume canales por su media, como 06_bedford.py.
Calcula Spearman por grabacion (session + participant), sin tratar los
bloques repetidos como observaciones independientes para un p-valor global.
No asume que los identificadores de carpeta equivalgan a personas distintas.
No demuestra causalidad ni elimina la posible influencia del nivel N-back.

Ejemplo de comparacion BEDFORD-EEG: Kim y Ji (2013),
https://doi.org/10.5143/JESK.2013.32.2.167 (alli emplean Pearson).
Aqui se usa Spearman para describir la asociacion por rangos de BEDFORD.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data_epoched/bedford_blocks.csv"
OUTPUT_ROOT = ROOT / "results/bedford_eeg_association"
FEATURES = ["log_theta", "log_alpha", "theta_alpha_ratio"]
MIN_EPOCHS = 10
MIN_BLOCKS = 5
KEY = ["session", "participant", "bedford_block_id"]


def build_table(path):
    blocks = pd.read_csv(INPUT)
    method = path.stem.replace("features_", "")
    epoch_key = ["session", "participant", "epoch"]
    feats = pd.read_csv(path, usecols=lambda c: c in epoch_key or
                        any(c.endswith("__" + f) for f in FEATURES))
    ids = []
    for meta in sorted((ROOT / "data_epoched" / method).glob("session_*/participant_*/epoch_info_labeled.csv")):
        info = pd.read_csv(meta, usecols=["epoch", "keep", "bedford_block_id"])
        info = info.loc[info.keep.eq(True), ["epoch", "bedford_block_id"]].copy()
        info["session"] = meta.parents[1].name
        info["participant"] = meta.parent.name
        ids.append(info)
    linked = feats.merge(pd.concat(ids), on=epoch_key, how="left",
                         validate="one_to_one", indicator=True)
    if not linked["_merge"].eq("both").all():
        raise ValueError(f"Features sin metadata: {method}")
    cols = [c for c in feats if "__" in c]
    grouped = linked.dropna(subset=["bedford_block_id"]).groupby(KEY)
    agg = grouped[cols].median()
    agg["n_epochs"] = grouped.size()
    return blocks.merge(agg.reset_index(), on=KEY, how="left", validate="one_to_one")


def correlations_and_summary(data):
    rows = []
    for (session, participant), group in data.groupby(KEY[:2]):
        for feature in FEATURES:
            valid = group[["bedford_score", feature]].dropna()
            status = "ok"
            rho = np.nan
            if len(valid) < MIN_BLOCKS:
                status = "pocos_bloques"
            elif valid.bedford_score.nunique() < 2 or valid[feature].nunique() < 2:
                status = "sin_variacion"
            else:
                rho = float(spearmanr(valid.bedford_score, valid[feature]).statistic)
            rows.append(dict(session=session, participant=participant, feature=feature,
                             n_blocks=len(valid), rho=rho, status=status))
    correlations = pd.DataFrame(rows)
    summary = correlations.groupby(["participant", "feature"]).agg(
        n_recordings=("rho", "count"), median_rho=("rho", "median"),
        min_rho=("rho", "min"), max_rho=("rho", "max"),
        n_positive=("rho", lambda x: int((x > 0).sum())),
        n_negative=("rho", lambda x: int((x < 0).sum())),
    )
    return correlations, summary


def analyze_method(path):
    method = path.stem.replace("features_", "")
    OUTPUT = OUTPUT_ROOT / method
    table = build_table(path)
    if table.duplicated(KEY).any():
        raise ValueError("La tabla contiene bloques duplicados")
    usable = table.bedford_score.between(1, 10) & table.n_epochs.ge(MIN_EPOCHS)
    data = table.loc[usable, KEY + ["bedford_score", "n_epochs"]].copy()
    if data.empty:
        raise ValueError("No hay bloques con EEG suficiente y BEDFORD valido")
    for feature in FEATURES:
        cols = [c for c in table if c.endswith("__" + feature)]
        if not cols:
            raise ValueError(f"Faltan columnas EEG: {feature}")
        values = table.loc[usable, cols].replace([np.inf, -np.inf], np.nan)
        data[feature] = values.mean(axis=1)
        data[feature + "_n_channels"] = values.notna().sum(axis=1)

    correlations, summary = correlations_and_summary(data)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT / "blocks_eeg_bedford.csv", index=False)
    correlations.to_csv(OUTPUT / "correlations_per_recording.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv")

    parts = sorted(data.participant.unique())
    fig, axes = plt.subplots(len(parts), len(FEATURES), squeeze=False,
                             figsize=(13, 3.7 * len(parts)))
    rng = np.random.default_rng(42)
    for i, participant in enumerate(parts):
        sub = data[data.participant.eq(participant)]
        for j, feature in enumerate(FEATURES):
            # Solo para visualizar: poner cada grabacion en su propia escala.
            # Las correlaciones anteriores se calculan con los valores originales.
            grouped = sub.groupby("session")[feature]
            y = (sub[feature] - grouped.transform("mean")) / grouped.transform("std").replace(0, np.nan)
            ax = axes[i, j]
            ax.scatter(sub.bedford_score + rng.uniform(-0.08, 0.08, len(sub)), y,
                       s=20, alpha=0.55)
            ax.axhline(0, color="grey", linewidth=0.7)
            ax.set(title=f"{participant}: {feature}", xlabel="BEDFORD",
                   ylabel="EEG estandarizado por grabacion", xticks=range(1, 11))
    fig.tight_layout()
    fig.savefig(OUTPUT / "eeg_vs_bedford.png", dpi=150)
    plt.close(fig)
    print(f"Bloques analizados: {len(data)} / {len(table)}; metodo: {method}.", flush=True)
    print(summary.round(3).to_string())
    print("rho positivo: EEG aumenta con BEDFORD; negativo: disminuye.")
    print("Resultados descriptivos por grabacion; sin p-valores globales ni clasificador.")
    print(f"Guardado en: {OUTPUT}")
    return data


def main():
    tables = {p.stem.replace("features_", ""): analyze_method(p)
              for p in sorted((ROOT / "data_features").glob("features_*.csv"))}
    # Comparacion adicional sobre los mismos bloques para todos los metodos.
    common = set.intersection(*(set(d.dropna(subset=FEATURES).set_index(KEY).index)
                                for d in tables.values()))
    rows = []
    for method, data in tables.items():
        data = data[data.set_index(KEY).index.isin(common)]
        corr, summary = correlations_and_summary(data)
        corr.to_csv(OUTPUT_ROOT / method / "correlations_common_blocks.csv", index=False)
        rows.append(summary.reset_index().assign(method=method, n_common_blocks=len(data)))
    comparison = pd.concat(rows, ignore_index=True)
    comparison.to_csv(OUTPUT_ROOT / "comparison_common_blocks.csv", index=False)
    print(f"\nComparacion sobre {len(common)} bloques comunes; mediana rho:")
    print(comparison.pivot(index=["participant", "feature"], columns="method", values="median_rho").round(3))


if __name__ == "__main__":
    main()
