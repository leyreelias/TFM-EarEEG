"""Diagnostico descriptivo de markers originales; no modifica datos EEG.

La fraccion de Correct se refiere a eventos registrados, no necesariamente
a todos los ensayos de la tarea. No se calculan RT ni pruebas inferenciales.
"""
from pathlib import Path
import importlib.util
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/behavior_diagnostic"
PERF = ["Correct", "Wrong_Pressed", "Wrong_Not_Pressed"]
TESTS = {"participant_01": "Solo N-back", "participant_02": "N-back + OpenMATB"}
NBACK = ["1_BACK", "2_BACK", "3_BACK"]
COLORS = ["#2166ac", "#c76020"]


def plot_performance(rec):
    """Una session es una persona; participant identifica la prueba realizada.

    Comparar exclusivamente niveles N-back equivalentes, sin mezclar los
    eventos de OPENMATB ni ponderar personas por su numero de eventos.
    """
    data = rec[rec.condition.isin(NBACK) & rec.participant.isin(TESTS)].copy()
    if data.duplicated(["session", "participant", "condition"]).any():
        raise ValueError("Registros de performance duplicados")
    sessions = sorted(data.session.unique(), key=lambda s: (int(s.split("_")[-1]), s))
    rows = []
    for condition in NBACK:
        subset = data[data.condition.eq(condition)]
        wide = subset.pivot(index="session", columns="participant",
                            values="fraction_correct_recorded").reindex(
                                index=sessions, columns=list(TESTS))
        for session, values in wide.iterrows():
            a, b = values["participant_01"], values["participant_02"]
            paired = pd.notna(a) and pd.notna(b)
            row = dict(session=session, condition=condition,
                       fraction_correct_nback=a, fraction_correct_nback_openmatb=b,
                       paired=paired,
                       difference_openmatb_minus_nback=b-a if paired else np.nan)
            for test in TESTS:
                original = subset[subset.session.eq(session) & subset.participant.eq(test)]
                row[f"n_events_{test}"] = int(original.n_recorded_events.iloc[0]) if len(original) else 0
            rows.append(row)
    pairs = pd.DataFrame(rows)
    pairs.to_csv(OUT / "performance_comparison_per_person.csv", index=False)
    matched = pairs[pairs.paired]
    summary = matched.groupby("condition").agg(
        n_people=("session", "nunique"),
        mean_nback=("fraction_correct_nback", "mean"),
        mean_nback_openmatb=("fraction_correct_nback_openmatb", "mean"),
        mean_difference=("difference_openmatb_minus_nback", "mean"),
        median_difference=("difference_openmatb_minus_nback", "median"),
        n_openmatb_higher=("difference_openmatb_minus_nback", lambda x: int((x > 0).sum())),
        n_nback_higher=("difference_openmatb_minus_nback", lambda x: int((x < 0).sum())),
        n_equal=("difference_openmatb_minus_nback", lambda x: int(np.isclose(x, 0, atol=1e-12, rtol=0).sum())),
    )
    summary.to_csv(OUT / "performance_comparison_summary.csv")

    # Dos barras por persona, manteniendo separados los tres niveles N-back.
    fig, axes = plt.subplots(3, 1, figsize=(max(11, len(sessions) * 0.7), 10), sharex=True)
    positions = np.arange(len(sessions))
    for ax, condition in zip(axes, NBACK):
        panel = pairs[pairs.condition.eq(condition)].set_index("session").reindex(sessions)
        for offset, field, label, color in zip(
                [-0.19, 0.19], ["fraction_correct_nback", "fraction_correct_nback_openmatb"],
                TESTS.values(), COLORS):
            ax.bar(positions + offset, panel[field], width=0.36, label=label, color=color)
        ax.set(title=condition.replace("_", "-"), ylim=(0, 1.08), ylabel="Eventos correctos (%)")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=2, frameon=False)
    axes[-1].set_xticks(positions, [s.replace("session_", "S") for s in sessions], rotation=45, ha="right")
    axes[-1].set_xlabel("Persona (identificada por su session)")
    fig.suptitle("Performance por persona y prueba", fontsize=16)
    fig.text(0.5, 0.015, "Correct / (Correct + Wrong_Pressed + Wrong_Not_Pressed). "
             "Solo eventos N-back registrados; una barra ausente indica falta de datos.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.045, 1, 0.925])
    fig.savefig(OUT / "performance_per_person.png", dpi=180)
    plt.close(fig)

    # Cada linea une las dos pruebas de la misma persona; resumen sin ponderar
    # por eventos, ya que las pruebas pueden tener distinta cantidad de bloques.
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    for ax, condition in zip(axes, NBACK):
        panel = matched[matched.condition.eq(condition)]
        for row in panel.itertuples(index=False):
            ax.plot([0, 1], [row.fraction_correct_nback, row.fraction_correct_nback_openmatb],
                    color="#888888", alpha=0.45, linewidth=1, marker="o", markersize=3)
        if len(panel):
            means = panel[["fraction_correct_nback", "fraction_correct_nback_openmatb"]].mean()
            ax.plot([0, 1], means.to_numpy(), color="black", linewidth=2.5,
                    marker="D", markersize=7, label="Media entre personas")
            delta = panel.difference_openmatb_minus_nback.mean() * 100
            ax.text(0.5, 0.04, f"Cambio medio: {delta:+.1f} puntos porcentuales",
                    transform=ax.transAxes, ha="center", fontsize=9)
        ax.set(title=f"{condition.replace('_', '-')} · {len(panel)} personas emparejadas",
               xticks=[0, 1], xticklabels=["Solo\nN-back", "N-back +\nOpenMATB"],
               xlim=(-0.25, 1.25), ylim=(0, 1.08))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Eventos correctos (%)")
    if len(matched):
        handles, labels = next(ax.get_legend_handles_labels() for ax in axes if ax.get_legend_handles_labels()[0])
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.06), frameon=False)
    fig.suptitle("Comparación emparejada de las dos pruebas", fontsize=15)
    fig.text(0.5, 0.015, "Cada línea gris representa una persona. Resumen descriptivo de eventos registrados; sin pruebas de significación.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.13, 1, 0.93])
    fig.savefig(OUT / "performance_paired_tests.png", dpi=180)
    plt.close(fig)
    print("\nComparacion entre pruebas (cada session = una persona):")
    print(summary.round(3).to_string())
    print(f"Graficos y tablas de performance: {OUT}")


def main():
    spec = importlib.util.spec_from_file_location("existing_labels", ROOT / "03_epoching/02_ label_epoch.py")
    labels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(labels)
    rows, unmatched = [], []
    for path in sorted((ROOT / "data").glob("session_*/participant_*/psychopymarkers_data.csv")):
        session, person = path.parents[1].name, path.parent.name
        markers = pd.read_csv(path)
        markers["timestamp"] = pd.to_numeric(markers.timestamp, errors="raise")
        # Reutilizar exactamente el detector START/END ya usado en etiquetado.
        intervals = [(c, a, b) for c, a, b in labels.get_task_intervals(markers, 0)
                     if c in ["1_BACK", "2_BACK", "3_BACK", "OPENMATB"]]
        perf = markers[markers.marker.isin(PERF)]
        assigned = set()
        for block, (cond, start, end) in enumerate(intervals, 1):
            events = perf[(perf.timestamp >= start) & (perf.timestamp <= end)]
            if assigned & set(events.index):
                raise ValueError(f"Intervalos solapados: {path}")
            assigned.update(events.index)
            counts = events.marker.value_counts()
            rows.append(dict(session=session, participant=person, condition=cond,
                             block=block, duration_sec=end-start,
                             **{name: int(counts.get(name, 0)) for name in PERF}))
        unmatched.append(dict(session=session, participant=person,
                              n_performance_markers=len(perf),
                              n_unassigned=len(set(perf.index) - assigned)))
    OUT.mkdir(parents=True, exist_ok=True)
    blocks = pd.DataFrame(rows)
    blocks.to_csv(OUT / "counts_per_block.csv", index=False)
    pd.DataFrame(unmatched).to_csv(OUT / "marker_assignment_audit.csv", index=False)
    rec = blocks.groupby(["session", "participant", "condition"])[PERF + ["duration_sec"]].sum().reset_index()
    rec["n_recorded_events"] = rec[PERF].sum(axis=1)
    rec["fraction_correct_recorded"] = rec.Correct / rec.n_recorded_events.replace(0, float("nan"))
    rec.to_csv(OUT / "counts_per_recording.csv", index=False)
    plot_performance(rec)
    pair = rec[rec.condition.isin(["1_BACK", "3_BACK"])].pivot(
        index=["session", "participant"], columns="condition", values="fraction_correct_recorded").dropna()
    pair["difference_3_minus_1"] = pair["3_BACK"] - pair["1_BACK"]
    pair.to_csv(OUT / "paired_1back_3back.csv")
    print("Fraccion Correct entre los eventos registrados (NO accuracy por ensayo):")
    print(pair.groupby("participant")[["1_BACK", "3_BACK", "difference_3_minus_1"]].mean().round(3))
    print("Grabaciones con menor fraccion en 3-back:", int((pair.difference_3_minus_1 < 0).sum()), "/", len(pair))
    print("Markers fuera de bloques:", sum(r["n_unassigned"] for r in unmatched))
    # Caso observado en el diagnostico PSD: auditar antes y despues de keep.
    audit = []
    for method_dir in sorted((ROOT / "data_epoched").iterdir()):
        path = method_dir / "session_015/participant_02/epoch_info_labeled.csv"
        if not path.exists():
            continue
        info = pd.read_csv(path)
        for cond in ["1_BACK", "3_BACK"]:
            sub = info[info.condition == cond]
            audit.append(dict(method=method_dir.name, condition=cond, n_candidates=len(sub),
                              n_kept=int(sub.keep.eq(True).sum()),
                              mean_nan_percent=sub.nan_percent.mean()))
    audit = pd.DataFrame(audit)
    audit.to_csv(OUT / "session_015_participant_02_epoch_audit.csv", index=False)
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
