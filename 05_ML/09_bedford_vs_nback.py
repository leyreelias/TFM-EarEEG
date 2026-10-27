"""
¿Qué relaciona mejor el EEG: el nivel de BEDFORD (subjetivo) o el nivel de N-BACK (objetivo)?

Hace EXACTAMENTE los mismos cálculos dos veces, una con cada objetivo, sobre los mismos bloques:
  - Bedford: la puntuación 1-10 que dio la persona al terminar el bloque
  - N-back : el nivel de la tarea (1, 2 o 3)

Unidad de análisis: el BLOQUE (un N-back de 60 s). Cada SESIÓN es una PERSONA;
participant_01 = N-back solo, participant_02 = N-back + OpenMATB. La réplica es la persona.
Todo se hace DENTRO de cada persona y prueba y se resume entre personas.

Pasos:
  1) Tabla de bloques (epochs asignados por tiempo, mediana por bloque)
  2) Auditoría de Bedford y de cuánto se parecen Bedford y N-back
  3) Relación univariante: por persona, Spearman(feature, objetivo); resumen entre personas
  4) Relación multivariante: Ridge (regularización elegida validando por personas) entrena con otras personas y predice el objetivo de una
     persona NUEVA (leave-one-person-out); correlación predicho-real por persona,
     IC y test de permutaciones. Se hace para cada objetivo.
  5) COMPARACIÓN: ¿cuál se predice mejor? Diferencia emparejada por persona + tabla cruzada
     (modelo entrenado con X evaluado contra Y)

Se elimina la deriva temporal (efecto del orden del bloque) de las features antes de analizar.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = ROOT / "data_features"          # o ROOT / "data_features_psd" (PSD fino)
BEDFORD_CSV = ROOT / "data_epoched" / "bedford_blocks.csv"
OUT_ROOT = ROOT / "results" / "bedford_vs_nback"

# Métodos de rereferencia a analizar (uno o varios)
METHODS = ["all_mean"]
# METHODS = ["all_mean", "contralateral_mean", "ipsilateral_mean",
#            "contralateral_bipolar", "ipsilateral_bipolar"]

TASK_NAMES = {"participant_01": "N-back", "participant_02": "N-back+OpenMATB"}
POOL_TASKS = True            # añade el análisis conjunto de ambas pruebas

# Objetivos que se comparan: nombre -> columna de la tabla de bloques
TARGETS = {"Bedford": "bedford_score", "N-back": "N"}

AGG = "median"               # cómo resumir los epochs de un bloque
MIN_EPOCHS_BLOCK = 10        # bloque con menos epochs válidos -> excluido (completo ~28)
MIN_BLOCKS_PERSON = 6        # mínimo de bloques de una persona para calcular su correlación
MIN_SESSIONS = 5             # mínimo de personas por análisis

REMOVE_ORDER_DRIFT = True    # quita la tendencia lineal con el orden del bloque (por persona y prueba)
ML_FEATURES = "all"          # "all" = canal x feature | "global" = media entre canales
N_BOOT = 1000                # remuestreos de personas para el IC
N_PERM = 200                 # permutaciones para el p-valor del modelo (0 = omitir)
RIDGE_ALPHAS = np.logspace(0, 6, 13)   # regularización candidata (se elige dentro del entrenamiento)
RANDOM_STATE = 42


# ============================================================
# UTILIDADES
# ============================================================

def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    n = len(q)
    if n == 0:
        return out
    order = np.argsort(q)
    ranked = q[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty(n)
    res[order] = np.clip(ranked, 0, 1)
    out[ok] = res
    return out


def global_from(M):
    """Media entre canales de cada tipo de feature (p. ej. log_alpha global)."""
    types = sorted({c.split("__", 1)[1] for c in M.columns})
    return pd.DataFrame(
        {t: M[[c for c in M.columns if c.endswith("__" + t)]].mean(axis=1) for t in types},
        index=M.index,
    )


def remove_order_drift(M, d):
    """Resta a cada feature su tendencia lineal con el orden del bloque, dentro de cada
    persona y prueba. Solo usa el orden (conocido), nunca el Bedford."""
    out = M.copy()
    for _, idx in d.groupby("unit").groups.items():
        oc = d.loc[idx, "block_order"].to_numpy(float)
        oc = oc - oc.mean()
        sub = M.loc[idx]
        xc = (sub - sub.mean()).to_numpy()
        den = (sub.notna().to_numpy() * (oc[:, None] ** 2)).sum(axis=0)
        num = np.nansum(xc * oc[:, None], axis=0)
        slope = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        out.loc[idx] = sub - oc[:, None] * slope[None, :]
    return out


def zscore_by_unit(M, d):
    g = M.groupby(d["unit"])
    return (M - g.transform("mean")) / g.transform("std").replace(0, np.nan)


def task_groups(table):
    """{nombre: bloques usables} por prueba, más ambas pruebas juntas."""
    usable = table[table["usable"]]
    out = {TASK_NAMES.get(p, p): d for p, d in usable.groupby("participant")}
    if POOL_TASKS and len(out) > 1:
        out["ambas_tareas"] = usable
    return out


# ============================================================
# 1) TABLA DE BLOQUES
# ============================================================

def build_block_table(method, out_dir):
    blocks = pd.read_csv(BEDFORD_CSV)
    feats = pd.read_csv(FEATURE_DIR / f"features_{method}.csv", low_memory=False)
    feats = feats.copy()   # consolida el DataFrame (evita avisos de fragmentación)
    feature_cols = [c for c in feats.columns if "__" in c]

    # Un epoch pertenece a un bloque si: misma sesión y prueba, misma condición y el
    # centro del epoch cae dentro de [start_sec, end_sec] del bloque.
    feats["mid"] = (feats["start_sec"] + feats["end_sec"]) / 2
    feats["bedford_block_id"] = np.nan

    for blk in blocks.itertuples(index=False):
        m = (
            (feats["session"] == blk.session)
            & (feats["participant"] == blk.participant)
            & (feats["condition"] == blk.condition)
            & (feats["mid"] >= blk.start_sec)
            & (feats["mid"] <= blk.end_sec)
        )
        if feats.loc[m, "bedford_block_id"].notna().any():
            raise ValueError(f"Epochs asignados a dos bloques: {blk.session}/{blk.participant}")
        feats.loc[m, "bedford_block_id"] = blk.bedford_block_id

    assigned = feats.dropna(subset=["bedford_block_id"]).copy()
    assigned["bedford_block_id"] = assigned["bedford_block_id"].astype(int)

    keys = ["session", "participant", "bedford_block_id"]
    g = assigned.groupby(keys)
    agg = g[feature_cols].agg(AGG)
    agg["n_epochs"] = g.size()
    agg = agg.reset_index()

    table = blocks.merge(agg, on=keys, how="left")
    table["block_order"] = table["bedford_block_id"]
    table["N"] = table["condition"].str[0].astype(int)   # nivel de N-back (1, 2, 3)
    table["unit"] = table["session"] + "|" + table["participant"]   # persona x prueba
    table["usable_eeg"] = table["n_epochs"].fillna(0) >= MIN_EPOCHS_BLOCK
    table["usable"] = table["usable_eeg"] & table["bedford_valid"]

    table.to_csv(out_dir / "block_table.csv", index=False)
    print(f"Bloques: {len(table)} | EEG suficiente: {int(table['usable_eeg'].sum())} "
          f"| Bedford válido: {int(table['bedford_valid'].sum())} "
          f"| usables (ambos): {int(table['usable'].sum())}")
    return table, feature_cols


# ============================================================
# 2) AUDITORÍA DEL BEDFORD
# ============================================================

def audit(table, out_dir):
    print("\n" + "=" * 70 + "\n2) AUDITORÍA: variabilidad del Bedford\n" + "=" * 70)
    groups = {k: v for k, v in task_groups(table).items() if k != "ambas_tareas"}
    fig, axes = plt.subplots(1, len(groups), figsize=(4.5 * len(groups), 3.5), squeeze=False)

    for ax, (name, d) in zip(axes[0], groups.items()):
        sd = d.groupby("session")["bedford_score"].std()
        rng_ = d.groupby("session")["bedford_score"].agg(lambda s: s.max() - s.min())
        print(f"\n{name}: {len(d)} bloques, {d['session'].nunique()} personas")
        print(f"  Bedford: media {d['bedford_score'].mean():.2f}, sd {d['bedford_score'].std():.2f}, "
              f"rango {d['bedford_score'].min():.0f}-{d['bedford_score'].max():.0f}")
        print(f"  Variación DENTRO de persona: sd mediana {sd.median():.2f} "
              f"(mín {sd.min():.2f}); rango mediano {rng_.median():.0f} puntos; "
              f"personas con sd<1: {int((sd < 1).sum())}")
        r_units = [stats.spearmanr(g["N"], g["bedford_score"])[0]
                   for _, g in d.groupby("unit") if g["bedford_score"].nunique() > 1]
        print(f"  Parecido Bedford-N dentro de cada persona: rho medio = {np.nanmean(r_units):.2f} "
              "(si es alto, los dos objetivos casi dicen lo mismo)")
        ax.hist(d["bedford_score"], bins=np.arange(0.5, 11.5, 1), edgecolor="k")
        ax.set_title(name); ax.set_xlabel("Bedford"); ax.set_ylabel("nº bloques")
    fig.tight_layout()
    fig.savefig(out_dir / "bedford_distribucion.png", dpi=130)
    plt.close(fig)


# ============================================================
# 3) RELACIÓN UNIVARIANTE
# ============================================================

def person_spearman(M, d, col):
    """Spearman(feature, objetivo) dentro de cada persona. Devuelve personas x features.
    Si una persona tiene dos pruebas (grupo conjunto), se promedian sus dos correlaciones."""
    R = M.groupby(d["unit"]).rank()
    b = d[col].groupby(d["unit"]).rank()
    rows = {}
    for u, idx in d.groupby("unit").groups.items():
        if len(idx) >= MIN_BLOCKS_PERSON:
            rows[u] = R.loc[idx].corrwith(b.loc[idx])
    rho_units = pd.DataFrame(rows).T
    return rho_units.groupby(rho_units.index.str.split("|").str[0]).mean()


def summarize_rho(rho_persons):
    out = []
    for f in rho_persons.columns:
        x = rho_persons[f].dropna().to_numpy()
        if len(x) < MIN_SESSIONS:
            continue
        p = stats.wilcoxon(x).pvalue if np.any(x != 0) else 1.0
        out.append({"feature": f, "n_personas": len(x), "rho_media": x.mean(),
                    "rho_mediana": np.median(x), "pct_personas_positivas": (x > 0).mean() * 100,
                    "p_wilcoxon": p})
    res = pd.DataFrame(out)
    if not res.empty:
        res["p_fdr"] = bh_fdr(res["p_wilcoxon"])
    return res


def heatmap(summaries, out_dir, tag):
    names = list(summaries)
    sample = summaries[names[0]]
    sample = sample.assign(ch=sample.feature.str.split("__").str[0],
                           ft=sample.feature.str.split("__").str[1])
    n_ch, n_ft = sample.ch.nunique(), sample.ft.nunique()
    fig, axes = plt.subplots(1, len(names), figsize=(max(5, 0.55 * n_ft + 3) * len(names),
                                                     max(4, 0.25 * n_ch + 2)), squeeze=False)
    for ax, name in zip(axes[0], names):
        s = summaries[name].copy()
        s["ch"] = s.feature.str.split("__").str[0]
        s["ft"] = s.feature.str.split("__").str[1]
        rho = s.pivot(index="ch", columns="ft", values="rho_media").reindex(
            index=sample.ch.unique(), columns=sample.ft.unique())
        pf = s.pivot(index="ch", columns="ft", values="p_fdr").reindex_like(rho)
        pr = s.pivot(index="ch", columns="ft", values="p_wilcoxon").reindex_like(rho)
        vmax = max(0.2, np.nanmax(np.abs(rho.to_numpy())))
        im = ax.imshow(rho.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(rho.shape[1])); ax.set_xticklabels(rho.columns, rotation=75, fontsize=7, ha="right")
        ax.set_yticks(range(rho.shape[0])); ax.set_yticklabels(rho.index, fontsize=7)
        for i in range(rho.shape[0]):
            for j in range(rho.shape[1]):
                if pf.iat[i, j] < 0.05:
                    ax.text(j, i, "*", ha="center", va="center", fontsize=10)
                elif pr.iat[i, j] < 0.05:
                    ax.text(j, i, "·", ha="center", va="center", fontsize=10)
        ax.set_title(f"{name}\nrho medio con {tag} (rojo: sube con {tag})", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("* = FDR<0.05   · = p<0.05 sin corregir (Wilcoxon entre personas)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"heatmap_rho_{tag}.png", dpi=130)
    plt.close(fig)


# ============================================================
# 4) RELACIÓN MULTIVARIANTE (leave-one-person-out)
# ============================================================

def ridge_fit_predict(Xtr, ytr, gtr, Xte):
    """Ridge cuya regularización (alpha) se elige con validación POR PERSONAS dentro del
    entrenamiento. La persona de test nunca interviene. (El selector automático de sklearn
    validaba bloque a bloque, mezclando bloques de una misma persona, y elegía casi sin
    regularización cuando hay más features que bloques -> sobreajuste.)"""
    imp = SimpleImputer(strategy="median").fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    A, B = sc.transform(imp.transform(Xtr)), sc.transform(imp.transform(Xte))

    errs = np.zeros(len(RIDGE_ALPHAS))
    for tr_i, va_i in GroupKFold(n_splits=min(5, len(np.unique(gtr)))).split(A, ytr, gtr):
        mu_a, mu_y = A[tr_i].mean(0), ytr[tr_i].mean()
        U, sv, Vt = np.linalg.svd(A[tr_i] - mu_a, full_matrices=False)
        Uty = U.T @ (ytr[tr_i] - mu_y)
        for k, a in enumerate(RIDGE_ALPHAS):
            w = Vt.T @ (sv / (sv ** 2 + a) * Uty)
            errs[k] += np.sum((ytr[va_i] - ((A[va_i] - mu_a) @ w + mu_y)) ** 2)
    alpha = RIDGE_ALPHAS[int(np.argmin(errs))]

    mu_a, mu_y = A.mean(0), ytr.mean()
    U, sv, Vt = np.linalg.svd(A - mu_a, full_matrices=False)
    w = Vt.T @ (sv / (sv ** 2 + alpha) * (U.T @ (ytr - mu_y)))
    return (B - mu_a) @ w + mu_y


def loso_predict(X, y, groups):
    pred = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        pred[te] = ridge_fit_predict(X[tr], y[tr], groups[tr], X[te])
    return pred


def person_rhos(pred, y, groups):
    out = {}
    for s in np.unique(groups):
        m = groups == s
        if m.sum() >= MIN_BLOCKS_PERSON and np.nanstd(y[m]) > 0 and np.nanstd(pred[m]) > 0:
            out[s] = stats.spearmanr(pred[m], y[m])[0]
    return pd.Series(out, dtype=float)


def _one_perm(seed, X, y, groups, unit_idx):
    rng = np.random.default_rng(seed)
    yp = y.copy()
    for idx in unit_idx:                       # se baraja el Bedford DENTRO de cada persona y prueba
        yp[idx] = rng.permutation(y[idx])
    return person_rhos(loso_predict(X, yp, groups), yp, groups).mean()


def predict_target(name, tag, col, d, M_all, M_glob):
    M = (M_all if ML_FEATURES == "all" else M_glob).copy()
    M = zscore_by_unit(M, d)
    X = M.to_numpy(dtype=float)

    # Objetivo centrado en cada persona y prueba (quita el nivel base individual)
    y = (d[col] - d.groupby("unit")[col].transform("mean")).to_numpy(float)
    groups = d["session"].to_numpy()
    units = d["unit"].to_numpy()

    pred = loso_predict(X, y, groups)
    rhos = person_rhos(pred, y, groups)
    mean_rho = rhos.mean()
    pooled = stats.spearmanr(pred, y)[0]
    p_w = stats.wilcoxon(rhos).pvalue if len(rhos) >= MIN_SESSIONS else np.nan

    rng = np.random.default_rng(RANDOM_STATE)
    boots = [rng.choice(rhos.to_numpy(), len(rhos), replace=True).mean() for _ in range(N_BOOT)]
    ci = np.percentile(boots, [2.5, 97.5])

    p_perm = np.nan
    if N_PERM > 0:
        unit_idx = [np.where(units == u)[0] for u in np.unique(units)]
        seeds = np.random.default_rng(RANDOM_STATE + 1).integers(0, 2**31 - 1, N_PERM)
        null = np.array(Parallel(n_jobs=-1)(delayed(_one_perm)(s, X, y, groups, unit_idx) for s in seeds))
        p_perm = (1 + np.sum(null >= mean_rho)) / (1 + N_PERM)

    print(f"\n{name} -> objetivo {tag}: {len(d)} bloques, {len(rhos)} personas, {X.shape[1]} features")
    print(f"  Correlación media {tag} predicho vs real (por persona): rho = {mean_rho:.3f} "
          f"[IC95% {ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"  Personas con rho>0: {int((rhos > 0).sum())}/{len(rhos)} | Wilcoxon p = {p_w:.4f} "
          f"| permutaciones p = {p_perm:.4f} | rho global (bloques) = {pooled:.3f}")

    return pred, y, rhos, {
        "grupo": name, "objetivo": tag, "n_bloques": len(d), "n_personas": len(rhos), "n_features": X.shape[1],
        "rho_medio": mean_rho, "ic_low": ci[0], "ic_high": ci[1],
        "pct_personas_positivas": (rhos > 0).mean() * 100,
        "p_wilcoxon": p_w, "p_permutaciones": p_perm, "rho_global": pooled,
    }


def dose_response_plot(results, out_dir, tag):
    names = list(results)
    fig, axes = plt.subplots(2, len(names), figsize=(4.6 * len(names), 7.5), squeeze=False)
    rng = np.random.default_rng(RANDOM_STATE)
    for j, name in enumerate(names):
        pred, y, rhos = results[name]
        ax = axes[0, j]
        ax.scatter(y + rng.uniform(-0.15, 0.15, len(y)), pred, s=8, alpha=0.3, color="grey")
        bins = np.round(y)
        xs = np.unique(bins)
        mu = [np.nanmean(pred[bins == b]) for b in xs]
        se = [np.nanstd(pred[bins == b]) / np.sqrt((bins == b).sum()) for b in xs]
        ax.errorbar(xs, mu, yerr=se, color="tab:red", marker="o", lw=2, capsize=3)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(f"{tag} real (centrado en la persona)")
        ax.set_ylabel(f"{tag} predicho por EEG (persona no vista)")

        ax = axes[1, j]
        ax.scatter(rng.uniform(-0.12, 0.12, len(rhos)), rhos.to_numpy(), s=30, color="tab:blue")
        ax.axhline(0, color="k", lw=0.8)
        ax.boxplot(rhos.to_numpy(), positions=[0], widths=0.4, showfliers=False)
        ax.set_xticks([]); ax.set_ylabel("rho por persona")
        ax.set_title(f"media {rhos.mean():.2f} · {int((rhos > 0).sum())}/{len(rhos)} personas >0", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"predicho_vs_real_{tag}.png", dpi=130)
    plt.close(fig)


# ============================================================
# 5) COMPARACIÓN BEDFORD vs N-BACK
# ============================================================

def compare_targets(name, d, res):
    """res[tag] = (pred, y, rhos). Devuelve fila resumen y la imprime."""
    groups = d["session"].to_numpy()
    (pB, yB, rB), (pN, yN, rN) = res["Bedford"], res["N-back"]

    cross = {
        ("Bedford", "Bedford"): rB.mean(),
        ("Bedford", "N-back"): person_rhos(pB, yN, groups).mean(),
        ("N-back", "Bedford"): person_rhos(pN, yB, groups).mean(),
        ("N-back", "N-back"): rN.mean(),
    }

    idx = rB.index.intersection(rN.index)
    diff = (rN[idx] - rB[idx]).to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)
    boots = [rng.choice(diff, len(diff), replace=True).mean() for _ in range(N_BOOT)]
    ci = np.percentile(boots, [2.5, 97.5])
    p_w = stats.wilcoxon(diff).pvalue if len(diff) >= MIN_SESSIONS and np.any(diff != 0) else np.nan

    print(f"\n  COMPARACIÓN en {name}  (correlación media entre personas)")
    print(f"  {'':34s}{'con Bedford':>13s}{'con N-back':>13s}")
    print(f"  {'Modelo entrenado con Bedford':34s}{cross[('Bedford', 'Bedford')]:13.3f}{cross[('Bedford', 'N-back')]:13.3f}")
    print(f"  {'Modelo entrenado con N-back':34s}{cross[('N-back', 'Bedford')]:13.3f}{cross[('N-back', 'N-back')]:13.3f}")
    print(f"  Diferencia (N-back - Bedford), cada modelo contra su propio objetivo: "
          f"{diff.mean():+.3f} [IC95% {ci[0]:+.3f}, {ci[1]:+.3f}]")
    print(f"  N-back gana en {int((diff > 0).sum())} de {len(diff)} personas | Wilcoxon p = {p_w:.3f}")

    return {
        "grupo": name,
        "entrenado_Bedford_vs_Bedford": cross[("Bedford", "Bedford")],
        "entrenado_Bedford_vs_Nback": cross[("Bedford", "N-back")],
        "entrenado_Nback_vs_Bedford": cross[("N-back", "Bedford")],
        "entrenado_Nback_vs_Nback": cross[("N-back", "N-back")],
        "dif_Nback_menos_Bedford": diff.mean(), "dif_ic_low": ci[0], "dif_ic_high": ci[1],
        "personas_donde_gana_Nback": int((diff > 0).sum()), "n_personas": len(diff),
        "p_wilcoxon_dif": p_w,
    }


# ============================================================
# EJECUCIÓN
# ============================================================

def run_method(method):
    out_dir = OUT_ROOT / method
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 70 + f"\n# {method}\n" + "#" * 70)
    print("=" * 70 + "\n1) TABLA DE BLOQUES\n" + "=" * 70)
    table, feature_cols = build_block_table(method, out_dir)

    audit(table, out_dir)

    print("\n" + "=" * 70 + "\n3-5) EEG ~ Bedford y EEG ~ N-back (dentro de persona)\n" + "=" * 70)
    sum_ch = {t: {} for t in TARGETS}
    ml_rows, cmp_rows, rho_rows, all_global = [], [], [], []
    ml_plot = {t: {} for t in TARGETS}

    for name, d in task_groups(table).items():
        if len(d) < 30 or d["session"].nunique() < MIN_SESSIONS:
            print(f"{name}: pocos datos, se omite")
            continue

        M_all = table.loc[d.index, feature_cols].copy()
        if REMOVE_ORDER_DRIFT:
            M_all = remove_order_drift(M_all, d)
        M_glob = global_from(M_all)

        print(f"\n{'=' * 70}\n{name} ({d['session'].nunique()} personas, {len(d)} bloques)\n{'=' * 70}")

        # --- univariante: mismo cálculo con cada objetivo
        s_gl = {}
        for t, col in TARGETS.items():
            s_ch = summarize_rho(person_spearman(M_all, d, col))
            sum_ch[t][name] = s_ch
            s_ch.assign(grupo=name, objetivo=t).to_csv(out_dir / f"univariate_channels_{name}_{t}.csv", index=False)
            s_gl[t] = summarize_rho(person_spearman(M_glob, d, col))
            all_global.append(s_gl[t].assign(grupo=name, objetivo=t))

        merged = s_gl["Bedford"][["feature", "rho_media", "p_fdr"]].merge(
            s_gl["N-back"][["feature", "rho_media", "p_fdr"]], on="feature", suffixes=("_Bedford", "_Nback"))
        merged = merged.reindex(merged[["rho_media_Bedford", "rho_media_Nback"]].abs().max(axis=1)
                                .sort_values(ascending=False).index)
        print("\n  Features globales (media entre canales): rho con cada objetivo")
        print(merged.round(3).to_string(index=False))
        for t in TARGETS:
            n_sig = int((sum_ch[t][name]["p_fdr"] < 0.05).sum())
            print(f"  Features canal x tipo con FDR<0.05 con {t}: {n_sig} de {len(sum_ch[t][name])}")

        # --- multivariante: mismo modelo con cada objetivo
        res = {}
        for t, col in TARGETS.items():
            pred, y, rhos, row = predict_target(name, t, col, d, M_all, M_glob)
            res[t] = (pred, y, rhos)
            ml_rows.append(row)
            ml_plot[t][name] = (pred, y, rhos)
            rho_rows.append(rhos.rename(f"{name}|{t}"))

        # --- comparación
        cmp_rows.append(compare_targets(name, d, res))

    if all_global:
        pd.concat(all_global).to_csv(out_dir / "univariate_global.csv", index=False)
    for t in TARGETS:
        if sum_ch[t]:
            heatmap(sum_ch[t], out_dir, t)
    if ml_rows:
        pd.DataFrame(ml_rows).to_csv(out_dir / "ml_por_objetivo.csv", index=False)
        pd.DataFrame(cmp_rows).to_csv(out_dir / "comparacion_Bedford_vs_Nback.csv", index=False)
        pd.concat(rho_rows, axis=1).to_csv(out_dir / "ml_rho_por_persona.csv")
        for t in TARGETS:
            dose_response_plot(ml_plot[t], out_dir, t)
    print(f"\nResultados en: {out_dir}")


if __name__ == "__main__":
    for f in [BEDFORD_CSV] + [FEATURE_DIR / f"features_{m}.csv" for m in METHODS]:
        if not f.exists():
            raise FileNotFoundError(f"No existe: {f}")
    for m in METHODS:
        run_method(m)
    print("\nTERMINADO")