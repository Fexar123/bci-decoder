"""Сколько электродов нужно? CSP+LDA на разных наборах каналов, все 9 человек.

Запуск (из projects/bci):
    .venv/Scripts/python electrodes.py

Протокол: 2 класса (левая/правая), учёба на дне 1, проверка на дне 2.
«Лучшие 4» подбираются жадно по 5-fold CV только внутри дня 1 —
день 2 в выборе не участвует, иначе было бы враньё.
Итог -> results/electrodes.json, figures/electrodes.png
"""

import bci_paths  # noqa: F401  — первым: пути MNE/MOABB внутри проекта

import json
import warnings
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.decoding import CSP
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from csp_lda_baseline import FMAX, FMIN

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

SETS = {
    "все 22": None,
    "8 · как Cyton": ["FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CP4"],
    "4 · как Ganglion": ["C3", "Cz", "C4", "CPz"],
    "4 · ≈Crown": ["C3", "C4", "CP3", "CP4"],
    "3 · C3 Cz C4": ["C3", "Cz", "C4"],
    "2 · C3 C4": ["C3", "C4"],
}
GREEDY_K = 4
GREEDY_NAME = "лучшие 4 (свои)"


def pipe(n_ch):
    k = min(6, n_ch)
    k -= k % 2  # 'alternate' берёт пары
    return make_pipeline(CSP(n_components=max(k, 2), log=True,
                             component_order="alternate"),
                         LinearDiscriminantAnalysis())


def t2e(X, y, tr, te, idx):
    return float(pipe(len(idx)).fit(X[tr][:, idx], y[tr]).score(X[te][:, idx], y[te]))


def greedy(X, y, names, k):
    """Добавляем по одному электроду, который больше всего поднимает CV на дне 1."""
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    chosen = [names.index("C3"), names.index("C4")]   # старт с пары — CSP нужно ≥2
    while len(chosen) < k:
        best, best_sc = None, -1
        for j in range(len(names)):
            if j in chosen:
                continue
            idx = chosen + [j]
            sc = cross_val_score(pipe(len(idx)), X[:, idx], y, cv=cv).mean()
            if sc > best_sc:
                best, best_sc = j, sc
        chosen.append(best)
    # честнее: дать отбору заменить и стартовую пару
    for pos in range(2):
        base = cross_val_score(pipe(k), X[:, chosen], y, cv=cv).mean()
        for j in range(len(names)):
            if j in chosen:
                continue
            idx = chosen.copy()
            idx[pos] = j
            sc = cross_val_score(pipe(k), X[:, idx], y, cv=cv).mean()
            if sc > base:
                chosen, base = idx, sc
    return chosen


def main():
    ds = BNCI2014_001()
    paradigm = LeftRightImagery(fmin=FMIN, fmax=FMAX)
    rows, picked = {}, Counter()
    for s in ds.subject_list:
        X, y, meta = paradigm.get_data(ds, subjects=[s])
        y = np.asarray(y).astype(str)
        epochs, _, _ = paradigm.get_data(ds, subjects=[s], return_epochs=True)
        names = epochs.ch_names
        sessions = sorted(meta["session"].unique())
        tr = (meta["session"] == sessions[0]).to_numpy()
        te = ~tr
        r = {}
        for name, chans in SETS.items():
            idx = list(range(len(names))) if chans is None else [names.index(c) for c in chans]
            r[name] = t2e(X, y, tr, te, idx)
        g = greedy(X[tr], y[tr], names, GREEDY_K)
        r[GREEDY_NAME] = t2e(X, y, tr, te, g)
        r["_greedy"] = [names[j] for j in g]
        picked.update(r["_greedy"])
        rows[s] = r
        print(f"S{s}: " + " · ".join(f"{k} {v:.0%}" for k, v in r.items() if not k.startswith("_"))
              + f" · свои: {', '.join(r['_greedy'])}", flush=True)

    keys = list(SETS) + [GREEDY_NAME]
    mean = {k: float(np.mean([rows[s][k] for s in rows])) for k in keys}
    print("\nсреднее: " + " · ".join(f"{k} {v:.1%}" for k, v in mean.items()))
    print("чаще всего выбирали:", picked.most_common(10))

    out = {"protocol": "2 класса, T->E, CSP+LDA, 8–30 Гц, эпоха 2–6 с",
           "sets": {k: v for k, v in SETS.items()}, "per_subject": rows,
           "mean": mean, "greedy_counts": dict(picked.most_common())}
    (bci_paths.RESULTS / "electrodes.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    # график: точки — люди, столбик — среднее
    fig, ax = plt.subplots(figsize=(11, 5.2))
    xs = np.arange(len(keys))
    ax.bar(xs, [mean[k] * 100 for k in keys], color="#a78bfa", alpha=.35, width=.65, zorder=1)
    for s in rows:
        ax.plot(xs, [rows[s][k] * 100 for k in keys], "-o", color="0.55", lw=.8, ms=4, alpha=.6, zorder=2)
        ax.text(xs[-1] + .12, rows[s][keys[-1]] * 100, f"S{s}", fontsize=8, va="center", color="0.4")
    for x, k in zip(xs, keys):
        ax.text(x, mean[k] * 100 + 1.5, f"{mean[k]:.1%}", ha="center", fontweight="bold")
    ax.axhline(50, color="r", ls="--", lw=1)
    ax.text(-.45, 51, "угадайка 50%", color="r", fontsize=9)
    ax.set_xticks(xs, keys, rotation=12)
    ax.set_ylim(40, 102)
    ax.set_ylabel("точность на дне 2, %")
    ax.set_title("Сколько электродов нужно: CSP+LDA, левая vs правая, 9 человек")
    fig.tight_layout()
    fig.savefig(bci_paths.FIGURES / "electrodes.png", dpi=120)


if __name__ == "__main__":
    main()
