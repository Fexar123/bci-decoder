"""Базовая модель: фильтр 8–30 Гц -> эпохи -> CSP -> LDA на BCI Competition IV 2a.

Запуск (из projects/bci):
    .venv/Scripts/python csp_lda_baseline.py --subject 1

Что считает для одного испытуемого:
  * 2 класса (левая / правая рука) и 4 класса (+ ноги, язык);
  * точность двумя способами:
      - cv5      — 5-кратная кросс-валидация по всем пробам обеих сессий;
      - T->E     — учим на сессии 1 (train), проверяем на сессии 2 (eval), как в соревновании;
  * топокарты CSP-паттернов -> figures/.
"""

import bci_paths  # noqa: F401  — первым: пути MNE/MOABB внутри проекта

import argparse
import json
import warnings
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.decoding import CSP
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery, MotorImagery
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", category=FutureWarning)

FMIN, FMAX = 8.0, 30.0  # mu (8–13) + beta (13–30): тут живёт ERD
MARK = ("C3", "Cz", "C4")
RU = {"left_hand": "лев. рука", "right_hand": "прав. рука",
      "feet": "ноги", "tongue": "язык"}


def make_pipe(n_classes):
    # 2 класса: 'alternate' даёт пары фильтров «вверх у левой / вверх у правой»
    order = "alternate" if n_classes == 2 else "mutual_info"
    csp = CSP(n_components=6 if n_classes == 2 else 8, reg=None, log=True,
              component_order=order)
    return make_pipeline(csp, LinearDiscriminantAnalysis())


def evaluate(epochs, y, meta, n_classes):
    X = epochs.get_data(copy=False)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(make_pipe(n_classes), X, y, cv=cv)

    sessions = sorted(meta["session"].unique())
    tr = (meta["session"] == sessions[0]).to_numpy()
    te = (meta["session"] == sessions[1]).to_numpy()
    te_score = make_pipe(n_classes).fit(X[tr], y[tr]).score(X[te], y[te])

    return {
        "n_classes": n_classes,
        "classes": [str(c) for c in sorted(set(y))],
        "n_trials": int(len(y)),
        "chance": round(1 / n_classes, 3),
        "cv5_mean": round(float(cv_scores.mean()), 3),
        "cv5_std": round(float(cv_scores.std()), 3),
        "train_to_eval": round(float(te_score), 3),
    }


def plot_patterns(epochs, y, n_classes, path, title):
    """Топокарты CSP-паттернов. Белые кружки — C3, Cz, C4."""
    X = epochs.get_data(copy=False)
    pipe = make_pipe(n_classes).fit(X, y)
    csp = pipe[0]
    feats = csp.transform(X)  # log-мощность каждого CSP-компонента
    classes = sorted(set(y))

    info = epochs.info
    if info.get_montage() is None:
        epochs.set_montage("standard_1020")
    names = [ch if ch in MARK else "" for ch in info.ch_names]
    mask = np.array([ch in MARK for ch in info.ch_names])[:, None]

    n = csp.n_components
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n, 2.9))
    for i, ax in enumerate(axes):
        mne.viz.plot_topomap(
            csp.patterns_[i], info, axes=ax, show=False, names=names,
            mask=mask, mask_params=dict(marker="", markersize=0),
            cmap="RdBu_r", contours=4,
        )
        for t in ax.texts:  # подписи C3/Cz/C4 — жирные, на белой плашке
            t.set(fontsize=8, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.15", fc="w", ec="k", lw=0.6))
        # у какого класса мощность компонента ниже всего = где ERD
        means = [feats[np.array(y) == c, i].mean() for c in classes]
        low = classes[int(np.argmin(means))]
        ax.set_title(f"CSP {i + 1}\nмощн. ↓ у: {RU.get(low, low)}", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1)
    args = ap.parse_args()
    s = args.subject
    dataset = BNCI2014_001()

    out = {"date": str(date.today()), "dataset": "BNCI2014_001 (BCI IV 2a)",
           "subject": s, "band_hz": [FMIN, FMAX], "pipeline": "CSP+LDA"}

    for n_classes, paradigm, tag in (
        (2, LeftRightImagery(fmin=FMIN, fmax=FMAX), "lr"),
        (4, MotorImagery(n_classes=4, fmin=FMIN, fmax=FMAX), "4class"),
    ):
        epochs, y, meta = paradigm.get_data(dataset, subjects=[s],
                                            return_epochs=True)
        y = np.asarray(y)
        res = evaluate(epochs, y, meta, n_classes)
        res["epoch_s"] = [round(float(epochs.tmin), 2), round(float(epochs.tmax), 2)]
        res["sfreq"] = float(epochs.info["sfreq"])
        res["n_channels"] = len(epochs.ch_names)
        out[tag] = res
        print(f"\n== S{s}, {n_classes} класса: {res['classes']}")
        print(f"   проб: {res['n_trials']}, каналов: {res['n_channels']}, "
              f"эпоха {res['epoch_s'][0]:g}–{res['epoch_s'][1]:g} c от начала пробы, "
              f"угадайка: {res['chance']:.0%}")
        print(f"   5-fold CV:  {res['cv5_mean']:.1%} ± {res['cv5_std']:.1%}")
        print(f"   T -> E:     {res['train_to_eval']:.1%}")

        fig_path = bci_paths.FIGURES / f"csp_patterns_s{s:02d}_{tag}.png"
        plot_patterns(epochs, y, n_classes, fig_path,
                      f"S{s}: CSP-паттерны, {n_classes} класса, {FMIN:g}–{FMAX:g} Гц")
        print(f"   топокарты:  {fig_path.relative_to(bci_paths.ROOT)}")

    res_path = bci_paths.RESULTS / f"baseline_s{s:02d}.json"
    res_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nцифры: {res_path.relative_to(bci_paths.ROOT)}")


if __name__ == "__main__":
    main()
