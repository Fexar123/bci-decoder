"""Калибровка v2: вес своих проб + сколько нужно «разогрева».

Запуск (из projects/bci):
    .venv/Scripts/python calibration_weighted.py

Продолжение calibration.py (тот же протокол, тот же свой CSP).
Часть A — вес своих проб. Доля своих в CSP и LDA:
  ea      — как в v1: свои пробы просто добавлены к 2304 чужим;
  ea_half — фиксированно 50% своих, 50% чужих (при N > 0);
  ea_auto — правило, заданное заранее: доля = N / (N + 10).
Часть B — разогрев: N = 0, выравнивание дня 2 по первым K пробам без ответов,
  K = 5, 10, 20, 40 и «весь день» (потолок, в жизни так нельзя).
  Проверка во всех K — одни и те же пробы дня 2 с 41-й.
Итог -> results/calibration_weighted.json, figures/calibration_weighted.png
"""

import bci_paths  # noqa: F401  — первым: пути MNE/MOABB внутри проекта

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from calibration import (N_COMP, N_LIST, REPS, TRIAL_S, WARMUP, align, csp_feats,
                         fit_score, lda, load_all)

K_LIST = [5, 10, 20, 40, "all"]
K_TEST_FROM = 40


def share_auto(n):
    return n / (n + 10)


def fit_score_w(Cs, ys, Cc, yc, share, Cte, yte):
    """CSP и LDA, где свои пробы (Cc) весят share, чужие (Cs) — 1 - share."""
    if len(yc) == 0 or share <= 0:
        return fit_score(Cs, ys, Cte, yte)
    cls = [(1 - share) * Cs[ys == k].mean(0) + share * Cc[yc == k].mean(0) for k in (0, 1)]
    vals, vecs = eigh(cls[0], cls[0] + cls[1])
    order = np.argsort(vals)
    W = vecs[:, np.r_[order[: N_COMP // 2], order[-(N_COMP // 2):]]]
    Fs, Fc = csp_feats(Cs, W), csp_feats(Cc, W)
    # вес в LDA — повтором своих проб, чтобы их доля была share
    rep = max(1, int(round(share / (1 - share) * len(Fs) / len(Fc)))) if share < 1 else 1
    F = np.r_[Fs, np.repeat(Fc, rep, axis=0)]
    y = np.r_[ys, np.repeat(yc, rep)]
    return float(lda().fit(F, y).score(csp_feats(Cte, W), yte))


def prepare(subj):
    src_ea = {}
    for s, d in subj.items():
        Ca = np.empty_like(d["C"])
        for day in (1, 2):
            m = d["day"] == day
            Ca[m] = align(d["C"][m], d["C"][m].mean(0))
        src_ea[s] = (Ca, d["y"])
    return src_ea


def part_a(subj, src_ea):
    rng = np.random.default_rng(0)
    methods = ("own", "ea", "ea_half", "ea_auto")
    per = {}
    for t, d in subj.items():
        others = [s for s in subj if s != t]
        Ce = np.concatenate([src_ea[s][0] for s in others]); ye = np.concatenate([src_ea[s][1] for s in others])
        d1, d2 = np.where(d["day"] == 1)[0], np.where(d["day"] == 2)[0]
        warm, test = d2[:WARMUP], d2[WARMUP:]
        Ct, yt = d["C"][test], d["y"][test]
        Ct_ea = align(Ct, d["C"][warm].mean(0))
        ps = {m: {} for m in methods}
        for n in N_LIST:
            acc = {m: [] for m in methods}
            for _ in range(1 if n in (0, 72) else REPS):
                cal = np.r_[[rng.choice(d1[d["y"][d1] == c], n, replace=False) for c in (0, 1)]].ravel().astype(int)
                Cc, yc = d["C"][cal], d["y"][cal]
                if n >= 2:
                    acc["own"].append(fit_score(Cc, yc, Ct, yt))
                Cc_ea = align(Cc, d["C"][np.unique(np.r_[d1[:WARMUP], cal])].mean(0))
                acc["ea"].append(fit_score(np.r_[Ce, Cc_ea], np.r_[ye, yc], Ct_ea, yt))
                acc["ea_half"].append(fit_score_w(Ce, ye, Cc_ea, yc, .5 if n else 0, Ct_ea, yt))
                acc["ea_auto"].append(fit_score_w(Ce, ye, Cc_ea, yc, share_auto(n), Ct_ea, yt))
            for m in methods:
                ps[m][n] = float(np.mean(acc[m])) if acc[m] else None
        per[t] = ps
        print(f"S{t}: " + " | ".join(f"{m}: " + " ".join("—" if ps[m][n] is None else f"{ps[m][n]:.0%}" for n in N_LIST)
                                     for m in methods), flush=True)
    mean = {m: {n: (float(np.mean([per[s][m][n] for s in per])) if per[1][m][n] is not None else None)
                for n in N_LIST} for m in methods}
    return mean, per


def part_b(subj, src_ea):
    per = {}
    for t, d in subj.items():
        others = [s for s in subj if s != t]
        Ce = np.concatenate([src_ea[s][0] for s in others]); ye = np.concatenate([src_ea[s][1] for s in others])
        d2 = np.where(d["day"] == 2)[0]
        test = d2[K_TEST_FROM:]
        Ct, yt = d["C"][test], d["y"][test]
        row = {}
        for k in K_LIST:
            ref = d["C"][d2].mean(0) if k == "all" else d["C"][d2[:k]].mean(0)
            row[str(k)] = fit_score(Ce, ye, align(Ct, ref), yt)
        row["no_align"] = fit_score(np.concatenate([subj[s]["C"] for s in others]),
                                    np.concatenate([subj[s]["y"] for s in others]), Ct, yt)
        per[t] = row
        print(f"S{t} разогрев: " + " · ".join(f"{k}: {v:.0%}" for k, v in row.items()), flush=True)
    mean = {k: float(np.mean([per[s][k] for s in per])) for k in per[1]}
    return mean, per


def plot(mean_a, per_a, mean_b):
    labels = {"own": "только свои", "ea": "выравнивание (v1)",
              "ea_half": "выравнивание + 50% вес своим", "ea_auto": "выравнивание + вес N/(N+10)"}
    colors = {"own": "#94a3b8", "ea": "#c4b5fd", "ea_half": "#60a5fa", "ea_auto": "#7c3aed"}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.6, 1]})
    xs = np.arange(len(N_LIST))
    for m in labels:
        ys = [mean_a[m][n] for n in N_LIST]
        idx = [i for i, v in enumerate(ys) if v is not None]
        ax.plot(xs[idx], [ys[i] * 100 for i in idx], "-o", color=colors[m], lw=2.4, label=labels[m])
        if m in ("own", "ea_auto"):
            for i in idx:
                ax.text(xs[i], ys[i] * 100 + 1.1, f"{ys[i]:.0%}", ha="center", fontsize=8.5,
                        color=colors[m], fontweight="bold")
    ax.axhline(50, color="r", ls="--", lw=1)
    ax.set_xticks(xs, [f"{n}\n≈{n * 2 * TRIAL_S / 60:.1f} мин" if n else "0" for n in N_LIST])
    ax.set_xlabel("своих проб на руку")
    ax.set_ylabel("точность на дне 2, %")
    ax.set_ylim(45, 90)
    ax.set_title("A. Вес своих проб (среднее по 9)")
    ax.legend(loc="lower right", fontsize=9)

    keys = [str(k) for k in K_LIST] + ["no_align"]
    names = [f"{k} проб" for k in K_LIST[:-1]] + ["весь день\n(потолок)", "без\nвыравнивания"]
    bx.bar(range(len(keys)), [mean_b[k] * 100 for k in keys],
           color=["#7c3aed"] * (len(keys) - 2) + ["#c4b5fd", "#94a3b8"])
    for i, k in enumerate(keys):
        bx.text(i, mean_b[k] * 100 + .6, f"{mean_b[k]:.1%}", ha="center", fontsize=9, fontweight="bold")
    bx.set_xticks(range(len(keys)), names, fontsize=9)
    bx.set_ylim(50, 85)
    bx.set_title("B. Сколько разогрева нужно (0 своих проб)")
    fig.tight_layout()
    fig.savefig(bci_paths.FIGURES / "calibration_weighted.png", dpi=120)


def main():
    subj = load_all()
    src_ea = prepare(subj)
    print("\n— A: вес своих проб —")
    mean_a, per_a = part_a(subj, src_ea)
    print("\nN на класс:   " + "  ".join(f"{n:>5}" for n in N_LIST))
    for m, row in mean_a.items():
        print(f"{m:<12} " + "  ".join("    —" if row[n] is None else f"{row[n]:5.1%}" for n in N_LIST))
    print("\n— B: разогрев —")
    mean_b, per_b = part_b(subj, src_ea)
    print("среднее: " + " · ".join(f"{k}: {v:.1%}" for k, v in mean_b.items()))
    (bci_paths.RESULTS / "calibration_weighted.json").write_text(json.dumps(
        {"n_list": N_LIST, "A_mean": mean_a, "A_per_subject": per_a,
         "B_mean": mean_b, "B_per_subject": per_b}, ensure_ascii=False, indent=1), encoding="utf-8")
    plot(mean_a, per_a, mean_b)


if __name__ == "__main__":
    main()
