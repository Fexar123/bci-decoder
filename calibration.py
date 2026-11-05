"""Быстрая калибровка: сколько проб нужно от нового человека?

Запуск (из projects/bci):
    .venv/Scripts/python calibration.py            # полный прогон
    .venv/Scripts/python calibration.py --check    # только сверка своего CSP с MNE

Протокол (leave-one-subject-out, 2 класса, 8–30 Гц, эпоха 2–6 с):
  * новый человек = один из 9, «чужие» = остальные 8 (оба дня);
  * калибровка = N случайных проб на класс из его дня 1, N = 0…72, 10 повторов;
  * проверка = его день 2 без первых 10 проб (они идут на «разогрев» без ответов).
Способы:
  own    — учимся только на его N пробах;
  naive  — чужие 8 как есть + его N проб;
  ea     — Euclidean Alignment (He & Wu, 2019): каждую сессию каждого человека
           «отбеливаем» её средней ковариацией. Ответы для этого не нужны:
           у нового человека — первые 10 проб дня 1 и его N проб (день 1),
           10 проб разогрева (день 2).
CSP свой, на ковариациях: так в сотни раз быстрее, чем гонять сигналы.
Итог -> results/calibration.json, figures/calibration.png
"""

import bci_paths  # noqa: F401  — первым: пути MNE/MOABB внутри проекта

import argparse
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.decoding import CSP
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline

from csp_lda_baseline import FMAX, FMIN

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

N_LIST = [0, 2, 5, 10, 20, 40, 72]
REPS = 10
WARMUP = 10          # проб без ответов для выравнивания
N_COMP = 6
TRIAL_S = 8.0        # одна проба ≈ 8 с -> минуты калибровки
CLASSES = ("left_hand", "right_hand")


# ---------- данные ----------
def load_all():
    ds, par = BNCI2014_001(), LeftRightImagery(fmin=FMIN, fmax=FMAX)
    subj = {}
    for s in ds.subject_list:
        X, y, meta = par.get_data(ds, subjects=[s])
        y = (np.asarray(y).astype(str) == CLASSES[1]).astype(int)   # 0 левая, 1 правая
        sess = sorted(meta["session"].unique())
        day = np.where(meta["session"] == sess[0], 1, 2)
        covs = np.einsum("nct,ndt->ncd", X, X) / X.shape[2]       # ковариация каждой пробы
        subj[s] = {"X": X, "y": y, "day": day, "C": covs}
        print(f"S{s}: {len(y)} проб", flush=True)
    return subj


# ---------- свой CSP на ковариациях ----------
def csp_fit(C, y, k=N_COMP):
    c0, c1 = C[y == 0].mean(0), C[y == 1].mean(0)
    vals, vecs = eigh(c0, c0 + c1)                   # обобщённая задача на собственные векторы
    order = np.argsort(vals)
    pick = np.r_[order[: k // 2], order[-(k // 2):]]  # крайние = самые разные между классами
    return vecs[:, pick]


def csp_feats(C, W):
    return np.log(np.einsum("ck,ncd,dk->nk", W, C, W))  # log мощности каждой смеси


def lda():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def fit_score(Ctr, ytr, Cte, yte):
    W = csp_fit(Ctr, ytr)
    clf = lda().fit(csp_feats(Ctr, W), ytr)
    return float(clf.score(csp_feats(Cte, W), yte))


# ---------- выравнивание ----------
def inv_sqrt(R):
    v, U = np.linalg.eigh(R)
    return U @ np.diag(v ** -0.5) @ U.T


def align(C, R):
    P = inv_sqrt(R)
    return P @ C @ P.T


def check(subj):
    """Свой CSP vs MNE CSP на полном протоколе T->E — должны совпадать по смыслу."""
    rows = []
    for s, d in subj.items():
        tr, te = d["day"] == 1, d["day"] == 2
        own = fit_score(d["C"][tr], d["y"][tr], d["C"][te], d["y"][te])
        mne_pipe = make_pipeline(CSP(n_components=N_COMP, log=True, component_order="alternate"),
                                 LinearDiscriminantAnalysis())
        ref = float(mne_pipe.fit(d["X"][tr], d["y"][tr]).score(d["X"][te], d["y"][te]))
        rows.append((own, ref))
        print(f"S{s}: свой CSP {own:.1%} · MNE CSP {ref:.1%}")
    o, r = np.mean(rows, 0)
    print(f"среднее: свой {o:.1%} · MNE {r:.1%}")


def run(subj):
    rng = np.random.default_rng(0)
    res = {m: {n: [] for n in N_LIST} for m in ("own", "naive", "ea")}
    per_subj = {}

    # чужие: выровненные ковариации каждой сессии каждого человека — заранее
    src_raw, src_ea = {}, {}
    for s, d in subj.items():
        src_raw[s] = (d["C"], d["y"])
        Ca = np.empty_like(d["C"])
        for day in (1, 2):
            m = d["day"] == day
            Ca[m] = align(d["C"][m], d["C"][m].mean(0))
        src_ea[s] = (Ca, d["y"])

    for t, d in subj.items():
        others = [s for s in subj if s != t]
        Cn = np.concatenate([src_raw[s][0] for s in others]); yn = np.concatenate([src_raw[s][1] for s in others])
        Ce = np.concatenate([src_ea[s][0] for s in others]); ye = np.concatenate([src_ea[s][1] for s in others])

        d1 = np.where(d["day"] == 1)[0]
        d2 = np.where(d["day"] == 2)[0]
        warm, test = d2[:WARMUP], d2[WARMUP:]
        Ct, yt = d["C"][test], d["y"][test]
        Ct_ea = align(Ct, d["C"][warm].mean(0))      # день 2: выравнивание по разогреву
        ps = {m: {} for m in res}

        for n in N_LIST:
            for m in res:
                ps[m][n] = []
            for rep in range(1 if n in (0, 72) else REPS):
                cal = np.r_[[rng.choice(d1[d["y"][d1] == c], n, replace=False) for c in (0, 1)]].ravel().astype(int)
                Cc, yc = d["C"][cal], d["y"][cal]
                if n >= 2:
                    ps["own"][n].append(fit_score(Cc, yc, Ct, yt))
                ps["naive"][n].append(fit_score(np.r_[Cn, Cc], np.r_[yn, yc], Ct, yt))
                ref_idx = np.unique(np.r_[d1[:WARMUP], cal])   # без ответов
                Cc_ea = align(Cc, d["C"][ref_idx].mean(0))
                ps["ea"][n].append(fit_score(np.r_[Ce, Cc_ea], np.r_[ye, yc], Ct_ea, yt))
            for m in res:
                if ps[m][n]:
                    v = float(np.mean(ps[m][n]))
                    ps[m][n] = v
                    res[m][n].append(v)
                else:
                    ps[m][n] = None
        per_subj[t] = ps
        print(f"S{t}: " + " | ".join(
            f"{m}: " + " ".join("—" if ps[m][n] is None else f"{ps[m][n]:.0%}" for n in N_LIST)
            for m in res), flush=True)

    mean = {m: {n: (float(np.mean(v)) if v else None) for n, v in res[m].items()} for m in res}
    print("\nN на класс:   " + "  ".join(f"{n:>5}" for n in N_LIST))
    for m in res:
        print(f"{m:<12} " + "  ".join("    —" if mean[m][n] is None else f"{mean[m][n]:5.1%}" for n in N_LIST))

    (bci_paths.RESULTS / "calibration.json").write_text(json.dumps(
        {"protocol": __doc__.split("Протокол")[1].split("CSP свой")[0].strip(),
         "n_list": N_LIST, "mean": mean, "per_subject": per_subj}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    plot(mean, per_subj)


def plot(mean, per_subj):
    labels = {"own": "только свои пробы", "naive": "чужие 8 мозгов + свои", "ea": "чужие + выравнивание + свои"}
    colors = {"own": "#94a3b8", "naive": "#60a5fa", "ea": "#a78bfa"}
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    xs = np.arange(len(N_LIST))
    for m in ("own", "naive", "ea"):
        ys = [mean[m][n] for n in N_LIST]
        idx = [i for i, v in enumerate(ys) if v is not None]
        lo = [min(per_subj[s][m][N_LIST[i]] for s in per_subj) * 100 for i in idx]
        hi = [max(per_subj[s][m][N_LIST[i]] for s in per_subj) * 100 for i in idx]
        ax.fill_between(xs[idx], lo, hi, color=colors[m], alpha=.08)
        ax.plot(xs[idx], [ys[i] * 100 for i in idx], "-o", color=colors[m], lw=2.5, label=labels[m])
        for i in idx:
            ax.text(xs[i], ys[i] * 100 + 1.2, f"{ys[i]:.0%}", ha="center", fontsize=8.5, color=colors[m], fontweight="bold")
    ax.axhline(50, color="r", ls="--", lw=1)
    ax.text(len(N_LIST) - 1, 51, "угадайка", color="r", ha="right", fontsize=9)
    ax.set_xticks(xs, [f"{n}\n≈{n * 2 * TRIAL_S / 60:.1f} мин" if n else "0\nбез калибровки" for n in N_LIST])
    ax.set_xlabel("проб на каждую руку от нового человека (и сколько это времени)")
    ax.set_ylabel("точность на дне 2, %")
    ax.set_ylim(40, 100)
    ax.set_title("Быстрая калибровка: 9 человек, каждый по очереди «новый». Полосы — разброс по людям")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(bci_paths.FIGURES / "calibration.png", dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    subj = load_all()
    if args.check:
        check(subj)
    else:
        run(subj)


if __name__ == "__main__":
    main()
