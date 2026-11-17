"""Сравнение моделей для реального времени: CSP+LDA, Riemann (MDM, TS+LDA, TS+SVM), EEGNet.

Запуск (из projects/bci):
    .venv/Scripts/python compare.py classic          # все классические, 22 и 4 канала, оба протокола (минуты)
    .venv/Scripts/python compare.py eegnet           # EEGNet на 4 каналах (долго; 22 канала — из eegnet.py)
    .venv/Scripts/python compare.py speed            # время одного решения на окне 2 с
    .venv/Scripts/python compare.py table            # таблица + figures/compare.png

Левая vs правая рука, BCI IV 2a, 9 человек. Протоколы — как в этапе 3 (eegnet.py, calibration.py):
  within: учёба на дне 1 человека, проверка на всём дне 2;
  cross:  учёба на 8 других (оба дня), каждая сессия выровнена своей средней ковариацией
          (Euclidean Alignment); день 2 нового человека выровнен по первым 10 пробам без ответов,
          проверка — с 11-й. Ответы нового человека не видит ни модель, ни выравнивание.
Каналы: full — все 22, 250 Гц; ganglion — C3 Cz C4 CPz, 200 Гц (как OpenBCI Ganglion).
Классические модели: полоса 8–30 Гц, эпоха 2–6 с. EEGNet: 4–40 Гц, 128 Гц (как в eegnet.py).
Riemann: ковариация (OAS) -> MDM (ближайшее риманово среднее класса) или касательное
пространство -> LDA (со сжатием) / линейный SVM.
Итог -> results/compare.json, results/compare_eegnet_ganglion.json, figures/compare.png
"""

import bci_paths  # noqa: F401  — первым: пути MNE/MOABB внутри проекта

import argparse
import json
import time
import warnings

import mne
import numpy as np
from mne.decoding import CSP
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from eegnet import WARMUP, ea

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

CONFIGS = {"full": {"channels": None, "fs": 250}, "ganglion": {"channels": ["C3", "Cz", "C4", "CPz"], "fs": 200}}
SUBJECTS = list(range(1, 10))
OUT = bci_paths.RESULTS / "compare.json"
OUT_NN = bci_paths.RESULTS / "compare_eegnet_ganglion.json"
NAMES = {"csp_lda": "CSP+LDA", "mdm": "Riemann MDM", "ts_lda": "Riemann TS+LDA",
         "ts_svm": "Riemann TS+SVM", "eegnet": "EEGNet"}


def pipelines(n_ch):
    cov = lambda: Covariances(estimator="oas")  # noqa: E731
    return {
        "csp_lda": make_pipeline(CSP(n_components=min(6, n_ch), log=True), LDA()),
        "mdm": make_pipeline(cov(), MDM()),
        "ts_lda": make_pipeline(cov(), TangentSpace(), LDA(solver="lsqr", shrinkage="auto")),
        "ts_svm": make_pipeline(cov(), TangentSpace(), SVC(kernel="linear", C=1.0)),
    }


def load(s, cfg, nn=False):
    c = CONFIGS[cfg]
    kw = {"channels": c["channels"]} if c["channels"] else {}
    par = (LeftRightImagery(fmin=4.0, fmax=40.0, resample=128.0, **kw) if nn
           else LeftRightImagery(fmin=8.0, fmax=30.0, resample=float(c["fs"]), **kw))
    X, y, meta = par.get_data(BNCI2014_001(), subjects=[s])
    sess = sorted(meta["session"].unique())
    return {"X": (X * 1e6).astype(np.float32 if nn else np.float64),
            "y": (np.asarray(y).astype(str) == "right_hand").astype(np.int64),
            "day": np.where(meta["session"] == sess[0], 1, 2)}


def aligned_days(d):
    Xa = np.empty_like(d["X"])
    for day in (1, 2):
        m = d["day"] == day
        Xa[m] = ea(d["X"][m])
    return Xa


def cross_split(data, t):
    others = [s for s in data if s != t]
    d2 = np.where(data[t]["day"] == 2)[0]
    warm, test = d2[:WARMUP], d2[WARMUP:]
    Xtr = np.concatenate([aligned_days(data[s]) for s in others])
    ytr = np.concatenate([data[s]["y"] for s in others])
    Xte = ea(data[t]["X"][test], ref=data[t]["X"][warm])
    return Xtr, ytr, Xte, data[t]["y"][test]


def run_classic():
    out = json.loads(OUT.read_text("utf-8")) if OUT.exists() else {}
    for cfg in CONFIGS:
        data = {s: load(s, cfg) for s in SUBJECTS}
        n_ch = data[1]["X"].shape[1]
        res = {"within": {}, "cross": {}}
        for s, d in data.items():
            tr, te = d["day"] == 1, d["day"] == 2
            res["within"][s] = {m: float(p.fit(d["X"][tr], d["y"][tr]).score(d["X"][te], d["y"][te]))
                                for m, p in pipelines(n_ch).items()}
            Xtr, ytr, Xte, yte = cross_split(data, s)
            res["cross"][s] = {m: float(p.fit(Xtr, ytr).score(Xte, yte)) for m, p in pipelines(n_ch).items()}
            print(f"[{cfg}] S{s} within " + " ".join(f"{m} {v:.0%}" for m, v in res["within"][s].items())
                  + " · cross " + " ".join(f"{m} {v:.0%}" for m, v in res["cross"][s].items()), flush=True)
        out[cfg] = res
        OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def run_eegnet(seeds_within=3, seeds_cross=3):
    from eegnet import train_eval
    out = json.loads(OUT_NN.read_text("utf-8")) if OUT_NN.exists() else {"within": {}, "cross": {}}
    data = {s: load(s, "ganglion", nn=True) for s in SUBJECTS}
    for s, d in data.items():
        if str(s) not in out["within"]:
            tr, te = d["day"] == 1, d["day"] == 2
            runs = [train_eval(d["X"][tr], d["y"][tr], d["X"][te], d["y"][te], seed, .5, 300, 60)["test"]
                    for seed in range(seeds_within)]
            out["within"][str(s)] = {"runs": runs, "mean": float(np.mean(runs))}
            OUT_NN.write_text(json.dumps(out, indent=1), encoding="utf-8")
            print(f"[ganglion] EEGNet within S{s}: {np.mean(runs):.1%} {np.round(runs, 3).tolist()}", flush=True)
    for s in data:
        if str(s) not in out["cross"]:
            Xtr, ytr, Xte, yte = cross_split(data, s)
            runs = [train_eval(Xtr, ytr, Xte, yte, seed, .25, 100, 20, batch=64)["test"] for seed in range(seeds_cross)]
            out["cross"][str(s)] = {"runs": runs, "mean": float(np.mean(runs))}
            OUT_NN.write_text(json.dumps(out, indent=1), encoding="utf-8")
            print(f"[ganglion] EEGNet cross S{s}: {np.mean(runs):.1%} {np.round(runs, 3).tolist()}", flush=True)


def run_speed():
    """Время одного решения (окно 2 с) после обучения: медиана по 200 вызовам."""
    import torch
    from eegnet import EEGNet
    out = json.loads(OUT.read_text("utf-8")) if OUT.exists() else {}
    speed = {}
    for cfg, c in CONFIGS.items():
        d = load(1, cfg)
        tr = d["day"] == 1
        n_ch, fs = d["X"].shape[1], c["fs"]
        win = d["X"][:1, :, : 2 * fs]
        speed[cfg] = {}
        for m, p in pipelines(n_ch).items():
            p.fit(d["X"][tr][:, :, : 2 * fs], d["y"][tr])
            ts = []
            for _ in range(200):
                t0 = time.perf_counter(); p.predict(win); ts.append(time.perf_counter() - t0)
            speed[cfg][m] = float(np.median(ts) * 1000)
        net = EEGNet(n_ch, 256).eval()
        x = torch.zeros(1, 1, n_ch, 256)
        ts = []
        with torch.no_grad():
            for _ in range(200):
                t0 = time.perf_counter(); net(x); ts.append(time.perf_counter() - t0)
        speed[cfg]["eegnet"] = float(np.median(ts) * 1000)
        print(f"[{cfg}] мс на решение: " + " ".join(f"{m} {v:.2f}" for m, v in speed[cfg].items()), flush=True)
    out["speed_ms"] = speed
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def collect():
    res = json.loads(OUT.read_text("utf-8"))
    nn_full = {p: json.loads((bci_paths.RESULTS / f"eegnet_{p}.json").read_text("utf-8"))["per_subject"]
               for p in ("within", "cross")}
    nn_g = json.loads(OUT_NN.read_text("utf-8")) if OUT_NN.exists() else {"within": {}, "cross": {}}
    tab = {}
    for cfg in CONFIGS:
        tab[cfg] = {}
        for prot in ("within", "cross"):
            per = {m: [res[cfg][prot][str(s)][m] for s in SUBJECTS] for m in ("csp_lda", "mdm", "ts_lda", "ts_svm")}
            src = nn_full[prot] if cfg == "full" else nn_g[prot]
            if all(str(s) in src for s in SUBJECTS):
                per["eegnet"] = [src[str(s)]["mean"] for s in SUBJECTS]
            tab[cfg][prot] = per
    return tab, res.get("speed_ms", {})


def table():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    tab, speed = collect()
    titles = {"within": "свой день 1 → день 2", "cross": "без калибровки (8 чужих + EA)"}
    for cfg in tab:
        for prot, per in tab[cfg].items():
            print(f"\n[{cfg}] {titles[prot]}")
            best = max(per, key=lambda m: np.mean(per[m]))
            for m, v in per.items():
                p = "" if m == best else f" · против лучшего p={wilcoxon(per[best], v).pvalue:.3f}"
                print(f"  {NAMES[m]:<16} {np.mean(v):.1%}  " + " ".join(f"{x * 100:.0f}" for x in v) + p)
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.6), sharey=True)
    for ax, (cfg, prot) in zip(axs, [(c, p) for c in tab for p in ("within", "cross")]):
        per = tab[cfg][prot]
        ms = list(per)
        means = [np.mean(per[m]) * 100 for m in ms]
        ax.bar(range(len(ms)), means, color=["#94a3b8", "#60a5fa", "#38bdf8", "#22d3ee", "#a78bfa"][:len(ms)])
        for i, m in enumerate(ms):
            ax.scatter(np.full(9, i) + np.linspace(-.25, .25, 9), np.array(per[m]) * 100, s=9, color="k", alpha=.5)
            ax.text(i, means[i] + 1, f"{means[i]:.1f}", ha="center", fontsize=9)
        ax.set_xticks(range(len(ms)), [NAMES[m].replace("Riemann ", "R ") for m in ms], rotation=25, fontsize=9)
        ax.axhline(50, color="r", ls="--", lw=1)
        ax.set_title(f"{'22 канала' if cfg == 'full' else '4 канала (Ganglion)'}: {titles[prot]}", fontsize=10)
    axs[0].set_ylabel("точность на дне 2, % (точки — люди)")
    axs[0].set_ylim(40, 102)
    fig.tight_layout()
    fig.savefig(bci_paths.FIGURES / "compare.png", dpi=110)
    if speed:
        print("\nмс на одно решение (окно 2 с):", json.dumps(speed))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["classic", "eegnet", "speed", "table"])
    a = ap.parse_args()
    {"classic": run_classic, "eegnet": run_eegnet, "speed": run_speed, "table": table}[a.part]()


if __name__ == "__main__":
    main()
