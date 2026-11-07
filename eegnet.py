"""EEGNet против CSP+LDA на BCI Competition IV 2a (левая / правая рука).

Запуск (из projects/bci):
    .venv/Scripts/python eegnet.py within            # на каждом человеке: день 1 -> день 2
    .venv/Scripts/python eegnet.py cross             # на 8 чужих + выравнивание -> девятый
    .venv/Scripts/python eegnet.py within --subjects 1 --seeds 1   # пробный прогон
    .venv/Scripts/python eegnet.py plot

EEGNet-8,2 по статье Lawhern et al., 2018 (J. Neural Eng.), на чистом PyTorch.
Вход: эпохи 2–6 с пробы, полоса 4–40 Гц (сеть сама выбирает частоты), 128 Гц.
Защита от зубрёжки: мало весов, dropout, ограничение размера весов (max-norm),
ранняя остановка по отложенным 20% учебных проб. День 2 в обучении и выборе
модели не участвует. Каждый замер — несколько запусков с разным случайным стартом.

within: учёба на дне 1 человека, проверка на всём дне 2 (как csp_lda_baseline.py).
cross:  учёба на 8 других (оба дня); каждая сессия выровнена своей средней
        ковариацией (Euclidean Alignment). День 2 нового человека выровнен по
        первым 10 пробам без ответов, проверка — с 11-й (как calibration.py).
Итог -> results/eegnet_within.json, results/eegnet_cross.json, figures/eegnet.png
"""

import bci_paths  # noqa: F401  — первым: пути MNE/MOABB внутри проекта

import argparse
import json
import time
import warnings

import mne
import numpy as np
import torch
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery
from torch import nn

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")
torch.set_num_threads(max(1, torch.get_num_threads()))

FMIN, FMAX, SFREQ = 4.0, 40.0, 128.0
WARMUP = 10
CLASSES = ("left_hand", "right_hand")


# ---------- модель ----------
class EEGNet(nn.Module):
    """Частотные фильтры -> смеси электродов -> признаки -> классификатор."""

    def __init__(self, n_ch, n_t, n_cls=2, F1=8, D=2, F2=16, k_len=64, drop=0.5):
        super().__init__()
        self.temporal = nn.Sequential(                      # 8 своих фильтров частот
            nn.Conv2d(1, F1, (1, k_len), padding=(0, k_len // 2), bias=False),
            nn.BatchNorm2d(F1))
        self.spatial = nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False)  # как CSP
        self.block1 = nn.Sequential(
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(drop))
        self.block2 = nn.Sequential(                        # признаки во времени
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(drop))
        with torch.no_grad():
            n_feat = self._features(torch.zeros(1, 1, n_ch, n_t)).shape[1]
        self.classify = nn.Linear(n_feat, n_cls)

    def _features(self, x):
        return self.block2(self.block1(self.spatial(self.temporal(x)))).flatten(1)

    def forward(self, x):
        return self.classify(self._features(x))

    def max_norm(self):
        # ограничение размера весов — ещё одна защита от зубрёжки
        for w, lim in ((self.spatial.weight, 1.0), (self.classify.weight, 0.25)):
            n = w.data.flatten(1).norm(dim=1, keepdim=True).clamp(min=lim)
            w.data = (w.data.flatten(1) * (lim / n)).view_as(w.data)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------- данные ----------
def load_subject(s):
    par = LeftRightImagery(fmin=FMIN, fmax=FMAX, resample=SFREQ)
    X, y, meta = par.get_data(BNCI2014_001(), subjects=[s])
    sess = sorted(meta["session"].unique())
    return {"X": (X * 1e6).astype(np.float32),
            "y": (np.asarray(y).astype(str) == CLASSES[1]).astype(np.int64),
            "day": np.where(meta["session"] == sess[0], 1, 2)}


def ea(X, ref=None):
    """Euclidean Alignment в сигналах: X -> R^{-1/2} X, R — средняя ковариация."""
    ref = X if ref is None else ref
    R = np.einsum("nct,ndt->cd", ref, ref) / (ref.shape[0] * ref.shape[2])
    v, U = np.linalg.eigh(R)
    return np.einsum("cd,ndt->nct", (U * v ** -0.5) @ U.T, X).astype(np.float32)


def standardize(Xtr, *others):
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
    return [(Xtr - mu) / sd] + [(o - mu) / sd for o in others]


# ---------- обучение ----------
def train_eval(Xtr, ytr, Xte, yte, seed, drop, max_epochs, patience, batch=32, log=None):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    # 20% учебных проб — на раннюю остановку, по классам поровну
    val = np.concatenate([rng.choice(np.where(ytr == c)[0], int(round(.2 * (ytr == c).sum())), replace=False)
                          for c in (0, 1)])
    fit = np.setdiff1d(np.arange(len(ytr)), val)
    Xf, Xv, Xt = standardize(Xtr[fit], Xtr[val], Xte)
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(1)  # noqa: E731
    Xf, Xv, Xt = T(Xf), T(Xv), T(Xt)
    yf, yv = torch.from_numpy(ytr[fit]), torch.from_numpy(ytr[val])

    model = EEGNet(Xf.shape[2], Xf.shape[3], drop=drop)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    best, best_state, wait, best_ep, train_acc = -1.0, None, 0, 0, 0.0
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(yf))
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            lossf(model(Xf[b]), yf[b]).backward()
            opt.step()
            model.max_norm()
        model.eval()
        with torch.no_grad():
            v_acc = (model(Xv).argmax(1) == yv).float().mean().item()
        if v_acc > best:
            best, wait, best_ep = v_acc, 0, ep
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_acc = (model(Xf).argmax(1) == yf).float().mean().item()
        test_acc = (model(Xt).argmax(1) == torch.from_numpy(yte)).float().mean().item()
    return {"test": test_acc, "train": train_acc, "val": best, "best_epoch": best_ep, "epochs": ep + 1}


def save(name, obj):
    (bci_paths.RESULTS / f"eegnet_{name}.json").write_text(
        json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def run_within(subjects, seeds):
    out = {"protocol": "день 1 -> весь день 2, 4–40 Гц, 128 Гц, dropout 0.5, до 300 эпох, терпение 60",
           "per_subject": {}}
    for s in subjects:
        d = load_subject(s)
        tr, te = d["day"] == 1, d["day"] == 2
        runs = []
        for seed in range(seeds):
            t0 = time.time()
            r = train_eval(d["X"][tr], d["y"][tr], d["X"][te], d["y"][te], seed, .5, 300, 60)
            r["sec"] = round(time.time() - t0, 1)
            runs.append(r)
            print(f"S{s} seed {seed}: экзамен {r['test']:.1%} · учёба {r['train']:.1%} · "
                  f"лучшая эпоха {r['best_epoch']} · {r['sec']} с", flush=True)
        out["per_subject"][s] = {"runs": runs,
                                 "mean": float(np.mean([r["test"] for r in runs])),
                                 "std": float(np.std([r["test"] for r in runs]))}
        save("within", out)
    return out


def run_cross(subjects, seeds):
    data = {s: load_subject(s) for s in BNCI2014_001().subject_list}
    aligned = {}
    for s, d in data.items():
        Xa = np.empty_like(d["X"])
        for day in (1, 2):
            m = d["day"] == day
            Xa[m] = ea(d["X"][m])
        aligned[s] = Xa
    out = {"protocol": "8 чужих (оба дня, EA по сессиям) -> день 2 нового без первых 10 проб; "
                       "EA нового по 10 пробам разогрева; dropout 0.25, до 100 эпох, терпение 20",
           "per_subject": {}}
    done_path = bci_paths.RESULTS / "eegnet_cross.json"
    if done_path.exists():   # продолжить с места остановки
        out["per_subject"] = json.loads(done_path.read_text("utf-8"))["per_subject"]
    for t in subjects:
        if str(t) in out["per_subject"] and len(out["per_subject"][str(t)]["runs"]) >= seeds:
            print(f"S{t}: уже посчитан, пропускаю", flush=True)
            continue
        others = [s for s in data if s != t]
        d2 = np.where(data[t]["day"] == 2)[0]
        warm, test = d2[:WARMUP], d2[WARMUP:]
        Xte = ea(data[t]["X"][test], ref=data[t]["X"][warm])
        yte = data[t]["y"][test]
        Xtr = np.concatenate([aligned[s] for s in others])
        ytr = np.concatenate([data[s]["y"] for s in others])
        runs = []
        for seed in range(seeds):
            t0 = time.time()
            r = train_eval(Xtr, ytr, Xte, yte, seed, .25, 100, 20, batch=64)
            r["sec"] = round(time.time() - t0, 1)
            runs.append(r)
            print(f"S{t} seed {seed}: экзамен {r['test']:.1%} · учёба {r['train']:.1%} · "
                  f"лучшая эпоха {r['best_epoch']} · {r['sec']} с", flush=True)
        out["per_subject"][t] = {"runs": runs,
                                 "mean": float(np.mean([r["test"] for r in runs])),
                                 "std": float(np.std([r["test"] for r in runs]))}
        save("cross", out)
    return out


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    within = json.loads((bci_paths.RESULTS / "eegnet_within.json").read_text("utf-8"))["per_subject"]
    cross = json.loads((bci_paths.RESULTS / "eegnet_cross.json").read_text("utf-8"))["per_subject"]
    csp = json.loads((bci_paths.RESULTS / "electrodes.json").read_text("utf-8"))["per_subject"]
    cal = json.loads((bci_paths.RESULTS / "calibration.json").read_text("utf-8"))["per_subject"]
    subj = [str(s) for s in range(1, 10)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    pairs = [
        (axes[0], "На каждом человеке: учёба день 1 → экзамен день 2",
         [("CSP+LDA", [csp[s]["все 22"] for s in subj], None, "#94a3b8"),
          ("EEGNet", [within[s]["mean"] for s in subj], [within[s]["std"] for s in subj], "#7c3aed")]),
        (axes[1], "Без калибровки: 8 чужих + выравнивание → новый человек",
         [("CSP+LDA", [cal[s]["ea"]["0"] for s in subj], None, "#94a3b8"),
          ("EEGNet", [cross[s]["mean"] for s in subj], [cross[s]["std"] for s in subj], "#7c3aed")]),
    ]
    xs = np.arange(len(subj))
    for ax, title, series in pairs:
        for k, (name, vals, err, col) in enumerate(series):
            ax.bar(xs + (k - .5) * .38, np.array(vals) * 100, .38, yerr=None if err is None else np.array(err) * 100,
                   color=col, label=f"{name}: {np.mean(vals):.1%}", capsize=3)
        ax.axhline(50, color="r", ls="--", lw=1)
        ax.set_xticks(xs, [f"S{s}" for s in subj])
        ax.set_title(title, fontsize=11)
        ax.legend(loc="upper left")
    axes[0].set_ylabel("точность на дне 2, %")
    axes[0].set_ylim(40, 102)
    fig.suptitle("EEGNet против CSP+LDA · левая vs правая рука · усы — разброс по запускам EEGNet")
    fig.tight_layout()
    fig.savefig(bci_paths.FIGURES / "eegnet.png", dpi=120)
    print("figures/eegnet.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["within", "cross", "plot", "info"])
    ap.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    if a.part == "info":
        m = EEGNet(22, 513)
        print(f"EEGNet: {n_params(m)} весов")
    elif a.part == "within":
        run_within(a.subjects, a.seeds)
    elif a.part == "cross":
        run_cross(a.subjects, a.seeds)
    else:
        plot()


if __name__ == "__main__":
    main()
