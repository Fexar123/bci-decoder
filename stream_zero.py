"""Живой курсор БЕЗ калибровки: модель на 8 чужих мозгах + выравнивание на ходу.

Запуск (из projects/bci, после stream_sim.py):
    .venv/Scripts/python stream_zero.py

Для каждого «нового» человека:
  * учимся на 8 других (оба дня). Окна по 2 с из времени воображения, как в
    stream_sim.py. Каждая сессия каждого человека выровнена своей средней
    ковариацией по всей записи (окна каждые 0,5 с, ответы не нужны);
  * его день 2 идёт потоком: причинный фильтр, окно 2 с, решение каждые 0,1 с.
    Выравнивание R считается только по прошлому, ответы не нужны.
    Режимы R:
      warm20 — среднее за первые 20 с, дальше заморожено;
      cumul  — среднее за всё время с начала;
      ema60 / ema180 — скользящее, «забывает» старое (постоянная 60 / 180 с);
      none   — без выравнивания (просто чужая модель).
  * own — своя модель на его дне 1 (как в stream_sim.py, только на ковариациях).
В app/data/stream_sXX.json дописывается блок "zero" (режим ZERO_MODE).
Итог -> results/stream_zero.json
"""

import bci_paths  # noqa: F401  — первым: пути MNE/MOABB внутри проекта

import json
import warnings

import mne
import numpy as np
from moabb.datasets import BNCI2014_001
from scipy.signal import butter, sosfilt

from calibration import csp_feats, csp_fit, inv_sqrt, lda
from csp_lda_baseline import FMAX, FMIN
from stream_sim import (APP_DATA, CODES, STEP_S, TRAIN_STARTS_S, WIN_S, cursor_trials,
                        day_signal)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

MODES = ("own", "none", "warm20", "cumul", "ema180", "ema60")
ZERO_MODE = "cumul"      # что показываем в приложении (выбрано до сравнения: самое простое)
R_EVERY_S = 0.5


def window_covs(f, ends, win, chunk=800):
    out = np.empty((len(ends), f.shape[0], f.shape[0]))
    for a in range(0, len(ends), chunk):
        e = ends[a:a + chunk]
        idx = e[:, None] - win + 1 + np.arange(win)[None, :]          # (k, win)
        seg = f[:, idx].transpose(1, 0, 2)                             # (k, ch, win)
        out[a:a + chunk] = np.einsum("kct,kdt->kcd", seg, seg) / win
    return out


def subject_cache(s, sos, fs_expected=250.0):
    sessions = BNCI2014_001().get_data(subjects=[s])[s]
    days = {}
    for di, key in enumerate(sorted(sessions), start=1):
        x, names, fs, ev = day_signal(sessions, key)
        assert fs == fs_expected
        f = sosfilt(sos, x, axis=1)
        win = int(WIN_S * fs)
        ends = []
        ys = []
        for e, _, code in ev:
            if code in (1, 2):
                for st in TRAIN_STARTS_S:
                    ends.append(e + int(st * fs) + win - 1)
                    ys.append(int(code == 2))
        C = window_covs(f, np.array(ends), win)
        R = window_covs(f, np.arange(win - 1, f.shape[1], int(R_EVERY_S * fs)), win).mean(0)
        days[di] = {"C": C, "y": np.array(ys), "R": R}
    return days


def aligned(C, R):
    P = inv_sqrt(R)
    return P @ C @ P.T


def online_R(Cs, t_steps, mode):
    """R на каждом шаге только по прошлому (включая текущее окно)."""
    n = len(Cs)
    R = np.empty_like(Cs)
    if mode == "cumul":
        R[:] = np.cumsum(Cs, axis=0) / np.arange(1, n + 1)[:, None, None]
    elif mode == "warm20":
        cum = np.cumsum(Cs, axis=0) / np.arange(1, n + 1)[:, None, None]
        k = int(np.searchsorted(t_steps, t_steps[0] + 20.0))
        R[:k] = cum[:k]
        R[k:] = cum[k - 1]
    elif mode.startswith("ema"):
        a = STEP_S / float(mode[3:])
        r = Cs[0].copy()
        for i in range(n):
            # первые шаги — накопительное среднее, дальше «забывание»
            w = max(a, 1.0 / (i + 1))
            r = (1 - w) * r + w * Cs[i]
            R[i] = r
    return R


def stream_scores(W, clf, norm, Cs, R=None):
    if R is not None:
        v, U = np.linalg.eigh(R)
        P = U @ (v[:, :, None] ** -0.5 * np.transpose(U, (0, 2, 1)))   # R^{-1/2} батчем
        Cs = P @ Cs @ np.transpose(P, (0, 2, 1))
    return clf.decision_function(csp_feats(Cs, W)) / norm


def train(Cs, ys):
    W = csp_fit(Cs, ys)
    clf = lda().fit(csp_feats(Cs, W), ys)
    return W, clf, float(np.std(clf.decision_function(csp_feats(Cs, W))))


def main():
    ds = BNCI2014_001()
    fs = 250.0
    sos = butter(4, [FMIN, FMAX], btype="band", fs=fs, output="sos")
    cache = {}
    for s in ds.subject_list:
        cache[s] = subject_cache(s, sos)
        print(f"S{s}: кэш готов", flush=True)

    results = {}
    win, step = int(WIN_S * fs), int(round(STEP_S * fs))
    for t in ds.subject_list:
        others = [s for s in ds.subject_list if s != t]
        Cpool = np.concatenate([aligned(cache[s][d]["C"], cache[s][d]["R"]) for s in others for d in (1, 2)])
        ypool = np.concatenate([cache[s][d]["y"] for s in others for d in (1, 2)])
        Craw = np.concatenate([cache[s][d]["C"] for s in others for d in (1, 2)])
        models = {"zero": train(Cpool, ypool), "none": train(Craw, ypool),
                  "own": train(cache[t][1]["C"], cache[t][1]["y"])}

        sessions = ds.get_data(subjects=[t])[t]
        x2, _, _, ev2 = day_signal(sessions, sorted(sessions)[1])
        f2 = sosfilt(sos, x2, axis=1)
        ends = np.arange(win - 1, f2.shape[1], step)
        t_steps = (ends + 1) / fs
        Cs = window_covs(f2, ends, win)

        row, scores = {}, {}
        for mode in MODES:
            if mode == "own":
                sc = stream_scores(*models["own"], Cs)
            elif mode == "none":
                sc = stream_scores(*models["none"], Cs)
            else:
                sc = stream_scores(*models["zero"], Cs, online_R(Cs, t_steps, mode))
            _, hits, counted = cursor_trials(sc, t_steps, ev2, fs, "right_hand")
            row[mode] = hits / counted
            scores[mode] = sc
        results[t] = row
        print(f"S{t}: " + " · ".join(f"{m} {v:.0%}" for m, v in row.items()), flush=True)

        # в приложение — выбранный заранее режим
        path = APP_DATA / f"stream_s{t:02d}.json"
        d = json.loads(path.read_text("utf-8"))
        assert len(d["score"]) == len(scores[ZERO_MODE]), "сетка шагов не совпала со stream_sim"
        trials, hits, counted = cursor_trials(scores[ZERO_MODE], t_steps, ev2, fs, "right_hand")
        d["zero"] = {"mode": ZERO_MODE, "score": np.round(scores[ZERO_MODE], 2).tolist(),
                     "trials": trials, "hits": hits, "counted": counted}
        path.write_text(json.dumps(d, separators=(",", ":")), encoding="utf-8")

    mean = {m: float(np.mean([results[s][m] for s in results])) for m in MODES}
    print("\nсреднее: " + " · ".join(f"{m} {v:.1%}" for m, v in mean.items()))
    (bci_paths.RESULTS / "stream_zero.json").write_text(json.dumps(
        {"modes": MODES, "app_mode": ZERO_MODE, "per_subject": results, "mean": mean},
        ensure_ascii=False, indent=1), encoding="utf-8")

    idx_path = APP_DATA / "stream_index.json"
    idx = json.loads(idx_path.read_text("utf-8"))
    for e in idx:
        e["zero"] = round(results[e["subject"]][ZERO_MODE], 4)
    idx_path.write_text(json.dumps(idx, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
