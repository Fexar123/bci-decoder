"""Симуляция онлайн-BCI: запись дня 2 проигрывается как прямой эфир.

Запуск (из projects/bci):
    .venv/Scripts/python stream_sim.py              # все 9 -> app/data/stream_sXX.json
    .venv/Scripts/python stream_sim.py --subjects 8

Честность «как вживую»:
  * фильтр 8–30 Гц причинный (Баттерворт, sosfilt): выход в момент t зависит
    только от сигнала до t — как в реальном времени;
  * модель видит только последние WIN_S секунд, решение каждые STEP_S секунд;
  * учится только на дне 1 (окна по WIN_S из времени воображения).
Курсор: в момент стрелки ставится в центр, за 4 с воображения едет на
GAIN × нормированную сумму LDA за шаг. Попадание = сторона в конце.
"""

import bci_paths  # noqa: F401  — первым: пути MNE/MOABB внутри проекта

import argparse
import json
import warnings

import mne
import numpy as np
from mne.decoding import CSP
from moabb.datasets import BNCI2014_001
from scipy.signal import butter, sosfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from csp_lda_baseline import FMAX, FMIN

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

APP_DATA = bci_paths.ROOT / "app" / "data"
WIN_S = 2.0                         # сколько секунд мозга видит модель
STEP_S = 0.1                        # как часто решает
TRAIN_STARTS_S = (2.5, 3.0, 3.5, 4.0)   # окна для учёбы, от начала пробы
CUE_S, MI_END_S = 2.0, 6.0
GAIN, CLIP = 0.04, 3.0              # движение курсора за шаг
CODES = {1: "left_hand", 2: "right_hand", 3: "feet", 4: "tongue"}


def day_signal(sessions, key):
    runs = [sessions[key][r].copy() for r in sorted(sessions[key])]
    raw = mne.concatenate_raws(runs)
    events = mne.find_events(raw, shortest_event=1)
    events = events[np.isin(events[:, 2], list(CODES))]
    eeg = raw.copy().pick("eeg")
    return eeg.get_data() * 1e6, eeg.ch_names, float(raw.info["sfreq"]), events


def window_logpower(Z, win):
    """log средней мощности за последние win отсчётов, для каждого конца окна."""
    c = np.cumsum(np.c_[np.zeros((Z.shape[0], 1)), Z ** 2], axis=1)
    return np.log((c[:, win:] - c[:, :-win]) / win)      # (компоненты, N-win+1)


def cursor_trials(score, t_steps, events, fs, positive):
    """Курсор по пробам: в момент стрелки в центр, 4 с едет на GAIN × сумму."""
    negative = [c for c in ("left_hand", "right_hand") if c != positive][0]
    trials, hits, counted = [], 0, 0
    for e, _, code in events:
        t0 = e / fs
        cls = CODES[code]
        m = (t_steps > t0 + CUE_S) & (t_steps <= t0 + MI_END_S)
        x = 0.0
        for v in score[m]:
            x = float(np.clip(x + GAIN * np.clip(v, -CLIP, CLIP), -1, 1))
        said = positive if x > 0 else negative
        tr = {"t": round(t0, 3), "y": cls, "end_x": round(x, 3), "said": said}
        if cls in ("left_hand", "right_hand"):
            tr["hit"] = said == cls
            hits += tr["hit"]
            counted += 1
        trials.append(tr)
    return trials, int(hits), int(counted)


def simulate(s):
    sessions = BNCI2014_001().get_data(subjects=[s])[s]
    k1, k2 = sorted(sessions)
    sos = None

    # --- день 1: учёба ---
    x1, names, fs, ev1 = day_signal(sessions, k1)
    sos = butter(4, [FMIN, FMAX], btype="band", fs=fs, output="sos")
    f1 = sosfilt(sos, x1, axis=1)
    win, step = int(WIN_S * fs), int(round(STEP_S * fs))
    Xtr, ytr = [], []
    for e, _, code in ev1:
        if code not in (1, 2):
            continue
        for st in TRAIN_STARTS_S:
            a = e + int(st * fs)
            Xtr.append(f1[:, a:a + win])
            ytr.append(CODES[code])
    Xtr, ytr = np.array(Xtr), np.array(ytr)
    csp = CSP(n_components=6, log=True, component_order="alternate").fit(Xtr, ytr)
    lda = LinearDiscriminantAnalysis().fit(csp.transform(Xtr), ytr)
    train_scores = lda.decision_function(csp.transform(Xtr))
    norm = float(np.std(train_scores))
    positive = str(lda.classes_[1])          # сумма > 0 -> этот класс

    # --- день 2: поток ---
    x2, _, _, ev2 = day_signal(sessions, k2)
    f2 = sosfilt(sos, x2, axis=1)
    N = f2.shape[1]
    Z = csp.filters_[:6] @ f2
    feats = window_logpower(Z, win)                  # конец окна = win-1 … N-1
    ends = np.arange(win - 1, N, step)               # отсчёт, на котором решаем
    F = feats[:, ends - (win - 1)].T
    score = lda.decision_function(F) / norm

    # «счётчики ритма» над C3 и C4: мощность за 1 с, % от медианы дня
    pw = {}
    for ch in ("C3", "C4"):
        p = np.exp(window_logpower(f2[[names.index(ch)]], int(fs))[0])
        p = p[np.clip(ends - (int(fs) - 1), 0, None)]
        pw[ch] = np.round((p / np.median(p) - 1) * 100, 0).astype(int).tolist()

    # --- пробы и курсор ---
    t_steps = (ends + 1) / fs
    trials, hits, counted = cursor_trials(score, t_steps, ev2, fs, positive)

    print(f"S{s}: поток — {hits} из {counted} проб = {hits / counted:.1%}", flush=True)
    disp = 2  # волны для экрана: 125 Гц
    return {
        "subject": s, "fs_disp": fs / disp, "positive": positive,
        "win_s": WIN_S, "step_s": STEP_S, "gain": GAIN, "clip": CLIP,
        "cue_s": CUE_S, "mi_end_s": MI_END_S,
        "t_first": round(float(t_steps[0]), 3), "duration": round(N / fs, 2),
        "score": np.round(score, 2).tolist(), "pw": pw,
        "eeg": {ch: np.round(f2[names.index(ch), ::disp], 1).tolist()
                for ch in ("C3", "C4")},
        "trials": trials, "hits": int(hits), "counted": int(counted),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    args = ap.parse_args()
    APP_DATA.mkdir(parents=True, exist_ok=True)
    summary_path = APP_DATA / "stream_index.json"
    summary = {}
    if summary_path.exists():
        summary = {e["subject"]: e for e in json.loads(summary_path.read_text("utf-8"))}
    for s in args.subjects:
        out = simulate(s)
        (APP_DATA / f"stream_s{s:02d}.json").write_text(
            json.dumps(out, separators=(",", ":")), encoding="utf-8")
        summary[s] = {"subject": s, "hits": out["hits"], "counted": out["counted"]}
        summary_path.write_text(json.dumps([summary[k] for k in sorted(summary)], indent=1),
                                encoding="utf-8")


if __name__ == "__main__":
    main()
