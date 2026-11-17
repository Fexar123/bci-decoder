"""Общее ядро этапа 4: конфиги каналов, данные с зонами покоя, модели, правило решения.

Используют idle_eval.py (раздельный расчёт по всей записи), live_decoder.py (поток LSL)
и lsl_source.py (симулятор и плата). Всё, что решает «двигать или нет», живёт здесь,
чтобы поток и раздельный расчёт считали одно и то же.

Разметка пробы BCI IV 2a (от метки начала пробы e, секунды):
    0–2 крест (человек ждёт), 2 стрелка, 2–6 воображение, потом перерыв до следующей пробы.
Покой: окно целиком после конца воображения прошлой пробы и до стрелки следующей,
то есть конец окна в [e_prev + 6 + WIN_S, e_next + 2]. Между ранами (пауза > 12 с) не берём.
"""

import bci_paths  # noqa: F401  — первым: пути MNE/MOABB внутри проекта

import numpy as np
from math import gcd
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.signal import butter, resample_poly, sosfilt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

FMIN, FMAX = 8.0, 30.0
WIN_S, STEP_S = 2.0, 0.1            # окно, которое видит модель; как часто решает
SMOOTH = 5                          # усреднение вероятностей за 5 шагов = 0,5 с
CUE_S, MI_END_S = 2.0, 6.0
MAX_GAP_S = 12.0                    # дольше — это пауза между ранами
REST_TARGET = 0.10                  # порог выбирается так: на дне 1 покой двигает ≤ 10 % окон
CODES = {1: "left_hand", 2: "right_hand", 3: "feet", 4: "tongue"}
LR = ("left_hand", "right_hand")

CONFIGS = {
    # все 22 канала записи, 250 Гц
    "full": {"channels": None, "fs": 250},
    # как OpenBCI Ganglion: 4 канала, 200 Гц (C3, Cz, C4 + CPz — см. electrodes.py, 69 % на 2 кл.)
    "ganglion": {"channels": ["C3", "Cz", "C4", "CPz"], "fs": 200},
}
APPROACHES = {
    "thr2": "2 класса (лево/право) + порог уверенности, данных покоя нет",
    "rest3": "3 класса: лево / право / покой",
    "two_stage": "детектор «думает / покой», потом лево/право",
}


def load_day(sessions, key, cfg):
    """Сигнал дня (мкВ) в нужных каналах и частоте + метки начала проб в отсчётах этой частоты."""
    from stream_sim import day_signal
    x, names, fs, events = day_signal(sessions, key)
    if cfg["channels"]:
        x = x[[names.index(c) for c in cfg["channels"]]]
        names = list(cfg["channels"])
    fs_out = cfg["fs"]
    if fs_out != fs:
        g = gcd(int(fs_out), int(fs))
        x = resample_poly(x, int(fs_out) // g, int(fs) // g, axis=1)
        events = events.copy()
        events[:, 0] = np.round(events[:, 0] * fs_out / fs).astype(int)
    return x, names, float(fs_out), events


def bandpass(fs):
    return butter(4, [FMIN, FMAX], btype="band", fs=fs, output="sos")


def rest_zones(events, fs):
    """Отрезки (в секундах), где конец 2-секундного окна видит только покой."""
    t = events[:, 0] / fs
    z = []
    for a, b in zip(t[:-1], t[1:]):
        if b - a > MAX_GAP_S:
            continue
        lo, hi = a + MI_END_S + WIN_S, b + CUE_S
        if hi > lo:
            z.append((lo, hi))
    return z


def training_windows(f, events, fs):
    """Окна дня 1: воображение (лево/право) и покой. Возвращает X, y, группа = номер пробы."""
    win = int(WIN_S * fs)
    X, y, g = [], [], []
    for i, (e, _, code) in enumerate(events):
        if CODES[code] in LR:
            for st in (2.5, 3.0, 3.5, 4.0):                 # как в stream_sim.py
                a = e + int(st * fs)
                X.append(f[:, a:a + win]); y.append(CODES[code]); g.append(i)
    for k, (lo, hi) in enumerate(rest_zones(events, fs)):
        for end in np.arange(lo, hi + 1e-9, 0.5):
            b = int(round(end * fs))
            X.append(f[:, b - win:b]); y.append("rest"); g.append(10_000 + k)
    return np.array(X), np.array(y), np.array(g)


def make_model():
    return make_pipeline(Covariances(estimator="oas"), TangentSpace(metric="riemann"),
                         LogisticRegression(max_iter=2000))


def fit_models(X, y):
    """Три подхода. Каждый возвращает модели; probs() приводит их к (p_лево, p_право, p_покой)."""
    lr = np.isin(y, LR)
    m2 = make_model().fit(X[lr], y[lr])
    m3 = make_model().fit(X, y)
    det = make_model().fit(X, np.where(lr, "mi", "rest"))
    return {"lr": m2, "rest3": m3, "det": det}


def probs(models, X):
    """Вероятности для всех подходов: dict подход -> (n, 3) [лево, право, покой]."""
    p2 = models["lr"].predict_proba(X)
    c2 = list(models["lr"].classes_)
    pl, pr = p2[:, c2.index("left_hand")], p2[:, c2.index("right_hand")]
    p3 = models["rest3"].predict_proba(X)
    c3 = list(models["rest3"].classes_)
    pd = models["det"].predict_proba(X)[:, list(models["det"].classes_).index("mi")]
    return {
        "thr2": np.c_[pl, pr, np.zeros_like(pl)],
        "rest3": np.c_[p3[:, c3.index("left_hand")], p3[:, c3.index("right_hand")], p3[:, c3.index("rest")]],
        "two_stage": np.c_[pd * pl, pd * pr, 1 - pd],
    }


def command(approach, P, thr):
    """Решение по (сглаженным) вероятностям [лево, право, покой]: -1 влево, +1 вправо, 0 стоять.

      thr2:      max(лево, право) ≥ thr;
      rest3:     победитель не «покой» и его вероятность ≥ thr;
      two_stage: P(думает) = лево + право ≥ thr, сторона — что больше.
    """
    P = np.atleast_2d(P)
    side = np.where(P[:, 1] > P[:, 0], 1, -1)
    best = np.maximum(P[:, 0], P[:, 1])
    if approach == "thr2":
        go = best >= thr
    elif approach == "rest3":
        go = (best > P[:, 2]) & (best >= thr)
    else:
        go = P[:, 0] + P[:, 1] >= thr
    return np.where(go, side, 0)


def dwell(cmd, d):
    """Выдержка: едем, только если последние d решений одинаковые и не «стоять»."""
    if d <= 1:
        return cmd
    out = cmd.copy()
    for j in range(1, d):
        prev = np.r_[np.zeros(j, int), cmd[:-j]]
        out = np.where(prev == cmd, out, 0)
    return out


def smooth(P, k=SMOOTH):
    """Скользящее среднее вероятностей по последним k шагам (только прошлое)."""
    c = np.cumsum(np.r_[np.zeros((1, P.shape[1])), P], axis=0)
    n = np.arange(1, len(P) + 1)
    lo = np.maximum(0, n - k)
    return (c[n] - c[lo]) / (n - lo)[:, None]


class CausalFilter:
    """sosfilt с сохранением состояния между кусками — как в потоке."""

    def __init__(self, sos, n_ch):
        self.sos = sos
        self.zi = np.zeros((sos.shape[0], n_ch, 2))

    def __call__(self, chunk):
        y, self.zi = sosfilt(self.sos, chunk, axis=1, zi=self.zi)
        return y
