"""Наглядно на реальных данных: что такое эпоха и как выглядит ERD.

Запуск (из projects/bci):
    .venv/Scripts/python explain_erd.py --subject 1

Картинка -> figures/erd_sXX.png, три ряда:
  1. запись подряд (канал C3) и куски, которые вырезаются в эпохи;
  2. одна эпоха: 22 канала × 4 с после фильтра 8–30 Гц;
  3. ERD: мощность 8–30 Гц над C3 и C4 по времени пробы, левая vs правая рука.
"""

import bci_paths  # noqa: F401  — первым: пути MNE/MOABB внутри проекта

import argparse
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from moabb.datasets import BNCI2014_001

mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", category=FutureWarning)

FMIN, FMAX = 8.0, 30.0
CUE_S, END_S = 2.0, 6.0           # стрелка на 2-й с пробы, воображение до 6-й
LABELS = {1: "← лев. рука", 2: "→ прав. рука", 3: "↓ ноги", 4: "↑ язык"}
COLORS = {1: "tab:blue", 2: "tab:red", 3: "tab:green", 4: "tab:purple"}


def load_train_raw(subject):
    """Все прогоны первой сессии (день 1) одной непрерывной записью."""
    sessions = BNCI2014_001().get_data(subjects=[subject])[subject]
    first = sorted(sessions)[0]
    runs = [sessions[first][r].copy() for r in sorted(sessions[first])]
    return mne.concatenate_raws(runs)


def erd_curve(epochs, ch, sfreq):
    """Мощность канала во времени, % от фона до стрелки (0.5–1.5 с)."""
    x = epochs.copy().pick([ch]).get_data()[:, 0, :]      # пробы × время
    power = (x ** 2).mean(axis=0)                          # среднее по пробам
    win = int(0.5 * sfreq)
    power = np.convolve(power, np.ones(win) / win, mode="same")
    t = epochs.times
    ref = power[(t >= 0.5) & (t <= 1.5)].mean()
    return t, (power / ref - 1) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1)
    s = ap.parse_args().subject

    raw = load_train_raw(s)
    sfreq = raw.info["sfreq"]
    events = mne.find_events(raw, shortest_event=1)
    events = events[np.isin(events[:, 2], list(LABELS))]
    eeg = raw.copy().pick("eeg")
    filt = eeg.copy().filter(FMIN, FMAX)

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.5, 1.2], hspace=0.45)

    # 1. запись подряд + вырезаемые куски
    ax = fig.add_subplot(gs[0, :])
    t0 = events[2, 0] / sfreq - 1.0
    t1 = t0 + 26.0
    seg = eeg.copy().pick(["C3"]).filter(1.0, 40.0).crop(t0, t1)
    ax.plot(seg.times + t0, seg.get_data()[0] * 1e6, color="0.35", lw=0.6)
    for samp, _, code in events:
        on = samp / sfreq
        if t0 <= on <= t1 - END_S:
            ax.axvspan(on + CUE_S, on + END_S, color=COLORS[code], alpha=0.18)
            ax.axvline(on, color="k", lw=0.6, ls=":")
            ax.text(on + CUE_S + 0.1, ax.get_ylim()[1] * 0.85, LABELS[code],
                    color=COLORS[code], fontsize=10, fontweight="bold")
    ax.set(title="1. Запись идёт подряд (канал C3). Пунктир — начало пробы, "
                 "цветной кусок — то, что вырежем в эпоху (2–6 с пробы)",
           xlabel="время записи, с", ylabel="мкВ", xlim=(t0, t1))

    # 2. одна эпоха: 22 канала
    ax = fig.add_subplot(gs[1, 0])
    samp, _, code = next(e for e in events if e[2] in (1, 2)
                         and t0 <= e[0] / sfreq <= t1)
    on = samp / sfreq
    ep = filt.copy().crop(on + CUE_S, on + END_S)
    data = ep.get_data() * 1e6
    step = np.percentile(np.abs(data), 99) * 1.6
    for i, ch in enumerate(ep.ch_names):
        color = "tab:orange" if ch in ("C3", "C4") else "0.4"
        lw = 1.1 if ch in ("C3", "C4") else 0.5
        ax.plot(ep.times, data[i] - i * step, color=color, lw=lw)
    ax.set_yticks([-i * step for i in range(len(ep.ch_names))])
    ax.set_yticklabels(ep.ch_names, fontsize=7)
    ax.set(title=f"2. Одна эпоха ({LABELS[code]}): {len(ep.ch_names)} канала "
                 f"× {data.shape[1]} отсчётов\nфильтр {FMIN:g}–{FMAX:g} Гц",
           xlabel="с от стрелки")

    # подпись-пояснение рядом с эпохой
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    n_lr = int(np.isin(events[:, 2], [1, 2]).sum())
    ax.text(0.0, 0.95, (
        "Эпоха = вырезанный кусок записи вокруг одной метки.\n\n"
        f"• 1 эпоха = таблица {len(ep.ch_names)} × {data.shape[1]}\n"
        f"   ({len(ep.ch_names)} электрода × 4 с × {sfreq:g} отсчётов/с)\n\n"
        f"• день 1: {n_lr} эпох «левая/правая»\n"
        "   + столько же в день 2 = 288\n\n"
        "• у каждой эпохи есть ответ: какая была стрелка\n"
        "   → на этом модель и учится\n\n"
        "Оранжевые линии — C3 и C4,\nнад моторной корой рук."),
        va="top", fontsize=11, family="DejaVu Sans")

    # 3. ERD над C3 и C4
    # с запасом по краям: сглаживание портит края, их потом обрезаем по xlim
    epochs = mne.Epochs(filt, events, event_id={"left": 1, "right": 2},
                        tmin=-1.0, tmax=8.0, baseline=None, preload=True)
    for col, ch in enumerate(("C3", "C4")):
        ax = fig.add_subplot(gs[2, col])
        for name, code in (("left", 1), ("right", 2)):
            t, erd = erd_curve(epochs[name], ch, sfreq)
            ax.plot(t, erd, color=COLORS[code], lw=2, label=LABELS[code])
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(CUE_S, color="k", lw=0.8, ls="--")
        ax.axvspan(CUE_S, END_S, color="0.9", zorder=0)
        ax.text(CUE_S + 0.05, ax.get_ylim()[0] * 0.9, "стрелка", fontsize=9)
        side = "левая кора" if ch == "C3" else "правая кора"
        ax.set(title=f"3. Мощность {FMIN:g}–{FMAX:g} Гц над {ch} ({side})",
               xlabel="с от начала пробы", xlim=(0, 7),
               ylabel="% от фона до стрелки")
        ax.legend(loc="lower left", fontsize=9)

    fig.suptitle(f"Испытуемый {s}: от записи к эпохам и ERD", fontsize=14)
    path = bci_paths.FIGURES / f"erd_s{s:02d}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print(path.relative_to(bci_paths.ROOT))

    # цифры провала: средняя мощность в 3–5 с пробы
    for ch in ("C3", "C4"):
        for name, code in (("left", 1), ("right", 2)):
            t, erd = erd_curve(epochs[name], ch, sfreq)
            m = erd[(t >= 3) & (t <= 5)].mean()
            print(f"{ch} {LABELS[code]}: {m:+.0f}%")


if __name__ == "__main__":
    main()
