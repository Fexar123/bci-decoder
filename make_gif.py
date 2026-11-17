"""GIF для README из лога настоящего прогона live_decoder.py (ничего не дорисовывается).

Запуск (из projects/bci, после прогона декодера):
    .venv/Scripts/python make_gif.py --log results/live_s01_ganglion_rest3.csv --start 300 --seconds 30

Кадр = одно решение декодера (0,1 с записи), GIF идёт в реальном времени (10 кадров/с).
Сверху — стрелка, которую человеку показали во время записи, и кубик; снизу — вероятности
[лево, право, покой] за последние 10 с и полоска команд.
"""

import bci_paths  # noqa: F401

import argparse
import csv

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

L, R, REST, BG = "#60a5fa", "#f472b6", "#94a3b8", "#0b0f17"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--fs", type=float, default=200.0)
    ap.add_argument("--start", type=float, default=300.0, help="с какой секунды записи")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", default=str(bci_paths.FIGURES / "live_demo.gif"))
    a = ap.parse_args()
    with open(a.log, encoding="utf-8") as f:
        rows = np.array([[float(v) for v in r] for r in list(csv.reader(f))[1:]])
    t = (rows[:, 0] + 1) / a.fs
    P, cmd, x, cue = rows[:, 1:4], rows[:, 4].astype(int), rows[:, 5], rows[:, 6].astype(int)
    idx = np.where((t >= a.start) & (t < a.start + a.seconds))[0]

    fig = plt.figure(figsize=(7.2, 3.6), facecolor=BG)
    top = fig.add_axes([0.03, 0.46, 0.94, 0.52], facecolor=BG)
    bot = fig.add_axes([0.08, 0.17, 0.89, 0.25], facecolor=BG)

    def frame(j):
        i = idx[j]
        top.clear(); top.set_facecolor(BG); top.set_xlim(-1.25, 1.25); top.set_ylim(-1, 1.25); top.axis("off")
        top.plot([-1.05, 1.05], [-0.35, -0.35], color="#1f2a3b", lw=3)
        top.add_patch(plt.Rectangle((-1.12, -0.6), 0.07, 0.5, color=L, alpha=0.3))
        top.add_patch(plt.Rectangle((1.05, -0.6), 0.07, 0.5, color=R, alpha=0.3))
        c = cmd[i]
        col = L if c < 0 else R if c > 0 else REST
        top.add_patch(plt.Rectangle((x[i] * 1.0 - 0.09, -0.53), 0.18, 0.36, color=col))
        if cue[i]:
            top.text(0, 0.55, {-1: "←", 1: "→", 2: "↓"}[cue[i]], color={-1: L, 1: R, 2: REST}[cue[i]],
                     ha="center", va="center", fontsize=34, fontweight="bold")
        top.text(-1.2, 1.08, {-1: "← ВЛЕВО", 1: "ВПРАВО →", 0: "ПОКОЙ"}[c], color=col, fontsize=15, fontweight="bold")
        top.text(1.2, 1.08, f"{int(t[i] // 60)}:{int(t[i] % 60):02d}", color="#8a97ab", fontsize=11, ha="right")
        top.text(1.2, 0.85, {-1: "подсказка: левая рука", 1: "подсказка: правая рука",
                             2: "подсказка: ноги/язык", 0: "отдых"}[cue[i]], color="#8a97ab", fontsize=9, ha="right")

        bot.clear(); bot.set_facecolor(BG)
        w = (t > t[i] - 10) & (t <= t[i])
        for k, cc in enumerate((L, R, REST)):
            bot.plot(t[w] - t[i], P[w, k], color=cc, lw=1.4)
        for tt, cm in zip(t[w] - t[i], cmd[w]):
            if cm:
                bot.add_patch(plt.Rectangle((tt - 0.05, -0.12), 0.1, 0.07, color=L if cm < 0 else R))
        bot.set_xlim(-10, 0); bot.set_ylim(-0.15, 1.02)
        bot.tick_params(colors="#8a97ab", labelsize=8)
        for s in bot.spines.values():
            s.set_color("#1f2a3b")
        bot.set_xlabel("секунды назад · синий лево, розовый право, серый покой", color="#8a97ab", fontsize=8)

    anim = FuncAnimation(fig, frame, frames=len(idx))
    anim.save(a.out, writer=PillowWriter(fps=10), dpi=80, savefig_kwargs={"facecolor": BG})
    print(f"{a.out}: {len(idx)} кадров", flush=True)


if __name__ == "__main__":
    main()
