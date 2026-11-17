"""Этап 4: класс «покой» — сколько раз курсор дёргается, когда человек ни о чём не думает.

Запуск (из projects/bci):
    .venv/Scripts/python idle_eval.py                    # 9 человек, оба конфига
    .venv/Scripts/python idle_eval.py --subjects 1 --configs ganglion

Протокол (честно, как вживую):
  * день 1 — учёба: окна воображения лево/право + окна покоя (live_common.training_windows);
  * порог каждого подхода выбирается на дне 1 по окнам, которые модель не видела
    (5 частей по пробам): наименьший порог, при котором покой двигает ≤ 10 % окон;
  * день 2 — проверка: фильтр причинный, окно 2 с, решение каждые 0,1 с,
    вероятности сглажены за 0,5 с, всё как в live_decoder.py.
Метрики дня 2:
  * покой: % времени, когда курсор едет; ложные старты в минуту (переход «стоит» → «едет»);
  * воображение лево/право (конец окна 4–6 с от начала пробы): % времени, когда едет
    в правильную сторону; точность стороны, когда едет; % проб, где курсор в конце
    на правильной стороне;
  * ноги/язык (другое воображение): % времени, когда едет.
Итог -> results/idle_eval.json, figures/idle_curve.png
"""

import bci_paths  # noqa: F401  — первым: пути MNE/MOABB внутри проекта

import argparse
import json
import warnings

import matplotlib
import mne
import numpy as np
from moabb.datasets import BNCI2014_001
from scipy.signal import sosfilt
from sklearn.model_selection import GroupKFold

import live_common as lc

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

GRID = np.round(np.arange(0.30, 0.995, 0.01), 2)
DWELLS = (3, 5)                     # выдержка 0,3 и 0,5 с


def oof_probs(X, y, g):
    out = {a: np.zeros((len(y), 3)) for a in lc.APPROACHES}
    for tr, te in GroupKFold(5).split(X, y, g):
        P = lc.probs(lc.fit_models(X[tr], y[tr]), X[te])
        for a in out:
            out[a][te] = P[a]
    return out


def pick_threshold(a, P, y):
    rest = y == "rest"
    for thr in GRID:
        if np.mean(lc.command(a, P[rest], thr) != 0) <= lc.REST_TARGET:
            return float(thr)
    return float(GRID[-1])


def stream_probs(models, f, fs, batch=1500):
    win, step = int(lc.WIN_S * fs), int(round(lc.STEP_S * fs))
    ends = np.arange(win - 1, f.shape[1], step)
    P = {a: [] for a in lc.APPROACHES}
    for i in range(0, len(ends), batch):
        e = ends[i:i + batch]
        X = np.stack([f[:, k - win + 1:k + 1] for k in e])
        for a, v in lc.probs(models, X).items():
            P[a].append(v)
    return (ends + 1) / fs, {a: np.vstack(v) for a, v in P.items()}


def masks(t, events, fs):
    idle = np.zeros(len(t), bool)
    for lo, hi in lc.rest_zones(events, fs):
        idle |= (t >= lo) & (t <= hi)
    mi, truth, other = np.zeros(len(t), bool), np.zeros(len(t), int), np.zeros(len(t), bool)
    trials = []
    for e, _, code in events:
        t0 = e / fs
        m = (t >= t0 + lc.CUE_S + lc.WIN_S) & (t <= t0 + lc.MI_END_S)
        if lc.CODES[code] in lc.LR:
            mi |= m
            truth[m] = -1 if lc.CODES[code] == "left_hand" else 1
            trials.append((t0, truth[m][0] if m.any() else 0))
        else:
            other |= m
    return idle, mi, truth, other, trials


def metrics(cmd, t, idle, mi, truth, other, trials):
    starts = idle[1:] & (cmd[1:] != 0) & (cmd[:-1] == 0)
    idle_min = idle.sum() * lc.STEP_S / 60
    moving = mi & (cmd != 0)
    # курсор пробы: от стрелки до конца воображения едет на шаг в сторону команды
    hits = 0
    for t0, side in trials:
        m = (t > t0 + lc.CUE_S) & (t <= t0 + lc.MI_END_S)
        hits += np.sign(cmd[m].sum()) == side
    return {
        "idle_move_pct": float(np.mean(cmd[idle] != 0) * 100),
        "false_starts_per_min": float(starts.sum() / idle_min),
        "mi_correct_move_pct": float(np.mean(cmd[mi] == truth[mi]) * 100),
        "mi_direction_acc_pct": float(np.mean(cmd[moving] == truth[moving]) * 100) if moving.any() else None,
        "trial_hit_pct": float(hits / len(trials) * 100),
        "other_mi_move_pct": float(np.mean(cmd[other] != 0) * 100),
    }


def run(s, cfg_name):
    cfg = lc.CONFIGS[cfg_name]
    sessions = BNCI2014_001().get_data(subjects=[s])[s]
    k1, k2 = sorted(sessions)
    x1, _, fs, ev1 = lc.load_day(sessions, k1, cfg)
    sos = lc.bandpass(fs)
    X, y, g = lc.training_windows(sosfilt(sos, x1, axis=1), ev1, fs)
    oof = oof_probs(X, y, g)
    thr = {a: pick_threshold(a, oof[a], y) for a in lc.APPROACHES}
    models = lc.fit_models(X, y)

    x2, _, _, ev2 = lc.load_day(sessions, k2, cfg)
    t, P = stream_probs(models, sosfilt(sos, x2, axis=1), fs)
    idle, mi, truth, other, trials = masks(t, ev2, fs)
    res, curves = {}, {}
    for a in lc.APPROACHES:
        Ps = lc.smooth(P[a])
        c0 = lc.command(a, Ps, thr[a])
        res[a] = {"threshold": thr[a], **metrics(c0, t, idle, mi, truth, other, trials)}
        # выдержка — проверка чувствительности на дне 2, не подбирается
        res[a]["dwell"] = {d: metrics(lc.dwell(c0, d), t, idle, mi, truth, other, trials) for d in DWELLS}
        cv = []
        for th in GRID:
            c = lc.command(a, Ps, th)
            cv.append((float(np.mean(c[idle] != 0) * 100), float(np.mean(c[mi] == truth[mi]) * 100)))
        curves[a] = cv
    msg = " · ".join(f"{a}: покой едет {r['idle_move_pct']:.0f}%, {r['false_starts_per_min']:.1f}/мин, "
                     f"верно едет {r['mi_correct_move_pct']:.0f}%, пробы {r['trial_hit_pct']:.0f}%"
                     for a, r in res.items())
    print(f"S{s} [{cfg_name}] {msg}", flush=True)
    return res, curves, {"rest_minutes": float(idle.sum() * lc.STEP_S / 60),
                         "n_rest_train": int((y == "rest").sum()), "n_mi_train": int((y != "rest").sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    ap.add_argument("--configs", nargs="*", default=list(lc.CONFIGS))
    args = ap.parse_args()
    out = {"grid": GRID.tolist(), "rest_target": lc.REST_TARGET, "approaches": lc.APPROACHES}
    for cfg in args.configs:
        per, curves = {}, {a: [] for a in lc.APPROACHES}
        for s in args.subjects:
            r, c, info = run(s, cfg)
            per[s] = {"approaches": r, **info}
            for a in c:
                curves[a].append(c[a])
        def avg(get, keys):
            return {k: float(np.mean([get(s)[k] for s in per if get(s)[k] is not None])) for k in keys}
        keys = [k for k in per[args.subjects[0]]["approaches"]["rest3"] if k != "dwell"]
        mean = {}
        for a in lc.APPROACHES:
            mean[a] = avg(lambda s: per[s]["approaches"][a], keys)
            mean[a]["dwell"] = {d: avg(lambda s: per[s]["approaches"][a]["dwell"][d], keys[1:]) for d in DWELLS}
        out[cfg] = {"per_subject": per, "mean": mean,
                    "curve_mean": {a: np.mean(curves[a], axis=0).tolist() for a in curves}}
        print(f"\nСРЕДНЕЕ [{cfg}], {len(per)} чел.:")
        for a, m in mean.items():
            print(f"  {a:<9} порог {m['threshold']:.2f} · покой: едет {m['idle_move_pct']:.1f}% времени, "
                  f"{m['false_starts_per_min']:.2f} ложных стартов/мин · воображение: верно едет "
                  f"{m['mi_correct_move_pct']:.1f}%, сторона {m['mi_direction_acc_pct']:.1f}%, "
                  f"пробы {m['trial_hit_pct']:.1f}% · ноги/язык едет {m['other_mi_move_pct']:.1f}%", flush=True)
            for d, md in m["dwell"].items():
                print(f"      выдержка {d * lc.STEP_S:.1f} с: покой едет {md['idle_move_pct']:.1f}%, "
                      f"{md['false_starts_per_min']:.2f}/мин · верно едет {md['mi_correct_move_pct']:.1f}%, "
                      f"пробы {md['trial_hit_pct']:.1f}%", flush=True)
    path = bci_paths.RESULTS / "idle_eval.json"
    old = json.loads(path.read_text("utf-8")) if path.exists() else {}
    old.update(out)
    path.write_text(json.dumps(old, ensure_ascii=False, indent=1), encoding="utf-8")

    fig, axs = plt.subplots(1, len(args.configs), figsize=(6 * len(args.configs), 4.5), squeeze=False)
    for ax, cfg in zip(axs[0], args.configs):
        for a, c in out[cfg]["curve_mean"].items():
            c = np.array(c)
            ax.plot(c[:, 0], c[:, 1], label=a)
            m = out[cfg]["mean"][a]
            ax.plot(m["idle_move_pct"], m["mi_correct_move_pct"], "o", color=ax.lines[-1].get_color())
        ax.set_xlabel("покой: % времени, когда курсор едет (ложно)")
        ax.set_ylabel("воображение: % времени, когда едет верно")
        ax.set_title(f"{cfg}: кривая по порогу, точка — порог дня 1")
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(bci_paths.FIGURES / "idle_curve.png", dpi=110)


if __name__ == "__main__":
    main()
