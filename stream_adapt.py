"""Живой курсор: старт без калибровки + дообучение по ходу игры.

Запуск (из projects/bci, после stream_sim.py и stream_zero.py):
    .venv/Scripts/python stream_adapt.py

Старт — как в stream_zero.py: 8 чужих мозгов + накопительное выравнивание.
В сеансе со стрелками ответ каждой пробы известен (какую руку просили), поэтому
после конца каждой пробы «лево/право» (6-я секунда) её окна добавляются в обучение
и модель пересобирается. Доля своих проб в CSP и LDA = N/(N+10) — правило из
calibration_weighted.py, заданное до потока. Будущее модель не видит: проба
попадает в обучение только после того, как закончилась.
В app/data/stream_sXX.json дописывается блок "adapt".
Итог -> results/stream_adapt.json
"""

import bci_paths  # noqa: F401  — первым: пути MNE/MOABB внутри проекта

import json
import warnings

import mne
import numpy as np
from moabb.datasets import BNCI2014_001
from scipy.linalg import eigh
from scipy.signal import butter, sosfilt

from calibration import N_COMP, csp_feats, inv_sqrt, lda
from calibration_weighted import share_auto
from csp_lda_baseline import FMAX, FMIN
from stream_sim import (APP_DATA, MI_END_S, STEP_S, TRAIN_STARTS_S, WIN_S, cursor_trials,
                        day_signal)
from stream_zero import aligned, online_R, subject_cache, train, window_covs

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")


def train_w(Cs, ys, Co, yo, share):
    """Как train(), но свои пробы Co весят share, чужие Cs — 1 - share."""
    if share <= 0 or len(set(yo)) < 2:
        return train(Cs, ys)
    cls = [(1 - share) * Cs[ys == k].mean(0) + share * Co[yo == k].mean(0) for k in (0, 1)]
    vals, vecs = eigh(cls[0], cls[0] + cls[1])
    order = np.argsort(vals)
    W = vecs[:, np.r_[order[: N_COMP // 2], order[-(N_COMP // 2):]]]
    Fs, Fo = csp_feats(Cs, W), csp_feats(Co, W)
    rep = max(1, int(round(share / (1 - share) * len(Fs) / len(Fo))))
    clf = lda().fit(np.r_[Fs, np.repeat(Fo, rep, axis=0)], np.r_[ys, np.repeat(yo, rep)])
    return W, clf, float(np.std(clf.decision_function(np.r_[Fs, Fo])))


def batch_align(Cs, R):
    v, U = np.linalg.eigh(R)
    P = U @ (v[:, :, None] ** -0.5 * np.transpose(U, (0, 2, 1)))
    return P @ Cs @ np.transpose(P, (0, 2, 1))


def thirds(trials):
    h = [t["hit"] for t in trials if "hit" in t]
    k = len(h) // 3
    return float(np.mean(h[:k])), float(np.mean(h[-k:]))


def main():
    ds = BNCI2014_001()
    fs = 250.0
    sos = butter(4, [FMIN, FMAX], btype="band", fs=fs, output="sos")
    win, step = int(WIN_S * fs), int(round(STEP_S * fs))
    cache = {s: subject_cache(s, sos) for s in ds.subject_list}
    print("кэш готов", flush=True)

    results = {}
    for t in ds.subject_list:
        others = [s for s in ds.subject_list if s != t]
        Cpool = np.concatenate([aligned(cache[s][d]["C"], cache[s][d]["R"]) for s in others for d in (1, 2)])
        ypool = np.concatenate([cache[s][d]["y"] for s in others for d in (1, 2)])

        sessions = ds.get_data(subjects=[t])[t]
        x2, _, _, ev2 = day_signal(sessions, sorted(sessions)[1])
        f2 = sosfilt(sos, x2, axis=1)
        ends = np.arange(win - 1, f2.shape[1], step)
        t_steps = (ends + 1) / fs
        Cs = window_covs(f2, ends, win)
        R = online_R(Cs, t_steps, "cumul")
        Cal = batch_align(Cs, R)

        model = train(Cpool, ypool)
        score = np.empty(len(ends))
        own_C, own_y, updates, cur = [], [], [], 0
        for e, _, code in ev2:
            if code not in (1, 2):
                continue
            t_upd = e / fs + MI_END_S
            j = int(np.searchsorted(t_steps, t_upd, side="right"))
            W, clf, norm = model
            if j > cur:
                score[cur:j] = clf.decision_function(csp_feats(Cal[cur:j], W)) / norm
            cur = j
            tr_ends = np.array([e + int(st * fs) + win - 1 for st in TRAIN_STARTS_S])
            P = inv_sqrt(R[j - 1])
            own_C.append(P @ window_covs(f2, tr_ends, win) @ P.T)
            own_y += [int(code == 2)] * len(TRAIN_STARTS_S)
            yo = np.array(own_y)
            n = min((yo == 0).sum(), (yo == 1).sum()) // len(TRAIN_STARTS_S)   # проб на руку
            share = share_auto(n)
            model = train_w(Cpool, ypool, np.concatenate(own_C), yo, share)
            updates.append({"t": round(t_upd, 2), "n": int(n), "share": round(share, 3)})
        W, clf, norm = model
        if cur < len(ends):
            score[cur:] = clf.decision_function(csp_feats(Cal[cur:], W)) / norm

        trials, hits, counted = cursor_trials(score, t_steps, ev2, fs, "right_hand")
        first, last = thirds(trials)
        results[t] = {"adapt": hits / counted, "first_third": first, "last_third": last}
        print(f"S{t}: по ходу игры {hits / counted:.0%} · первая треть {first:.0%} → последняя {last:.0%}", flush=True)

        path = APP_DATA / f"stream_s{t:02d}.json"
        d = json.loads(path.read_text("utf-8"))
        assert len(d["score"]) == len(score)
        d["adapt"] = {"score": np.round(score, 2).tolist(), "trials": trials,
                      "hits": hits, "counted": counted, "updates": updates}
        path.write_text(json.dumps(d, separators=(",", ":")), encoding="utf-8")

    zero = json.loads((bci_paths.RESULTS / "stream_zero.json").read_text("utf-8"))["per_subject"]
    for s in results:
        results[s]["own_day1"] = zero[str(s)]["own"]
        results[s]["zero"] = zero[str(s)]["cumul"]
    mean = {k: float(np.mean([results[s][k] for s in results])) for k in results[1]}
    print("\nсреднее: " + " · ".join(f"{k} {v:.1%}" for k, v in mean.items()))
    (bci_paths.RESULTS / "stream_adapt.json").write_text(json.dumps(
        {"per_subject": results, "mean": mean}, ensure_ascii=False, indent=1), encoding="utf-8")

    idx_path = APP_DATA / "stream_index.json"
    idx = json.loads(idx_path.read_text("utf-8"))
    for e in idx:
        e["adapt"] = round(results[e["subject"]]["adapt"], 4)
    idx_path.write_text(json.dumps(idx, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
