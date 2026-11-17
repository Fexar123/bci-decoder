"""Декодер реального времени: LSL -> фильтр -> окно 2 с -> «влево / вправо / стоять» -> браузер.

Запуск (из projects/bci), два окна терминала:
    .venv/Scripts/python live_decoder.py --subject 1 --config ganglion          # потом открыть http://localhost:8766
    .venv/Scripts/python lsl_source.py replay --subject 1 --config ganglion
Проверка, что поток считает то же, что раздельный расчёт (быстро, ×20):
    .venv/Scripts/python live_decoder.py --subject 1 --check --no-server
    .venv/Scripts/python lsl_source.py replay --subject 1 --speed 20

Что внутри:
  * модели и порог учатся на дне 1 этого человека (как в idle_eval.py) и кэшируются
    в results/models/; подход по умолчанию — rest3 (лево / право / покой);
  * поток: куски из LSL -> причинный фильтр 8–30 Гц с сохранением состояния ->
    кольцевой буфер 2 с -> каждые 0,1 с вероятности [лево, право, покой] ->
    сглаживание за 0,5 с -> правило порога (live_common.command);
  * курсор: на шаге едет на GAIN в сторону команды; в момент стрелки новой пробы — в центр;
  * браузер получает состояние через Server-Sent Events (/events), страница — app/live.html;
  * лог решений -> results/live_sXX_<config>.csv; с --check — сверка с раздельным расчётом
    -> results/live_check.json.
"""

import bci_paths  # noqa: F401  — первым: пути MNE/MOABB внутри проекта

import argparse
import csv
import json
import threading
import time
import warnings
from collections import deque
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import joblib
import numpy as np
from pylsl import StreamInlet, resolve_byprop

import live_common as lc

warnings.filterwarnings("ignore")
APP = bci_paths.ROOT / "app"
MODELS = bci_paths.RESULTS / "models"
GAIN = 0.04


# ---------------- модели ----------------
def day_data(subject, cfg_name):
    from moabb.datasets import BNCI2014_001
    sessions = BNCI2014_001().get_data(subjects=[subject])[subject]
    k1, k2 = sorted(sessions)
    return lc.load_day(sessions, k1, lc.CONFIGS[cfg_name]), (sessions, k2)


def train_or_load(subject, cfg_name):
    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / f"live_s{subject:02d}_{cfg_name}.joblib"
    if path.exists():
        return joblib.load(path)
    from scipy.signal import sosfilt
    from idle_eval import oof_probs, pick_threshold
    (x1, names, fs, ev1), _ = day_data(subject, cfg_name)
    X, y, g = lc.training_windows(sosfilt(lc.bandpass(fs), x1, axis=1), ev1, fs)
    oof = oof_probs(X, y, g)
    pack = {"models": lc.fit_models(X, y), "names": names, "fs": fs,
            "thr": {a: pick_threshold(a, oof[a], y) for a in lc.APPROACHES}}
    joblib.dump(pack, path)
    return pack


# ---------------- браузер ----------------
class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.data, self.version = {"status": "жду поток"}, 0

    def set(self, d):
        with self.lock:
            self.data, self.version = d, self.version + 1

    def get(self):
        with self.lock:
            return self.version, self.data


class Handler(SimpleHTTPRequestHandler):
    state = None

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path != "/events":
            return super().do_GET()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        last = -1
        try:
            while True:
                v, d = self.state.get()
                if v != last:
                    self.wfile.write(f"data: {json.dumps(d, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
                    last = v
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


def serve(state, port):
    Handler.state = state
    srv = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(APP)))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"демо: http://localhost:{port}/live.html", flush=True)


# ---------------- поток ----------------
def run(args):
    pack = train_or_load(args.subject, args.config)
    fs, names = pack["fs"], pack["names"]
    thr = pack["thr"][args.approach]
    print(f"S{args.subject} [{args.config}] {args.approach}: порог дня 1 = {thr:.2f}; каналы {names}", flush=True)
    state = State()
    if not args.no_server:
        serve(state, args.port)

    info = resolve_byprop("name", "BCI-EEG", timeout=300)[0]
    if info.channel_count() != len(names) or abs(info.nominal_srate() - fs) > 1e-6:
        raise SystemExit(f"поток {info.channel_count()} кан. {info.nominal_srate()} Гц, а модель {len(names)} кан. {fs} Гц")
    inlet = StreamInlet(info, max_buflen=360, max_chunklen=0)
    mk = resolve_byprop("name", "BCI-Markers", timeout=2)
    mk = StreamInlet(mk[0]) if mk else None
    inlet.open_stream()
    if mk:
        mk.open_stream()

    flt = lc.CausalFilter(lc.bandpass(fs), len(names))
    win, step = int(lc.WIN_S * fs), int(round(lc.STEP_S * fs))
    buf = np.zeros((len(names), 0), np.float64)
    n_total, k_next = 0, win - 1
    hist = deque(maxlen=lc.SMOOTH)
    x, cue_side, events, rows = 0.0, 0, [], []
    t_proc, n_dec, last_data, ended = 0.0, 0, time.time(), False

    while not ended:
        if mk:
            while True:
                s, _ = mk.pull_sample(timeout=0.0)
                if s is None:
                    break
                idx, lab = s[0].split("|")
                events.append((int(idx), lab))
        chunk, _ = inlet.pull_chunk(timeout=0.2, max_samples=8192)
        if not chunk:
            ended = any(lab == "end" and n_total >= int(i) for i, lab in events) or \
                (n_total and time.time() - last_data > args.idle_exit)
            continue
        last_data = time.time()
        c = np.asarray(chunk, np.float64).T
        t0 = time.perf_counter()
        buf = np.concatenate([buf, flt(c)], axis=1)
        n_total += c.shape[1]
        ends = []
        while k_next < n_total:
            ends.append(k_next)
            k_next += step
        if ends:
            off = n_total - buf.shape[1]             # глобальный номер buf[:, 0]
            X = np.stack([buf[:, k - off - win + 1:k - off + 1] for k in ends])
            P = lc.probs(pack["models"], X)[args.approach]
            for k, p in zip(ends, P):
                hist.append(p)
                ps = np.mean(hist, axis=0)
                cmd = int(lc.command(args.approach, ps, thr)[0])
                # стрелка: от 2 до 6 с после начала пробы; в момент стрелки курсор в центр
                cue_side = 0
                for e, lab in events[-3:]:
                    if lab in lc.LR and e + lc.CUE_S * fs <= k < e + lc.MI_END_S * fs:
                        cue_side = -1 if lab == "left_hand" else 1
                        if k - step < e + lc.CUE_S * fs:
                            x = 0.0
                    elif lab in ("feet", "tongue") and e + lc.CUE_S * fs <= k < e + lc.MI_END_S * fs:
                        cue_side = 2
                x = float(np.clip(x + GAIN * cmd, -1, 1))
                rows.append((k, *np.round(ps, 4), cmd, round(x, 3), cue_side))
            n_dec += len(ends)
        buf = buf[:, -win:]
        t_proc += time.perf_counter() - t0
        if rows:
            k, pl, pr, prest, cmd, xx, cs = rows[-1]
            state.set({"status": "поток", "subject": args.subject, "config": args.config,
                       "approach": args.approach, "t": round((k + 1) / fs, 2), "p": [pl, pr, prest],
                       "cmd": cmd, "x": xx, "cue": cs, "thr": thr,
                       "ms_per_decision": round(t_proc / max(1, n_dec) * 1000, 2)})

    dur = n_total / fs
    print(f"конец: {n_total} отсчётов = {dur / 60:.1f} мин, {n_dec} решений, "
          f"{t_proc / max(1, n_dec) * 1000:.2f} мс на решение", flush=True)
    log = bci_paths.RESULTS / f"live_s{args.subject:02d}_{args.config}_{args.approach}.csv"
    with log.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["end_sample", "p_left", "p_right", "p_rest", "cmd", "x", "cue"])
        w.writerows(rows)
    print(f"лог: {log}", flush=True)
    if args.check:
        check(args, pack, rows, n_total, t_proc, n_dec)


def check(args, pack, rows, n_total, t_proc, n_dec):
    from scipy.signal import sosfilt
    from idle_eval import stream_probs
    _, (sessions, k2) = day_data(args.subject, args.config)
    x2, _, fs, _ = lc.load_day(sessions, k2, lc.CONFIGS[args.config])
    _, P = stream_probs(pack["models"], sosfilt(lc.bandpass(fs), x2, axis=1), fs)
    ref = lc.command(args.approach, lc.smooth(P[args.approach]), pack["thr"][args.approach])
    got = np.array([r[4] for r in rows])
    n = min(len(ref), len(got))
    res = {"subject": args.subject, "config": args.config, "approach": args.approach,
           "decisions_stream": len(got), "decisions_batch": len(ref),
           "match_pct": float(np.mean(ref[:n] == got[:n]) * 100),
           "max_prob_diff": float(np.max(np.abs(lc.smooth(P[args.approach])[:n]
                                                 - np.array([r[1:4] for r in rows])[:n]))),
           "ms_per_decision": t_proc / max(1, n_dec) * 1000,
           "minutes": n_total / fs / 60}
    print(f"СВЕРКА: решений поток {len(got)} / раздельно {len(ref)}, совпало {res['match_pct']:.2f}%, "
          f"макс. разница вероятностей {res['max_prob_diff']:.1e}", flush=True)
    path = bci_paths.RESULTS / "live_check.json"
    old = json.loads(path.read_text("utf-8")) if path.exists() else []
    old = [o for o in old if (o["subject"], o["config"], o["approach"]) != (args.subject, args.config, args.approach)]
    path.write_text(json.dumps(old + [res], ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--config", choices=list(lc.CONFIGS), default="ganglion")
    ap.add_argument("--approach", choices=list(lc.APPROACHES), default="rest3")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-server", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--idle-exit", type=float, default=10.0, help="выйти, если данных нет N секунд")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
