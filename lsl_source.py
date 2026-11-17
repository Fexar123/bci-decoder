"""Источник ЭЭГ в LSL: запись BCI IV 2a «как прямой эфир» или живая плата через BrainFlow.

Запуск (из projects/bci):
    .venv/Scripts/python lsl_source.py replay --subject 1 --config ganglion            # реальное время
    .venv/Scripts/python lsl_source.py replay --subject 1 --config ganglion --speed 20 # в 20 раз быстрее
    .venv/Scripts/python lsl_source.py brainflow --board synthetic                     # проверка без платы
    .venv/Scripts/python lsl_source.py brainflow --board ganglion --serial-port COM3   # OpenBCI Ganglion

LSL (Lab Streaming Layer) — стандартная «шина» лабораторий: одна программа публикует поток
отсчётов, другие подписываются по имени/типу, хоть на этом же компьютере, хоть по сети.
Публикуются два потока:
  * BCI-EEG (тип EEG): каналы в мкВ, float32, частота конфига;
  * BCI-Markers (тип Markers): строки «номер_отсчёта|событие» — начало пробы и её класс
    (только у записи; у платы меток нет). Номер отсчёта, а не время, — чтобы сверка
    работала при любой скорости проигрывания.
Запись начинает идти, только когда подписался приёмник (иначе первые отсчёты потеряются
и фильтр приёмника начнёт с другого места, чем раздельный расчёт).

Плата: BrainFlow отдаёт отсчёты в мкВ. Ganglion — 4 канала, 200 Гц; какие электроды
куда воткнуты, задаёт --channels (по умолчанию C3 Cz C4 CPz, как конфиг ganglion).
Модель, обученная на чужих людях из BCI IV 2a, на своём мозге без калибровки работать
не обязана — плата здесь проверяет трубу, а калибровка своего мозга — этап 5.
"""

import bci_paths  # noqa: F401  — первым: пути MNE/MOABB внутри проекта

import argparse
import time
import warnings

import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock

import live_common as lc

warnings.filterwarnings("ignore")
CHUNK_S = 0.04                      # кусок 40 мс — как у типичной платы


def make_outlets(names, fs, source_id, markers=True):
    info = StreamInfo("BCI-EEG", "EEG", len(names), fs, "float32", source_id)
    chans = info.desc().append_child("channels")
    for n in names:
        ch = chans.append_child("channel")
        ch.append_child_value("label", n)
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", "EEG")
    eeg = StreamOutlet(info, chunk_size=0, max_buffered=360)
    mk = StreamOutlet(StreamInfo("BCI-Markers", "Markers", 1, 0, "string", source_id + "-m")) if markers else None
    return eeg, mk


def wait_consumer(outlet, what):
    print(f"{what}: жду приёмник (live_decoder.py)…", flush=True)
    while not outlet.wait_for_consumers(1.0):
        pass
    time.sleep(0.5)                 # дать приёмнику подписаться и на метки


def replay(args):
    from moabb.datasets import BNCI2014_001
    cfg = lc.CONFIGS[args.config]
    sessions = BNCI2014_001().get_data(subjects=[args.subject])[args.subject]
    day2 = sorted(sessions)[1]
    x, names, fs, events = lc.load_day(sessions, day2, cfg)
    x = x.astype(np.float32)
    eeg, mk = make_outlets(names, fs, f"bci2a-s{args.subject}-{args.config}")
    wait_consumer(eeg, f"S{args.subject} день 2, {len(names)} каналов, {fs:.0f} Гц, ×{args.speed}")
    step = int(round(CHUNK_S * fs))
    ev = list(events)
    t_start, n = local_clock(), x.shape[1]
    print(f"иду: {n / fs / 60:.1f} мин записи", flush=True)
    for a in range(0, n, step):
        b = min(n, a + step)
        while ev and ev[0][0] < b:                       # метки, попавшие в этот кусок
            e, _, code = ev.pop(0)
            mk.push_sample([f"{int(e)}|{lc.CODES[code]}"])
        eeg.push_chunk(x[:, a:b].T.copy())
        lag = t_start + b / fs / args.speed - local_clock()
        if lag > 0:
            time.sleep(lag)
    mk.push_sample([f"{n}|end"])
    print("запись кончилась", flush=True)
    time.sleep(2)


def brainflow(args):
    from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
    board_id = {"synthetic": BoardIds.SYNTHETIC_BOARD, "ganglion": BoardIds.GANGLION_BOARD}[args.board].value
    params = BrainFlowInputParams()
    if args.serial_port:
        params.serial_port = args.serial_port
    fs = BoardShim.get_sampling_rate(board_id)
    rows = BoardShim.get_eeg_channels(board_id)[:len(args.channels)]
    eeg, _ = make_outlets(args.channels, fs, f"brainflow-{args.board}", markers=False)
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    print(f"{args.board}: {len(rows)} каналов {args.channels}, {fs} Гц -> LSL BCI-EEG. Ctrl+C — стоп", flush=True)
    try:
        while True:
            data = board.get_board_data()                # всё, что накопилось с прошлого раза
            if data.shape[1]:
                eeg.push_chunk(data[rows].T.astype(np.float32).copy())
            time.sleep(CHUNK_S)
    except KeyboardInterrupt:
        pass
    finally:
        board.stop_stream()
        board.release_session()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--subject", type=int, default=1)
    r.add_argument("--config", choices=list(lc.CONFIGS), default="ganglion")
    r.add_argument("--speed", type=float, default=1.0)
    b = sub.add_parser("brainflow")
    b.add_argument("--board", choices=["synthetic", "ganglion"], default="synthetic")
    b.add_argument("--serial-port", default="")
    b.add_argument("--channels", nargs="+", default=lc.CONFIGS["ganglion"]["channels"])
    args = ap.parse_args()
    replay(args) if args.mode == "replay" else brainflow(args)


if __name__ == "__main__":
    main()
