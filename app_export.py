"""Готовит данные для визуализатора app/ — по JSON-файлу на испытуемого.

Запуск (из projects/bci):
    .venv/Scripts/python app_export.py              # все 9
    .venv/Scripts/python app_export.py --subjects 1 3

Модель для картинок — как в соревновании: CSP+LDA учим на дне 1 (T),
проверяем на дне 2 (E), 2 класса (левая / правая рука).
"""

import bci_paths  # noqa: F401  — первым: пути MNE/MOABB внутри проекта

import argparse
import json
import warnings

import mne
import numpy as np
from mne.decoding import CSP
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery, MotorImagery
from scipy.interpolate import RBFInterpolator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from csp_lda_baseline import FMAX, FMIN, evaluate
from explain_erd import erd_curve, load_train_raw

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

APP_DATA = bci_paths.ROOT / "app" / "data"
N_COMP = 6
GRID = 64


def r(a, nd=3):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def sphere_fit(points):
    """Сфера по точкам (МНК): |p-c|^2 = R^2 -> линейная система."""
    A = np.c_[2 * points, np.ones(len(points))]
    b = (points ** 2).sum(1)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    c = sol[:3]
    return c, float(np.sqrt(sol[3] + c @ c))


def project_2d(xyz, center):
    """Азимутальная проекция как у топокарт MNE: макушка в центре, нос вверх."""
    p = xyz - center
    p = p / np.linalg.norm(p, axis=1, keepdims=True)
    theta = np.arccos(np.clip(p[:, 2], -1, 1))
    phi = np.arctan2(p[:, 1], p[:, 0])
    rho = theta / (np.pi / 2)
    return np.c_[rho * np.cos(phi), rho * np.sin(phi)]


def topo_grid(pos2d, values, radius):
    """Гладкая карта головы GRID×GRID; вне круга — None."""
    lin = np.linspace(-radius, radius, GRID)
    gx, gy = np.meshgrid(lin, lin)
    pts = np.c_[gx.ravel(), gy.ravel()]
    z = RBFInterpolator(pos2d, values, kernel="thin_plate_spline",
                        smoothing=1e-3)(pts)
    z[np.hypot(pts[:, 0], pts[:, 1]) > radius] = np.nan
    z = z.reshape(GRID, GRID)
    return [[None if np.isnan(v) else round(float(v), 3) for v in row]
            for row in z]


def head_geometry(info):
    full = mne.channels.make_standard_montage("standard_1005")
    all_pos = np.array(list(full.get_positions()["ch_pos"].values()))
    center, radius = sphere_fit(all_pos)
    ch_pos = epochs_montage_positions(info)
    xyz = np.array([ch_pos[ch] for ch in info.ch_names])
    pos2d = project_2d(xyz, center)
    # фон для 3D: все точки 10-05 на коже, только верхняя полусфера
    upper = all_pos[(all_pos - center)[:, 2] > -0.25 * radius]
    return {
        "center": r(center, 5), "radius": round(radius, 5),
        "ch_xyz": r(xyz, 5), "ch_2d": r(pos2d, 4),
        "grid_radius": round(float(np.abs(pos2d).max() * 1.12), 4),
        "montage_xyz": r(upper, 4),
    }


def epochs_montage_positions(info):
    mont = info.get_montage()
    if mont is None:
        mont = mne.channels.make_standard_montage("standard_1005")
    return mont.get_positions()["ch_pos"]


def describe_component(feats, y, classes, k):
    means = [float(feats[y == c, k].mean()) for c in classes]
    return classes[int(np.argmin(means))]


def mix_formula(weights, names, top=5):
    order = np.argsort(-np.abs(weights))[:top]
    return [{"ch": names[i], "w": round(float(weights[i]), 3)} for i in order]


def export_subject(s):
    print(f"S{s}: эпохи…", flush=True)
    paradigm = LeftRightImagery(fmin=FMIN, fmax=FMAX)
    epochs, y, meta = paradigm.get_data(BNCI2014_001(), subjects=[s],
                                        return_epochs=True)
    y = np.asarray(y).astype(str)
    X = epochs.get_data(copy=False)
    info = epochs.info
    names = info.ch_names
    sfreq = float(info["sfreq"])
    classes = sorted(set(y))

    sessions = sorted(meta["session"].unique())
    train = (meta["session"] == sessions[0]).to_numpy()
    test = ~train

    print(f"S{s}: точность…", flush=True)
    acc2 = evaluate(epochs, y, meta, 2)
    ep4, y4, meta4 = MotorImagery(n_classes=4, fmin=FMIN, fmax=FMAX).get_data(
        BNCI2014_001(), subjects=[s], return_epochs=True)
    acc4 = evaluate(ep4, np.asarray(y4).astype(str), meta4, 4)

    # модель: учим на дне 1
    csp = CSP(n_components=N_COMP, log=True, component_order="alternate")
    csp.fit(X[train], y[train])
    feats = csp.transform(X)
    lda = LinearDiscriminantAnalysis().fit(feats[train], y[train])
    scores = lda.decision_function(feats)
    pred = lda.predict(feats)

    filters = csp.filters_[:N_COMP]
    patterns = csp.patterns_[:N_COMP]
    geo = head_geometry(info)

    comps = []
    for k in range(N_COMP):
        f = filters[k] / np.abs(filters[k]).max()
        p = patterns[k] / np.abs(patterns[k]).max()
        sep = abs(feats[train & (y == classes[0]), k].mean()
                  - feats[train & (y == classes[1]), k].mean()) \
            / feats[train, k].std()
        comps.append({
            "filter": r(f), "pattern": r(p),
            "filter_grid": topo_grid(np.array(geo["ch_2d"]), f, geo["grid_radius"]),
            "pattern_grid": topo_grid(np.array(geo["ch_2d"]), p, geo["grid_radius"]),
            "low_for": describe_component(feats[train], y[train], classes, k),
            "separation": round(float(sep), 3),
            "mix": mix_formula(f, names),
            "lda_w": round(float(lda.coef_[0, k]), 3),
        })

    # 2D-картинка: два самых разделяющих компонента + своя LDA-линия по ним
    best = sorted(range(N_COMP), key=lambda k: -comps[k]["separation"])[:2]
    lda2 = LinearDiscriminantAnalysis().fit(feats[train][:, best], y[train])

    trials = [{
        "f": r(feats[i]), "y": y[i], "day": 1 if train[i] else 2,
        "score": round(float(scores[i]), 3), "pred": pred[i],
    } for i in range(len(y))]

    # одна эпоха целиком (день 2, первая «левая» и первая «правая»): 22 × 250 (125 Гц)
    examples = {}
    for c in classes:
        i = int(np.where(test & (y == c))[0][0])
        examples[c] = {"index": i, "data": r(X[i][:, ::2] * 1e6, 2)}

    # ERD по C3/C4 и кусок сырой записи
    print(f"S{s}: ERD…", flush=True)
    raw = load_train_raw(s)
    events = mne.find_events(raw, shortest_event=1)
    events = events[np.isin(events[:, 2], [1, 2])]
    filt = raw.copy().pick("eeg").filter(FMIN, FMAX)
    erd_ep = mne.Epochs(filt, events, event_id={"left_hand": 1, "right_hand": 2},
                        tmin=-1.0, tmax=8.0, baseline=None, preload=True)
    erd = {}
    for ch in ("C3", "C4"):
        for c in classes:
            t, curve = erd_curve(erd_ep[c], ch, sfreq)
            keep = (t >= 0) & (t <= 7)
            erd[f"{ch}|{c}"] = r(curve[keep][::5], 1)
    erd_t = r(t[keep][::5], 2)

    on = events[3, 0] / sfreq
    seg_raw = raw.copy().pick(["C3"]).filter(1.0, 40.0).crop(on - 0.5, on + 8.0)
    seg_f = filt.copy().pick(["C3"]).crop(on - 0.5, on + 8.0)
    raw_example = {
        "label": "left_hand" if events[3, 2] == 1 else "right_hand",
        "t0": -0.5, "dt": 2 / sfreq,
        "raw": r(seg_raw.get_data()[0][::2] * 1e6, 2),
        "filt": r(seg_f.get_data()[0][::2] * 1e6, 2),
    }

    per_class = {c: int(((y == c) & train).sum()) for c in classes}
    out = {
        "subject": s, "classes": classes, "channels": names, "sfreq": sfreq,
        "n_trials": len(y), "per_class_per_day": per_class,
        "epoch_window_s": [float(epochs.tmin), float(epochs.tmax)],
        "band_hz": [FMIN, FMAX],
        "acc": {"lr": acc2, "four": acc4},
        "geo": geo, "components": comps, "best2": best,
        "lda": {"w": r(lda.coef_[0]), "b": round(float(lda.intercept_[0]), 3),
                "positive": str(lda.classes_[1])},
        "lda2": {"w": r(lda2.coef_[0]), "b": round(float(lda2.intercept_[0]), 3)},
        "trials": trials, "examples": examples,
        "erd": {"t": erd_t, "curves": erd}, "raw_example": raw_example,
    }
    APP_DATA.mkdir(parents=True, exist_ok=True)
    (APP_DATA / f"s{s:02d}.json").write_text(
        json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"S{s}: готово — 2 кл. T->E {acc2['train_to_eval']:.1%}, "
          f"4 кл. T->E {acc4['train_to_eval']:.1%}", flush=True)
    return {"subject": s, "lr": acc2, "four": acc4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    args = ap.parse_args()

    index_path = APP_DATA / "index.json"
    index = {}
    if index_path.exists():
        index = {e["subject"]: e for e in json.loads(index_path.read_text("utf-8"))["subjects"]}
    for s in args.subjects:
        index[s] = export_subject(s)
        index_path.write_text(json.dumps({
            "dataset": "BCI Competition IV 2a (BNCI2014_001)",
            "session_minutes": 38.7, "runs_per_day": 6, "trials_per_run": 48,
            "subjects": [index[k] for k in sorted(index)],
        }, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
