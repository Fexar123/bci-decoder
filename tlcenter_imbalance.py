"""pyRiemann #490: решают ли веса проб (sample_weight) перекос классов в TLCenter?

Задача: https://github.com/pyRiemann/pyRiemann/issues/490
Вопрос сопровождающего (03.09.2026): «В TLCenter.fit() есть sample_weight —
решает ли он проблему несбалансированных классов? Покажите результаты».

Запуск (из projects/bci):
    .venv/Scripts/python tlcenter_imbalance.py --subjects 1 --reps 3   # проба
    .venv/Scripts/python tlcenter_imbalance.py                         # все 9

Протокол — как у автора задачи (BCI IV 2a, 2 класса, T -> E):
  * источник = день 1 человека, цель = его день 2;
  * буфер = 40 проб дня 2 без ответов, доля класса большинства 50/70/80/90/100%;
  * TLCenter учится на источнике + буфере; модель TangentSpace + LogisticRegression
    учится на выровненном источнике; проверка — на ВСЁМ дне 2 (он сбалансирован).
Способы выравнивания цели:
  none    — без адаптации (день 2 центрируется центром дня 1);
  plain   — TLCenter без весов (как сейчас в библиотеке);
  oracle  — веса по НАСТОЯЩИМ ответам буфера (в жизни их нет — это потолок);
  pseudo_post — веса по догадкам модели ПОСЛЕ подстройки без весов;
  pseudo_pre  — веса по догадкам ДО подстройки (модель источника, честный вариант);
  pseudo_iter — pseudo_pre + 3 круга «подстройка с весами -> новые догадки».
Итог -> results/tlcenter_imbalance.json
"""

import bci_paths  # noqa: F401  — первым: пути MNE/MOABB внутри проекта

import argparse
import json
import warnings

import mne
import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.transfer import TLCenter, encode_domains
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

BUFFER = 40
MAJORITY = [0.5, 0.7, 0.8, 0.9, 1.0]
METHODS = ("none", "plain", "oracle", "pseudo_post", "pseudo_pre", "pseudo_iter")
SRC, TGT = "src", "tgt"


# ======================================================================
# ЗАДАНИЕ ДЛЯ СЕВЫ — ключевая идея всего эксперимента (5–10 строк).
#
# На входе: labels — массив классов буфера, например [0, 0, 0, 1, 0, 0, ...]
#           (настоящие ответы для oracle или догадки модели для pseudo).
# На выходе: веса проб (массив той же длины), такие что ОБА класса вместе
#           весят одинаково, сколько бы проб каждого ни было.
#
# Пример: в буфере 30 проб класса 0 и 10 проб класса 1.
#   Сейчас (без весов) класс 0 тянет центр в 3 раза сильнее.
#   Нужно: суммарный вес всех проб класса 0 == суммарный вес всех проб класса 1.
#
# Тонкости:
#   * если в буфере вообще один класс — весам нечего выравнивать,
#     верни одинаковые веса (np.ones);
#   * абсолютный масштаб не важен, TLCenter сам нормирует веса.
#
# Подсказка: np.unique(labels, return_counts=True) даёт классы и сколько их.
# ======================================================================
def balanced_weights(labels):
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:                       # один класс — уравнивать не с чем
        return np.ones(len(labels))
    weight_of = dict(zip(classes, 1.0 / counts))   # каждая проба = 1 / размер её класса
    return np.array([weight_of[c] for c in labels])


# ---------- дальше каркас (готов) ----------
def load(s):
    X, y, meta = LeftRightImagery(fmin=8, fmax=30).get_data(BNCI2014_001(), subjects=[s])
    C = Covariances("oas").fit_transform(X)
    sess = sorted(meta["session"].unique())
    return C, (np.asarray(y).astype(str) == "right_hand").astype(int), (meta["session"] == sess[0]).to_numpy()


def run_once(Cs, ys, Cbuf, ybuf, Ct, yt, method):
    """Учит TLCenter + модель, возвращает точность на всём дне 2."""
    X = np.concatenate([Cs, Cbuf])
    y_dummy = np.r_[ys, np.zeros(len(Cbuf), int)]          # ответы буфера НЕ используются
    dom = np.r_[[SRC] * len(Cs), [TGT] * len(Cbuf)]
    X_, y_enc = encode_domains(X, y_dummy, dom)

    w = np.ones(len(X))
    if method == "oracle":
        w[len(Cs):] = balanced_weights(ybuf)
    elif method.startswith("pseudo"):
        pre = TLCenter(target_domain=TGT).fit(X_, y_enc)
        clf = make_pipeline(TangentSpace(), LogisticRegression(max_iter=1000))
        clf.fit(_center(pre, Cs, SRC), ys)
        if method == "pseudo_post":
            # догадки ПОСЛЕ подстройки без весов — круг: буфер уже «нейтрален»
            guess = clf.predict(pre.transform(Cbuf))
        else:
            # догадки ДО подстройки: буфер центрируется центром источника
            guess = clf.predict(_center(pre, Cbuf, SRC))
        w[len(Cs):] = balanced_weights(guess)
        if method == "pseudo_iter":
            # уточняем: подстройка с весами -> новые догадки -> новые веса (3 круга)
            for _ in range(3):
                tl = TLCenter(target_domain=TGT).fit(X_, y_enc, sample_weight=w)
                clf.fit(_center(tl, Cs, SRC), ys)
                w[len(Cs):] = balanced_weights(clf.predict(tl.transform(Cbuf)))

    center = TLCenter(target_domain=TGT).fit(X_, y_enc, sample_weight=w)
    model = make_pipeline(TangentSpace(), LogisticRegression(max_iter=1000))
    model.fit(_center(center, Cs, SRC), ys)
    if method == "none":
        Ct_c = _center(center, Ct, SRC)        # день 2 центром дня 1 = без адаптации
    else:
        Ct_c = center.transform(Ct)            # центром цели (по буферу)
    return float(model.score(Ct_c, yt))


def _center(tl, C, domain):
    """Выровнять матрицы C центром заданного домена."""
    return tl.centers_[domain].transform(C)


def sample_buffer(yt, frac, rng):
    """Индексы буфера: BUFFER проб дня 2, доля класса большинства frac."""
    major = rng.integers(2)                                # какой класс в большинстве
    n_major = int(round(frac * BUFFER))
    idx_major = rng.choice(np.where(yt == major)[0], n_major, replace=False)
    idx_minor = rng.choice(np.where(yt != major)[0], BUFFER - n_major, replace=False)
    return np.r_[idx_major, idx_minor]


def synthetic(reps, seps=(0.5, 1.0, 1.5, 2.0, 3.0), buffer=60):
    """Искусственные данные как в задаче: make_classification_transfer,
    class_sep=2, class_disp=1, theta=pi/6, 200 матриц, буфер 60, seed 42."""
    from pyriemann.datasets import make_classification_transfer
    from pyriemann.transfer import decode_domains

    global BUFFER
    BUFFER, saved = buffer, BUFFER
    rng = np.random.default_rng(0)
    out = {}
    for sep in seps:
        X, y_enc = make_classification_transfer(n_matrices=200, class_sep=2.0, class_disp=1.0,
                                                domain_sep=sep, theta=np.pi / 6, random_state=42)
        X, y, d = decode_domains(X, y_enc)
        y = (np.asarray(y) == np.unique(y)[1]).astype(int)
        s = d == "source_domain"
        Cs, ys, Ct, yt = X[s], y[s], X[~s], y[~s]
        out[str(sep)] = {}
        for frac in MAJORITY:
            acc = {m: [] for m in METHODS}
            for _ in range(reps):
                b = sample_buffer(yt, frac, rng)
                for m in METHODS:
                    acc[m].append(run_once(Cs, ys, Ct[b], yt[b], Ct, yt, m))
            out[str(sep)][str(frac)] = {m: float(np.mean(v)) for m, v in acc.items()}
        print(f"domain_sep {sep}: " + " | ".join(
            f"{int(f * 100)}%: " + " ".join(f"{m} {out[str(sep)][str(f)][m]:.3f}" for m in METHODS)
            for f in MAJORITY), flush=True)
    BUFFER = saved
    (bci_paths.RESULTS / "tlcenter_imbalance_synthetic.json").write_text(json.dumps(
        {"buffer": buffer, "reps": reps, "results": out}, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--synthetic", action="store_true", help="искусственные данные из задачи")
    a = ap.parse_args()
    if a.synthetic:
        synthetic(a.reps)
        return
    rng = np.random.default_rng(0)
    res = {}
    for s in a.subjects:
        C, y, day1 = load(s)
        Cs, ys, Ct, yt = C[day1], y[day1], C[~day1], y[~day1]
        res[s] = {}
        for frac in MAJORITY:
            acc = {m: [] for m in METHODS}
            for _ in range(a.reps):
                b = sample_buffer(yt, frac, rng)
                for m in METHODS:
                    acc[m].append(run_once(Cs, ys, Ct[b], yt[b], Ct, yt, m))
            res[s][str(frac)] = {m: float(np.mean(v)) for m, v in acc.items()}
        print(f"S{s}: " + " | ".join(
            f"{int(f * 100)}/{int(round((1 - f) * 100))}: " + " ".join(f"{m} {res[s][str(f)][m]:.0%}" for m in METHODS)
            for f in MAJORITY), flush=True)

    # сводка: прирост против «без адаптации», среднее по людям и p (парный t-тест)
    print("\nдоля большинства | " + " | ".join(f"{m}: Δ (p)" for m in METHODS[1:]))
    summary = {}
    for f in map(str, MAJORITY):
        none = np.array([res[s][f]["none"] for s in res])
        row = {}
        for m in METHODS[1:]:
            v = np.array([res[s][f][m] for s in res])
            p = float(ttest_rel(v, none).pvalue) if len(res) > 1 else float("nan")
            row[m] = {"delta": float((v - none).mean()), "p": p}
        summary[f] = row
        print(f"{float(f):>15.0%} | " + " | ".join(f"{row[m]['delta']:+.3f} ({row[m]['p']:.3f})" for m in METHODS[1:]))
    (bci_paths.RESULTS / "tlcenter_imbalance.json").write_text(json.dumps(
        {"buffer": BUFFER, "per_subject": res, "summary": summary}, ensure_ascii=False, indent=1),
        encoding="utf-8")


if __name__ == "__main__":
    main()
