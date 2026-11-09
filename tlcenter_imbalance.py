"""pyRiemann #490: решают ли веса проб (sample_weight) перекос классов в TLCenter?

Задача: https://github.com/pyRiemann/pyRiemann/issues/490
Вопрос сопровождающего (03.09.2026): «В TLCenter.fit() есть sample_weight —
решает ли он проблему несбалансированных классов? Покажите результаты».

Запуск (из projects/bci):
    .venv/Scripts/python tlcenter_imbalance.py --subjects 1 --reps 3          # проба
    .venv/Scripts/python tlcenter_imbalance.py                                # BCI IV 2a, 9 человек
    .venv/Scripts/python tlcenter_imbalance.py --dataset dreyer --reps 10     # Dreyer2023, 87 человек
    .venv/Scripts/python tlcenter_imbalance.py --synthetic                    # данные из задачи

Протокол — как у автора задачи (2 класса, источник -> цель одного человека):
  * BCI IV 2a: источник = день 1, цель = день 2;
    Dreyer2023: источник = 2 калибровочных прогона, цель = 4 онлайн-прогона;
  * буфер = 40 проб цели без ответов, доля класса большинства 50/70/80/90/100%;
  * TLCenter учится на источнике + буфере; модель TangentSpace + LogisticRegression
    учится на выровненном источнике; проверка — на ВСЕЙ цели (она сбалансирована).
Способы выравнивания цели (веса проб буфера):
  none        — без адаптации (цель центрируется центром источника);
  plain       — TLCenter без весов (как сейчас в библиотеке);
  oracle      — веса по НАСТОЯЩИМ ответам буфера (в жизни их нет — это потолок);
  pseudo_post — по догадкам модели ПОСЛЕ подстройки без весов;
  pseudo_pre  — по догадкам ДО подстройки (буфер центрирован центром источника).
Рецепты «когда доверять догадкам» — правила заданы ДО прогона, не подбирались:
  pseudo_soft — мягкие веса по вероятностям модели ДО подстройки: при неуверенной
                модели вероятности ~0.5 и веса сами становятся почти равными;
  pseudo_gate — веса pseudo_pre, только если средняя уверенность модели >= 0.75,
                иначе без весов;
  pseudo_dist — веса pseudo_pre, только если сдвиг цели (риманово расстояние центра
                буфера до центра источника) меньше обычного разброса проб источника,
                иначе без весов.
Итог -> results/tlcenter_imbalance[_dreyer|_synthetic].json
"""

import bci_paths  # noqa: F401  — первым: пути MNE/MOABB внутри проекта

import argparse
import json
import warnings

import mne
import numpy as np
from moabb.paradigms import LeftRightImagery
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.transfer import TLCenter, encode_domains
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.mean import mean_riemann
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

BUFFER = 40
MAJORITY = [0.5, 0.7, 0.8, 0.9, 1.0]
METHODS = ("none", "plain", "oracle", "pseudo_post", "pseudo_pre",
           "pseudo_soft", "pseudo_gate", "pseudo_dist")
GATE_CONF = 0.75
SRC, TGT = "src", "tgt"


def balanced_weights(labels):
    """Веса проб: каждый класс в сумме весит одинаково (проба = 1 / размер её класса)."""
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:                       # один класс — уравнивать не с чем
        return np.ones(len(labels))
    weight_of = dict(zip(classes, 1.0 / counts))
    return np.array([weight_of[c] for c in labels])


def soft_balanced_weights(proba):
    """То же по вероятностям: класс k весит sum_i p_ik / sum_j p_jk."""
    return (proba / proba.sum(axis=0, keepdims=True)).sum(axis=1)


# ---------- данные ----------
def load_bnci(s):
    from moabb.datasets import BNCI2014_001
    X, y, meta = LeftRightImagery(fmin=8, fmax=30).get_data(BNCI2014_001(), subjects=[s])
    sess = sorted(meta["session"].unique())
    return X, y, (meta["session"] == sess[0]).to_numpy()


def load_dreyer(s):
    from moabb.datasets import Dreyer2023
    X, y, meta = LeftRightImagery(fmin=8, fmax=30).get_data(Dreyer2023(), subjects=[s])
    src = ~meta["run"].astype(str).str.contains("online").to_numpy()   # калибровка = источник
    return X, y, src


def prepare(X, y, src):
    C = Covariances("oas").fit_transform(X)
    y = (np.asarray(y).astype(str) == "right_hand").astype(int)
    return C[src], y[src], C[~src], y[~src]


# ---------- один буфер — все способы сразу ----------
def evaluate_all(Cs, ys, Cbuf, ybuf, Ct, yt):
    X = np.concatenate([Cs, Cbuf])
    y_dummy = np.r_[ys, np.zeros(len(Cbuf), int)]          # ответы буфера НЕ используются
    dom = np.r_[[SRC] * len(Cs), [TGT] * len(Cbuf)]
    X_, y_enc = encode_domains(X, y_dummy, dom)

    plain = TLCenter(target_domain=TGT).fit(X_, y_enc)
    # центр источника не зависит от весов цели -> модель одна на все способы
    Cs_c = plain.centers_[SRC].transform(Cs)
    model = make_pipeline(TangentSpace(), LogisticRegression(max_iter=1000)).fit(Cs_c, ys)

    acc = {"none": model.score(plain.centers_[SRC].transform(Ct), yt),
           "plain": model.score(plain.transform(Ct), yt)}

    Cbuf_src = plain.centers_[SRC].transform(Cbuf)          # буфер «вчерашним нулём»
    proba = model.predict_proba(Cbuf_src)
    guess_pre = proba.argmax(axis=1)
    guess_post = model.predict(plain.transform(Cbuf))
    w_pre = balanced_weights(guess_pre)

    conf = proba.max(axis=1).mean()
    shift = distance_riemann(mean_riemann(Cbuf_src), np.eye(Cs.shape[1]))
    spread = np.median([distance_riemann(c, np.eye(Cs.shape[1])) for c in Cs_c])

    w_buf = {
        "oracle": balanced_weights(ybuf),
        "pseudo_post": balanced_weights(guess_post),
        "pseudo_pre": w_pre,
        "pseudo_soft": soft_balanced_weights(proba),
        "pseudo_gate": w_pre if conf >= GATE_CONF else np.ones(len(Cbuf)),
        "pseudo_dist": w_pre if shift < spread else np.ones(len(Cbuf)),
    }
    for m, wb in w_buf.items():
        w = np.r_[np.ones(len(Cs)), wb]
        tl = TLCenter(target_domain=TGT).fit(X_, y_enc, sample_weight=w)
        acc[m] = model.score(tl.transform(Ct), yt)
    return {m: float(acc[m]) for m in METHODS}, {"conf": float(conf), "shift_ratio": float(shift / spread)}


def sample_buffer(yt, frac, rng):
    """Индексы буфера: BUFFER проб цели, доля класса большинства frac."""
    major = rng.integers(2)
    n_major = int(round(frac * BUFFER))
    idx_major = rng.choice(np.where(yt == major)[0], n_major, replace=False)
    idx_minor = rng.choice(np.where(yt != major)[0], BUFFER - n_major, replace=False)
    return np.r_[idx_major, idx_minor]


def run_block(Cs, ys, Ct, yt, reps, rng):
    out, diag = {}, []
    for frac in MAJORITY:
        acc = {m: [] for m in METHODS}
        for _ in range(reps):
            b = sample_buffer(yt, frac, rng)
            a, d = evaluate_all(Cs, ys, Ct[b], yt[b], Ct, yt)
            for m in METHODS:
                acc[m].append(a[m])
            diag.append(d)
        out[str(frac)] = {m: float(np.mean(v)) for m, v in acc.items()}
    return out, {"conf": float(np.mean([d["conf"] for d in diag])),
                 "shift_ratio": float(np.mean([d["shift_ratio"] for d in diag]))}


def summarize(res):
    print("\nдоля большинства | " + " | ".join(f"{m}: Δ (p)" for m in METHODS[1:]))
    summary = {}
    for f in map(str, MAJORITY):
        none = np.array([res[s][f]["none"] for s in res])
        row = {}
        for m in METHODS[1:]:
            v = np.array([res[s][f][m] for s in res])
            p = float(ttest_rel(v, none).pvalue) if len(res) > 1 else float("nan")
            row[m] = {"delta": float((v - none).mean()), "p": p,
                      "worse_than_plain": int((v < np.array([res[s][f]["plain"] for s in res]) - 1e-9).sum())}
        summary[f] = row
        print(f"{float(f):>15.0%} | " + " | ".join(f"{row[m]['delta']:+.3f} ({row[m]['p']:.3f})" for m in METHODS[1:]))
    return summary


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
        out[str(sep)], diag = run_block(X[s], y[s], X[~s], y[~s], reps, rng)
        out[str(sep)]["_diag"] = diag
        print(f"domain_sep {sep} (уверенность {diag['conf']:.2f}, сдвиг/разброс {diag['shift_ratio']:.2f}): " + " | ".join(
            f"{int(f * 100)}%: " + " ".join(f"{m} {out[str(sep)][str(f)][m]:.3f}" for m in METHODS)
            for f in MAJORITY), flush=True)
    BUFFER = saved
    (bci_paths.RESULTS / "tlcenter_imbalance_synthetic.json").write_text(json.dumps(
        {"buffer": buffer, "reps": reps, "methods": METHODS, "results": out}, ensure_ascii=False, indent=1),
        encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["bnci", "dreyer"], default="bnci")
    ap.add_argument("--subjects", type=int, nargs="*")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    if a.synthetic:
        synthetic(a.reps)
        return
    loader = load_bnci if a.dataset == "bnci" else load_dreyer
    subjects = a.subjects or (list(range(1, 10)) if a.dataset == "bnci" else list(range(1, 88)))
    out_path = bci_paths.RESULTS / ("tlcenter_imbalance.json" if a.dataset == "bnci"
                                    else "tlcenter_imbalance_dreyer.json")
    rng = np.random.default_rng(0)
    res, diag = {}, {}
    for s in subjects:
        try:
            Cs, ys, Ct, yt = prepare(*loader(s))
        except Exception as e:      # нет данных / битый файл — пропускаем человека
            print(f"S{s}: пропуск — {e}", flush=True)
            continue
        res[s], diag[s] = run_block(Cs, ys, Ct, yt, a.reps, rng)
        print(f"S{s} (уверенность {diag[s]['conf']:.2f}, сдвиг/разброс {diag[s]['shift_ratio']:.2f}): " + " | ".join(
            f"{int(f * 100)}%: " + " ".join(f"{m[:4] if m in ('none', 'plain') else m.replace('pseudo_', 'p_')} {res[s][str(f)][m]:.0%}"
                                           for m in METHODS) for f in (0.5, 0.9, 1.0)), flush=True)
        out_path.write_text(json.dumps({"dataset": a.dataset, "buffer": BUFFER, "reps": a.reps,
                                        "methods": METHODS, "per_subject": res, "diag": diag},
                                       ensure_ascii=False, indent=1), encoding="utf-8")
    summary = summarize(res)
    out_path.write_text(json.dumps({"dataset": a.dataset, "buffer": BUFFER, "reps": a.reps, "methods": METHODS,
                                    "per_subject": res, "diag": diag, "summary": summary},
                                   ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
