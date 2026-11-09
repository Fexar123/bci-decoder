"""Self-contained check for pyRiemann #490: does sample_weight fix class-imbalanced
target buffers in TLCenter? Needs only pyriemann + scikit-learn, no downloads.

Weights for the unlabelled target buffer, all passed via TLCenter.fit(sample_weight=...):
  plain       no weights (current default)
  oracle      class-balanced weights from the true buffer labels (upper bound)
  pseudo_post class-balanced weights from labels predicted AFTER unweighted centering
  pseudo_pre  class-balanced weights from labels predicted BEFORE target centering
  pseudo_dist pseudo_pre, used only if the target shift (Riemannian distance of the
              buffer mean to the source center) is smaller than the median source
              dispersion; otherwise no weights
"""
import numpy as np
from pyriemann.datasets import make_classification_transfer
from pyriemann.tangentspace import TangentSpace
from pyriemann.transfer import TLCenter, decode_domains, encode_domains
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.mean import mean_riemann
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

BUFFER, REPS = 60, 20
METHODS = ["none", "plain", "oracle", "pseudo_post", "pseudo_pre", "pseudo_dist"]


def balanced_weights(labels):
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        return np.ones(len(labels))
    w = dict(zip(classes, 1.0 / counts))
    return np.array([w[c] for c in labels])


def evaluate(Xs, ys, Xbuf, ybuf, Xt, yt):
    X, y_enc = encode_domains(np.concatenate([Xs, Xbuf]), np.r_[ys, np.zeros(len(Xbuf), int)],
                              np.r_[["src"] * len(Xs), ["tgt"] * len(Xbuf)])
    plain = TLCenter(target_domain="tgt").fit(X, y_enc)
    src = plain.centers_["src"]
    Xs_c = src.transform(Xs)
    clf = make_pipeline(TangentSpace(), LogisticRegression(max_iter=1000)).fit(Xs_c, ys)
    acc = {"none": clf.score(src.transform(Xt), yt), "plain": clf.score(plain.transform(Xt), yt)}

    Xbuf_src = src.transform(Xbuf)
    w_pre = balanced_weights(clf.predict(Xbuf_src))
    eye = np.eye(Xs.shape[1])
    shift = distance_riemann(mean_riemann(Xbuf_src), eye)
    spread = np.median([distance_riemann(c, eye) for c in Xs_c])
    weights = {
        "oracle": balanced_weights(ybuf),
        "pseudo_post": balanced_weights(clf.predict(plain.transform(Xbuf))),
        "pseudo_pre": w_pre,
        "pseudo_dist": w_pre if shift < spread else np.ones(len(Xbuf)),
    }
    for m, wb in weights.items():
        tl = TLCenter(target_domain="tgt").fit(X, y_enc, sample_weight=np.r_[np.ones(len(Xs)), wb])
        acc[m] = clf.score(tl.transform(Xt), yt)
    return acc


rng = np.random.default_rng(0)
print("domain_sep  majority  " + "  ".join(f"{m:>11}" for m in METHODS))
for sep in (0.5, 1.0, 3.0):
    X, y_enc = make_classification_transfer(n_matrices=200, class_sep=2.0, class_disp=1.0,
                                            domain_sep=sep, theta=np.pi / 6, random_state=42)
    X, y, d = decode_domains(X, y_enc)
    y = (y == np.unique(y)[1]).astype(int)
    s = d == "source_domain"
    Xs, ys, Xt, yt = X[s], y[s], X[~s], y[~s]
    for frac in (0.5, 0.9, 1.0):
        res = {m: [] for m in METHODS}
        for _ in range(REPS):
            major = rng.integers(2)
            n_major = int(round(frac * BUFFER))
            b = np.r_[rng.choice(np.where(yt == major)[0], n_major, replace=False),
                      rng.choice(np.where(yt != major)[0], BUFFER - n_major, replace=False)]
            for m, v in evaluate(Xs, ys, Xt[b], yt[b], Xt, yt).items():
                res[m].append(v)
        print(f"{sep:>10}  {frac:>8.0%}  " + "  ".join(f"{np.mean(res[m]):>11.3f}" for m in METHODS))
