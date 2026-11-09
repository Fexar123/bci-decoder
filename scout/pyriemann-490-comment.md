# Черновик комментария в pyRiemann #490

Куда: https://github.com/pyRiemann/pyRiemann/issues/490 — от аккаунта Fexar123.
Статус: **опубликован 09.11.2026** от Fexar123 — https://github.com/pyRiemann/pyRiemann/issues/490#issuecomment-6750783190 (без пометки про ИИ — решение Севы).
Код: `projects/bci/tlcenter_imbalance.py`; повтор без скачиваний: `scout/repro_490_weights.py`.
Сырые цифры: `results/tlcenter_imbalance.json` (BCI IV 2a), `results/tlcenter_imbalance_dreyer.json` (Dreyer2023), `results/tlcenter_imbalance_synthetic.json`.

---

## Текст (английский, как уйдёт в GitHub)

Hi @qbarthelemy — the question about `sample_weight` was still open, so I ran it on the synthetic setup from this issue, on BCI IV-2a (n=9) and on Dreyer2023 (n=30). Short answer: **`sample_weight` is a sufficient mechanism, but whether it helps depends entirely on where the weights come from — and there is a simple label-free choice that works.**

**Setup.** `TLCenter` → `TangentSpace` → `LogisticRegression`, binary MI. Target buffer of unlabelled trials with a controlled majority-class fraction; evaluation on the full balanced target set. Buffer weights are inverse class frequency (each class sums to 1):

```python
def balanced_weights(labels):
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        return np.ones(len(labels))
    w = dict(zip(classes, 1.0 / counts))
    return np.array([w[c] for c in labels])
```

Where the labels for the weights come from:
- `plain` — no weights (current default);
- `oracle` — true buffer labels (upper bound, not available online);
- `pseudo_post` — labels predicted *after* unweighted centering;
- `pseudo_pre` — labels predicted *before* target centering (buffer centred with the source center);
- `pseudo_dist` — `pseudo_pre`, but only if the target shift is small: Riemannian distance of the source-centred buffer mean to I is below the median distance of the source matrices to I; otherwise no weights.

**Dreyer2023** (calibration runs → online runs within subject, 8–30 Hz, OAS, buffer 40, n=30, 10 buffers per cell; Δ = accuracy minus no adaptation, paired t-test p):

| majority in buffer | plain | oracle | pseudo_post | pseudo_pre | pseudo_dist |
|---|---|---|---|---|---|
| 50% | +0.008 (0.17) | +0.008 (0.17) | +0.007 (0.23) | +0.008 (0.05) | +0.008 (0.05) |
| 70% | −0.001 (0.84) | +0.007 (0.28) | +0.002 (0.73) | +0.008 (0.05) | +0.007 (0.05) |
| 80% | −0.014 (0.06) | +0.007 (0.24) | −0.007 (0.25) | +0.007 (0.10) | +0.006 (0.12) |
| 90% | **−0.028 (0.008)** | +0.005 (0.39) | **−0.021 (0.020)** | +0.005 (0.27) | +0.003 (0.37) |
| 100% | **−0.046 (0.002)** | **−0.046 (0.002)** | **−0.047 (0.002)** | −0.008 (0.31) | −0.008 (0.27) |

`pseudo_dist` vs `plain` directly (paired, n=30): 50% ±0.0 (p=0.96), 70% +0.009 (p=0.014), 80% +0.020, 90% +0.031, 100% +0.038 (all p<0.001).

**BCI IV-2a** (T→E, buffer 40, n=9, 20 buffers per cell) — same direction, not significant at n=9: `plain` goes from +0.048 (50%) to −0.059 (100%); `pseudo_dist` from +0.034 to −0.014; `oracle` stays at ≈ +0.045 up to 90%.

**Synthetic** (`make_classification_transfer`, class_sep=2, class_disp=1, theta=π/6, 200 matrices, buffer 60, seed 42 — as in the repro script). The numbers from the original report reproduce (`plain` ≈ 0.86 balanced, ≈ 0.73–0.75 single-class).

| domain_sep | no centering | plain 90% | plain 100% | oracle 90% | pseudo_post 90% | pseudo_pre 90% | pseudo_pre 100% | pseudo_dist 90% | pseudo_dist 100% |
|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.830 | 0.783 | 0.748 | 0.853 | 0.792 | 0.838 | 0.847 | 0.838 | 0.847 |
| 1.0 | 0.790 | 0.785 | 0.741 | 0.855 | 0.798 | 0.821 | 0.803 | 0.811 | 0.779 |
| 3.0 | 0.510 | 0.780 | 0.727 | 0.854 | 0.799 | 0.714 | 0.673 | 0.780 | 0.727 |

**Takeaways**
1. **True class-balanced weights remove the imbalance penalty** for partial imbalance — the mechanism is enough. They can't help a single-class buffer (nothing to balance).
2. **Labels predicted after unweighted centering don't help** (`pseudo_post` ≈ `plain`, still significantly harmful on Dreyer). Centering on the biased mean makes the buffer look neutral, so predictions come out close to 50/50 and the weights are nearly uniform — the estimate is circular.
3. **Labels predicted before target centering work in the regime where the issue reports harm** (small shift): on Dreyer they match `oracle` at 70–90% and also fix the single-class case (−0.008 vs −0.046), at no cost on balanced buffers. Our guess for the 100% case: the few minority predictions get half the total weight and pull the center back towards the class boundary.
4. **Under a large shift pre-centering predictions are near chance** and `pseudo_pre` hurts (synthetic domain_sep=3). The distance gate (`pseudo_dist`) falls back to unweighted centering there; we fixed the threshold (median source dispersion) before running anything and did not tune it.
5. Confidence-based gating does *not* work as a safeguard: under a large shift the pre-centering model is confidently wrong (mean max-probability ≈ 0.95 at domain_sep=3).

So I'd support the documentation note proposed in the issue, plus a short example: class-balanced `sample_weight` from pre-centering pseudo-labels, gated by the target shift. Happy to turn `pseudo_dist` into a doc example if that's useful. A self-contained script (synthetic data, no downloads) is below.

Tested with pyriemann 0.12. (#506 changed `TLCenter.transform` with `target_domain="last"`; we always pass the target domain explicitly.)

<details><summary>repro script</summary>

(вставить содержимое `scout/repro_490_weights.py`)

</details>

---

## Перевод для Севы (по-русски, по смыслу)

Привет. Вопрос про `sample_weight` так и висел, поэтому я проверил его на искусственных данных из задачи, на BCI IV-2a (9 человек) и на Dreyer2023 (30 человек). Коротко: **самого механизма весов хватает, но польза зависит только от того, откуда берутся веса. И есть простой вариант без ответов, который работает.**

**Как проверяли.** Цепочка TLCenter → TangentSpace → LogisticRegression, 2 класса. В буфере без ответов задана доля класса большинства, проверка — на всём сбалансированном наборе. Веса = 1 / размер класса. Откуда берём классы для весов:
- `plain` — без весов, как сейчас;
- `oracle` — настоящие ответы (потолок);
- `pseudo_post` — догадки ПОСЛЕ подстройки без весов;
- `pseudo_pre` — догадки ДО подстройки, по вчерашнему центру;
- `pseudo_dist` — догадки ДО, но только если сдвиг маленький: расстояние среднего буфера до центра меньше обычного разброса попыток. Иначе без весов.

**Dreyer2023, 30 человек:** без весов при перекосе 90% и 100% подстройка **значимо вредит** (−2,8 и −4,6 пункта). Догадки ПОСЛЕ — тоже вредят. Догадки ДО и рецепт по сдвигу — не вредят. Напрямую рецепт против «без весов»: на сбалансированном буфере ничья, при перекосе от +0,9 до +3,8 пункта, всё значимо.

**BCI IV-2a, 9 человек:** то же направление, но без значимости — людей мало.

**Искусственные данные:** цифры автора повторились. Там видно, что при большом сдвиге догадки ДО ломаются, а рецепт по сдвигу в этом случае просто выключает веса.

**Выводы:**
1. Веса по настоящим ответам убирают штраф при частичном перекосе — механизма хватает. При одном классе не помогают.
2. Догадки ПОСЛЕ не помогают: замкнутый круг.
3. Догадки ДО работают при маленьком сдвиге. На Dreyer они догоняют настоящие ответы и спасают даже буфер из одного класса, без потерь на сбалансированном.
4. При большом сдвиге догадки ДО на уровне монетки. Рецепт по сдвигу в этом случае откатывается к «без весов». Порог мы задали заранее и не подбирали.
5. Защита «по уверенности модели» не работает: при большом сдвиге модель уверенно ошибается.

Поддерживаю заметку в документации и предлагаю пример с рецептом по сдвигу. Могу оформить его. Ниже скрипт для повтора без скачиваний.

---

## Что решить перед публикацией

1. **Понимаешь ли ты каждый пункт?** Если спросят «почему pseudo_post круговой?» или «как выбран порог?» — отвечать тебе.
2. **Писать ли, что помогал ИИ?** Правил pyRiemann на этот счёт я не проверял. Решаешь ты.
3. Вставить в спойлер текст `scout/repro_490_weights.py` (78 строк).
4. После «да» публикую через `gh` от Fexar123 — один комментарий, больше ничего.
