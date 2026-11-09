# Черновик комментария в pyRiemann #490

Куда: https://github.com/pyRiemann/pyRiemann/issues/490 — от аккаунта Fexar123.
Статус: **черновик, НЕ опубликован.** Публикуется только после «да» Севы.
Код: `projects/bci/tlcenter_imbalance.py`, сырые цифры: `results/tlcenter_imbalance.json`, `results/tlcenter_imbalance_synthetic.json`.

---

## Текст (английский, как уйдёт в GitHub)

Hi @qbarthelemy, the original question about `sample_weight` was still open, so I ran it on both the synthetic setup from this issue and BCI IV-2a. Short answer: **`sample_weight` is a sufficient mechanism, but whether it helps depends entirely on where the weights come from.**

**Setup.** `TLCenter` → `TangentSpace` → `LogisticRegression`, binary. Target buffer = unlabelled trials with a controlled majority-class fraction; evaluation on the full balanced target set; 20 random buffers per cell. Weights for the buffer are inverse class frequency (each class sums to 1):

```python
def balanced_weights(labels):
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        return np.ones(len(labels))
    w = dict(zip(classes, 1.0 / counts))
    return np.array([w[c] for c in labels])
```

Variants for the target-domain weights:
- `plain` — no weights (current default);
- `oracle` — weights from the true buffer labels (upper bound, not available online);
- `pseudo_post` — weights from labels predicted *after* unweighted centering;
- `pseudo_pre` — weights from labels predicted *before* target centering (target matrices centred with the source center);
- `pseudo_iter` — `pseudo_pre` + 3 rounds of (weighted centering → re-predict → re-weight).

**Synthetic** (`make_classification_transfer`, class_sep=2, class_disp=1, theta=π/6, 200 matrices, buffer 60, seed 42 — same as the repro script). The numbers from the original report reproduce: `plain` ≈ 0.86 on a balanced buffer, ≈ 0.73–0.75 on a single-class buffer.

| domain_sep | no centering | plain 50% | plain 90% | plain 100% | oracle 90% | pseudo_post 90% | pseudo_pre 90% | pseudo_pre 100% |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.830 | 0.860 | 0.783 | 0.748 | 0.853 | 0.792 | 0.838 | **0.847** |
| 1.0 | 0.790 | 0.862 | 0.785 | 0.741 | 0.855 | 0.798 | 0.821 | **0.803** |
| 1.5 | 0.705 | 0.859 | 0.777 | 0.739 | 0.849 | 0.791 | 0.777 | 0.789 |
| 2.0 | 0.608 | 0.862 | 0.791 | 0.741 | 0.856 | 0.809 | 0.752 | 0.722 |
| 3.0 | 0.510 | 0.861 | 0.780 | 0.727 | 0.854 | 0.799 | 0.714 | 0.673 |

**BCI IV-2a** (T→E within subject, 8–30 Hz, OAS covariances, buffer 40, n=9; Δ = accuracy minus no adaptation, mean over subjects, paired t-test p):

| majority in buffer | plain | oracle | pseudo_post | pseudo_pre | pseudo_iter |
|---|---|---|---|---|---|
| 50% | +0.048 (0.16) | +0.048 (0.16) | +0.046 (0.17) | +0.027 (0.17) | +0.043 (0.20) |
| 70% | +0.025 (0.43) | +0.047 (0.17) | +0.032 (0.30) | +0.025 (0.18) | +0.037 (0.24) |
| 80% | +0.001 (0.99) | +0.046 (0.17) | +0.016 (0.60) | +0.026 (0.17) | +0.031 (0.29) |
| 90% | −0.026 (0.51) | +0.042 (0.21) | −0.019 (0.62) | +0.020 (0.20) | +0.018 (0.51) |
| 100% | −0.059 (0.20) | −0.059 (0.20) | −0.059 (0.22) | −0.017 (0.45) | −0.050 (0.22) |

None of the real-data differences is significant with n=9 (one subject, S7, has a very large session shift and dominates the variance), but the direction matches the synthetic results.

**Takeaways**
1. **True class-balanced weights fully remove the imbalance penalty** for partial imbalance (70–90%): `oracle` stays at the balanced-buffer level in every condition. So the mechanism itself is enough.
2. **They cannot help a single-class buffer** — there is nothing to balance (`oracle` = `plain` at 100%).
3. **Pseudo-labels predicted after unweighted centering do not help** (`pseudo_post` ≈ `plain`). Centering on the biased buffer mean makes the buffer look neutral, so the predictions come out close to 50/50 and the weights end up nearly uniform — the estimate is circular.
4. **Pseudo-labels predicted before target centering work exactly in the regime where the issue reports harm** (small domain shift, uncentered model already decent): they recover most of the loss and even fix the single-class case (0.847 vs 0.748 at domain_sep=0.5). Our guess as to why: the few minority predictions in a single-class buffer get half the total weight, which pulls the center back towards the class boundary. **Under a large shift they fail**, because the pre-centering predictions are near chance (0.67 vs 0.73 at domain_sep=3).
5. Iterating (`pseudo_iter`) is a compromise between the two, never the best.

So I'd support the documentation note proposed above, and add a short pointer that `sample_weight` with class-balanced weights addresses label shift when labels (or reliable pseudo-labels) are available. Choosing the pseudo-label source automatically (e.g. from prediction confidence) looks like the open part. Happy to turn the snippet above into a doc example if useful.

---

## Перевод для Севы (по-русски, по смыслу)

Привет. Твой вопрос про `sample_weight` так и висел, поэтому я прогнал его на искусственных данных из этой задачи и на BCI IV-2a. Коротко: **сам механизм весов достаточен, но поможет он или нет, зависит только от того, откуда берутся веса.**

**Как проверяли.** Цепочка TLCenter → TangentSpace → LogisticRegression, 2 класса. Буфер нового дня без ответов, доля класса большинства задана. Проверка на всём сбалансированном дне, 20 случайных буферов на клетку. Веса: 1 / размер класса (каждый класс в сумме весит 1). Варианты весов:
- `plain` — без весов, как сейчас;
- `oracle` — по настоящим ответам (потолок, в жизни недоступно);
- `pseudo_post` — по догадкам модели ПОСЛЕ подстройки без весов;
- `pseudo_pre` — по догадкам ДО подстройки;
- `pseudo_iter` — pseudo_pre плюс 3 круга уточнения.

**Искусственные данные** — те же настройки, что в скрипте автора. Его цифры повторились: ≈ 0.86 на сбалансированном буфере и ≈ 0.73–0.75 на буфере из одного класса.

**Реальные данные** (9 человек): ни одна разница статистически не значима — людей мало, а S7 с огромным сдвигом между днями даёт большой разброс. Но направление то же, что на искусственных данных.

**Выводы:**
1. **Веса по настоящим ответам полностью убирают штраф** при частичном перекосе (70–90%). Значит, механизма хватает.
2. **При буфере из одного класса не помогают** — уравнивать не с чем.
3. **Догадки ПОСЛЕ подстройки без весов не помогают:** подстройка по кривому среднему делает буфер «нейтральным», догадки выходят ~50/50, веса почти одинаковые. Замкнутый круг.
4. **Догадки ДО подстройки работают ровно там, где автор задачи показывал вред** (маленький сдвиг, модель и без подстройки неплохая): почти всё чинят и даже спасают буфер из одного класса. Наше предположение почему: несколько «ошибочных» догадок второго класса получают половину общего веса и тянут центр обратно к границе между классами. **При большом сдвиге не работают**: догадки до подстройки на уровне монетки.
5. Круги уточнения — середина, никогда не лучший вариант.

Поэтому поддерживаю заметку в документации, которую предлагал автор, и предлагаю добавить подсказку про веса. Открытый вопрос — как автоматически выбирать, откуда брать догадки (например, по уверенности модели). Могу оформить пример в документацию.

---

## Что решить перед публикацией

1. **Понимаешь ли ты каждый пункт?** Если разработчик спросит «почему pseudo_post круговой?» — отвечать тебе. Непонятные места разберём.
2. **Писать ли, что помогал ИИ?** Правил pyRiemann на этот счёт я не проверял. Честнее одной строкой упомянуть, что анализ делался с помощью ИИ-ассистента. Решаешь ты.
3. После «да» публикую через `gh` от Fexar123 — один комментарий, больше ничего.
