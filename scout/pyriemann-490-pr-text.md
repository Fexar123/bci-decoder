# Текст PR в pyRiemann (опубликован 17.11.2026: https://github.com/pyRiemann/pyRiemann/pull/515)

Откуда: ветка `Fexar123:doc-tlcenter-class-imbalance` → `pyRiemann/pyRiemann:master`.
Заголовок: **Document class imbalance bias of TLCenter and its correction with sample weights**

---

#### Reference Issues/PRs
Fixes #490.

#### What does this implement/fix? Explain your changes.
Following the discussion in #490, this PR completes the documentation and fleshes out the existing RPA example, as suggested by @qbarthelemy. No change in behaviour.

- `TLCenter`: add a note explaining that the center of each domain is an unweighted mean, so it implicitly assumes similar class proportions across domains. A class-imbalanced domain (e.g. a buffer of recent unlabeled trials in online use) gives a biased center, which can decrease accuracy, especially when the domain shift is small. The note explains how to correct it with class-balanced `sample_weight`, and how to estimate target labels when they are unknown (classifier trained on source, applied before target recentering), with its limitation under large shifts.
- `examples/transfer/plot_rpa_steps.py`: new section "Recentering with a class-imbalanced target domain". The target domain keeps all matrices of one class and only 5 of the other; unweighted vs class-balanced weighted recentering, with a figure in the same style as the existing one, and MDM accuracy on the whole target domain: 0.64 without recentering, 0.86 after unweighted recentering, 1.00 after weighted recentering.
- `whatsnew.rst`: entry.

Checked locally: `flake8 examples tests pyriemann setup.py` passes, `pytest` passes, the example runs. I did not build the full documentation locally.

#### AI usage disclosure
I used AI assistance for:
- Documentation (including examples)
- Research and understanding

The experiments behind #490 (BCI IV-2a, Dreyer2023) and these changes were checked by me.

#### Any other comments?
Happy to shorten the note or the example section if you prefer something more compact.

---

## Перевод для Севы
- **Ссылка:** закрывает задачу #490.
- **Что сделано:** по обсуждению в #490 дополнил документацию и расширил существующий пример, как просил qbarthelemy. Поведение кода не меняется.
  - **TLCenter:** заметка в справке. Центр каждого домена — среднее без весов, поэтому он молча предполагает одинаковые доли классов. При перекосе (например, буфер последних попыток без ответов) центр съезжает, точность может упасть — особенно если сдвиг между доменами маленький. Как исправить: веса «1 / размер класса» через `sample_weight`. Если ответов нет — угадывать классы моделью источника ДО подстройки цели, но при большом сдвиге это ненадёжно.
  - **Пример:** новый раздел. В цели оставлены все матрицы одного класса и только 5 другого. Сравнение подстройки без весов и с весами, картинка в стиле существующей. Точность на всей цели: 0,64 без подстройки, 0,86 после обычной, 1,00 после подстройки с весами.
  - **whatsnew:** строка в список изменений.
- **Проверено у себя:** flake8 и pytest проходят, пример запускается. Полную документацию локально не собирал.
- **Про ИИ:** использовал ИИ для документации, примеров и разбора темы. Эксперименты из #490 и эти изменения проверены мной.
- **Другое:** готов сократить заметку или пример, если нужно компактнее.
