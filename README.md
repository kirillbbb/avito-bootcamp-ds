# Avito Bootcamp DS — Bot Detection

Решение задачи классификации `cookie_id` на автоматизированный и человеческий трафик.

## Быстрый запуск

Требуется Python 3.13.

1. Положите в `data/` файлы:
   - `train.csv`
   - `test.csv`
   - `events.csv.gz`
2. Запустите из корня проекта:

```bash
python run.py
```

Скрипт автоматически:
- создаст `.venv`, если окружение ещё не существует;
- установит зависимости из `requirements.txt`;
- проверит наличие файлов данных;
- запустит подготовку признаков и обучение;
- создаст `submission.csv`.

Для проверки хронологической валидации:

```bash
python run.py --validate
```

Команда `run.py` кроссплатформенная: отдельная активация виртуального окружения не требуется.

## Ручной запуск

Если нужно запускать отдельные этапы вручную:

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python solution.py --data-dir data --validate
python solution.py --data-dir data --output submission.csv
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python solution.py --data-dir data --validate
python solution.py --data-dir data --output submission.csv
```

Также можно проверить реализацию метрики:

```bash
python metric.py
```

Ожидаемый результат:

```
metric: self-check passed
```

## Подход

- удаляются только полные дубликаты строк событий;
- события ограничиваются строгим окном `[window_start_ts, window_end_ts)`;
- строятся агрегаты по типам событий, платформам, объявлениям, категориям, локациям, продавцам и поиску;
- добавляются entropy/diversity, интервалы между событиями, интенсивность, временные признаки, pointer statistics и UA-флаги;
- train/test schema выравнивается явно;
- финальная модель — `CatBoostClassifier`, seed 42;
- в submission сохраняется `predict_proba[:, 1]`, порог не фиксируется.

Raw `cookie_id`, `item_id` и `search_query` не используются как идентификаторы модели: они нужны только для агрегатов.

## Валидация

Используются хронологические holdout по нескольким cutoff, а не случайный split. Метрика — Precision при Recall >= 70%, одинаковые score обрабатываются одной группой.

На проверенном holdout CatBoost был сильнее проверенных LightGBM и простого ансамбля, поэтому финальный pipeline оставлен на CatBoost.

Проверенный средний результат по трём хронологическим cutoff:

| Cutoff | P@R>=0.70 | Recall@1%FPR |
|---|---:|---:|
| 2026-04-17 | 0.727273 | 0.581250 |
| 2026-04-18 | 0.702128 | 0.542553 |
| 2026-04-19 | 0.731707 | 0.571429 |
| Mean | 0.720369 | — |

Это локальная валидация на предоставленном датасете; она не гарантирует такой же результат на скрытом тесте.

## Воспроизводимость

- Python 3.13
- CatBoost 1.2.8
- pandas 2.2.3
- NumPy 2.3.5
- random_seed=42

Данные соревнования не включаются в Git. Внешние API и LLM не используются.
