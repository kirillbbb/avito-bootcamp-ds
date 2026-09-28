# Avito Bootcamp DS — Bot Detection

Решение задачи классификации cookie_id на автоматизированный и человеческий трафик.

## Запуск

Требуется Python 3.13.5. Положите в data/: train.csv, test.csv, events.csv.gz.

PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python solution.py --data-dir data --output submission.csv

Для временной валидации: python solution.py --data-dir data --validate

## Подход

- удаляются только полные дубликаты строк событий;
- события ограничиваются строгим окном [window_start_ts, window_end_ts);
- строятся агрегаты по типам событий, платформам, объявлениям, категориям, локациям, продавцам и поиску;
- добавляются entropy/diversity, интервалы между событиями, интенсивность, временные признаки, pointer statistics и UA-флаги;
- train/test schema выравнивается явно;
- финальная модель — CatBoostClassifier, seed 42;
- в submission сохраняется predict_proba[:, 1], порог не фиксируется.

Raw cookie_id, item_id и search_query не используются как идентификаторы модели: они нужны только для агрегатов.

## Валидация

Используются хронологические holdout по нескольким cutoff, а не случайный split. Метрика — Precision при Recall >= 70%, одинаковые score обрабатываются одной группой.

На проверенном holdout CatBoost был сильнее проверенных LightGBM и простого ансамбля, поэтому финальный pipeline оставлен на CatBoost.

## Воспроизводимость

- Python 3.13.5
- CatBoost 1.2.8
- pandas 2.2.3
- NumPy 2.3.5
- random_seed=42

Данные соревнования не включаются в Git. Внешние API и LLM не используются.