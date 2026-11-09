# ML R&D: ConnectX
## Проект: Обучение игрового агента с подкреплением (RL) для победы в игре «Четыре в ряд» против других ботов

Задача — [Kaggle ConnectX](https://www.kaggle.com/competitions/connectx): поле 6×7, четыре в ряд.

Ветка `main` содержит каркас проекта. Домашние задания выполняются в отдельных ветках и вливаются через pull request:

| Работа | Ветка | Содержание |
|---|---|---|
| ДЗ1 | `hw1` | Постановка задачи, гипотезы, EDA, DQN baseline, протокол оценки |

## Структура

```text
configs/           # конфиги экспериментов
scripts/           # точки входа для запуска
src/connectx_rl/   # код: среда, агенты, обучение, оценка
tests/             # тесты
results/           # отчёты, метрики, графики
```

## Окружение

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```
