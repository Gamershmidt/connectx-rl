"""Агенты-соперники и эталонные агенты.

Единый интерфейс: agent.act(board, rng) -> column. Вся случайность агента идёт
через переданный rng (numpy Generator), поэтому партии воспроизводимы по seed.

Пул соперников:
  random          — равномерно случайный легальный ход (тривиальная нижняя граница);
  tactical        — правило: выиграть за 1 ход > заблокировать угрозу > случайный
                    «безопасный» ход (не подставляющий под немедленную победу соперника);
  lookahead       — negamax + alpha-beta на глубину 3 с эвристикой по «окнам» из 4 клеток
                    (идея эвристики — курс Kaggle «Intro to Game AI and RL», реализация своя);
  kaggle_negamax  — встроенный агент Kaggle (kaggle_environments, depth=4). Внешняя, не наша
                    реализация — честный «отложенный» соперник.
"""
from __future__ import annotations

import math
import random as _pyrandom
from typing import Callable

import numpy as np

from .board import COLS, ROWS, WINDOWS, Board

CENTER_ORDER = [3, 2, 4, 1, 5, 0, 6]


class Agent:
    name = "agent"

    def act(self, board: Board, rng: np.random.Generator) -> int:  # pragma: no cover
        raise NotImplementedError

    def reset(self) -> None:
        """Вызывается перед каждой партией (для агентов с состоянием)."""


# --------------------------------------------------------------------------- simple
class RandomAgent(Agent):
    name = "random"

    def act(self, board: Board, rng: np.random.Generator) -> int:
        moves = board.legal_moves()
        return moves[int(rng.integers(len(moves)))]


def safe_moves(board: Board) -> list[int]:
    """Ходы, после которых у соперника НЕТ немедленной победы."""
    me = board.to_move
    res = []
    for c in board.legal_moves():
        board.play(c)
        if not board.winning_moves(1 - me):
            res.append(c)
        board.undo(c)
    return res


class TacticalAgent(Agent):
    """Win-in-1 > block > safe random > random. Детерминирован с точностью до rng."""

    name = "tactical"

    def act(self, board: Board, rng: np.random.Generator) -> int:
        me = board.to_move
        wins = board.winning_moves(me)
        if wins:
            return wins[int(rng.integers(len(wins)))]
        threats = board.winning_moves(1 - me)
        if threats:
            return threats[int(rng.integers(len(threats)))]
        moves = safe_moves(board) or board.legal_moves()
        return moves[int(rng.integers(len(moves)))]


# ---------------------------------------------------------------------- lookahead
WIN_SCORE = 1_000_000
_CENTER_COL_MASK = ((1 << ROWS) - 1) << (3 * (ROWS + 1))


def heuristic(board: Board, player: int) -> float:
    """Оценка позиции для `player` по всем 69 окнам из 4 клеток."""
    me, opp = board.bb[player], board.bb[1 - player]
    score = 3 * (me & _CENTER_COL_MASK).bit_count() - 3 * (opp & _CENTER_COL_MASK).bit_count()
    for w in WINDOWS:
        a, b = me & w, opp & w
        if a and not b:
            n = a.bit_count()
            score += 10 if n == 3 else (1 if n == 2 else 0)
        elif b and not a:
            n = b.bit_count()
            score -= 12 if n == 3 else (1 if n == 2 else 0)
    return score


def _negamax(board: Board, depth: int, alpha: float, beta: float) -> float:
    me = board.to_move
    moves = board.legal_moves()
    if not moves:
        return 0.0
    for c in moves:  # немедленная победа
        if board.is_winning_move(c, me):
            return WIN_SCORE - board.n_moves
    if depth == 0:
        return heuristic(board, me)
    best = -math.inf
    for c in CENTER_ORDER:
        if c not in moves:
            continue
        board.play(c)
        val = -_negamax(board, depth - 1, -beta, -alpha)
        board.undo(c)
        if val > best:
            best = val
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


class LookaheadAgent(Agent):
    """Negamax с alpha-beta на `depth` полуходов; ничьи по оценке — случайно."""

    def __init__(self, depth: int = 3) -> None:
        self.depth = depth
        self.name = f"lookahead{depth}" if depth != 3 else "lookahead"

    def act(self, board: Board, rng: np.random.Generator) -> int:
        me = board.to_move
        moves = board.legal_moves()
        wins = [c for c in moves if board.is_winning_move(c, me)]
        if wins:
            return wins[int(rng.integers(len(wins)))]
        scores = {}
        for c in moves:  # корень — с полным окном, чтобы равенство оценок было точным
            board.play(c)
            scores[c] = -_negamax(board, self.depth - 1, -math.inf, math.inf)
            board.undo(c)
        best = max(scores.values())
        cands = [c for c in moves if scores[c] == best]
        return cands[int(rng.integers(len(cands)))]


# ---------------------------------------------------------------- kaggle adapters
class _Obs(dict):
    __getattr__ = dict.__getitem__


KAGGLE_CONFIG = _Obs(columns=COLS, rows=ROWS, inarow=4, actTimeout=2, timeout=2)


class KaggleFnAgent(Agent):
    """Обёртка над функцией в формате Kaggle `agent(observation, configuration)`."""

    def __init__(self, fn: Callable, name: str) -> None:
        self.fn = fn
        self.name = name

    def act(self, board: Board, rng: np.random.Generator) -> int:
        # встроенные агенты Kaggle используют модуль random — сидируем его из rng
        _pyrandom.seed(int(rng.integers(2**31)))
        obs = _Obs(board=board.to_kaggle_board(), mark=board.to_move + 1)
        return int(self.fn(obs, KAGGLE_CONFIG))


def kaggle_builtin(name: str) -> KaggleFnAgent:
    from kaggle_environments.envs.connectx.connectx import agents as kaggle_agents

    return KaggleFnAgent(kaggle_agents[name], f"kaggle_{name}")


def to_kaggle_fn(agent: Agent, seed: int = 0) -> Callable:
    """Обратное направление: наш агент -> функция для kaggle_environments."""
    rng = np.random.default_rng(seed)

    def fn(observation, configuration):
        board = Board.from_kaggle(list(observation["board"]), int(observation["mark"]))
        return int(agent.act(board, rng))

    return fn


# ----------------------------------------------------------------------- registry
def make_agent(spec: str) -> Agent:
    """Фабрика: 'random' | 'tactical' | 'lookahead' | 'lookahead<d>' | 'kaggle_negamax'
    | 'kaggle_random' | 'dqn:<path/to/checkpoint.pt>'."""
    if spec == "random":
        return RandomAgent()
    if spec == "tactical":
        return TacticalAgent()
    if spec.startswith("lookahead"):
        d = spec[len("lookahead"):]
        return LookaheadAgent(int(d) if d else 3)
    if spec.startswith("kaggle_"):
        return kaggle_builtin(spec[len("kaggle_"):])
    if spec.startswith("dqn:"):
        from .model import DQNAgent

        return DQNAgent.from_checkpoint(spec[len("dqn:"):])
    raise ValueError(f"неизвестный агент: {spec}")
