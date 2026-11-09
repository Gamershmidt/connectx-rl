"""Проверка корректности собственного движка против официальной реализации Kaggle."""
import random

import numpy as np
import pytest

from connectx_rl.agents import TacticalAgent, make_agent, to_kaggle_fn
from connectx_rl.board import Board, has_won

ke = pytest.importorskip("kaggle_environments")
from kaggle_environments.envs.connectx.connectx import is_win, play  # noqa: E402


class Cfg(dict):
    __getattr__ = dict.__getitem__


CFG = Cfg(columns=7, rows=6, inarow=4)


def test_vertical_horizontal_diagonal():
    assert Board.from_moves([0, 1, 0, 1, 0, 1]).is_winning_move(0)
    assert Board.from_moves([0, 0, 1, 1, 2, 2]).is_winning_move(3)
    # диагональ / : X в (0,0),(1,1),(2,2),(3,3)
    b = Board.from_moves([0, 1, 1, 2, 2, 3, 2, 3, 3, 6])
    assert b.is_winning_move(3)
    assert not has_won(0)


@pytest.mark.parametrize("seed", range(200))
def test_random_games_match_kaggle(seed):
    """Случайные партии: победа/ничья и доска совпадают с kaggle_environments."""
    rng = random.Random(seed)
    b = Board()
    kb = [0] * 42
    while True:
        moves = b.legal_moves()
        assert moves == [c for c in range(7) if kb[c] == 0]
        c = rng.choice(moves)
        mark = b.to_move + 1
        ours = b.is_winning_move(c)
        play(kb, c, mark, CFG)
        b.play(c)
        assert b.to_kaggle_board() == kb
        assert ours == is_win(kb, c, mark, CFG)
        assert Board.from_kaggle(kb).to_kaggle_board() == kb
        if ours or b.is_full():
            break


def test_planes_perspective():
    b = Board.from_moves([3])  # X сыграл, ходит O
    p = b.planes()
    assert p[0].sum() == 0 and p[1].sum() == 1 and p[1][5, 3] == 1


def test_tactical_wins_and_blocks():
    rng = np.random.default_rng(0)
    b = Board.from_moves([0, 6, 0, 6, 0])  # O должен блокировать столбец 0
    assert TacticalAgent().act(b, rng) == 0
    b = Board.from_moves([0, 6, 0, 6, 0, 6, 1])  # O выигрывает в столбце 6
    assert TacticalAgent().act(b, rng) == 6


def test_kaggle_env_runs_our_agent():
    """Наш агент через адаптер играет в официальной среде без ошибок."""
    from kaggle_environments import make

    env = make("connectx", debug=True)
    env.run([to_kaggle_fn(make_agent("tactical")), "random"])
    assert env.state[0].status == "DONE" and env.state[1].status == "DONE"
