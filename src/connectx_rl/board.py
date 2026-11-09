"""Быстрый движок Connect Four (6x7, 4 в ряд) на битбордах.

Раскладка битов (классическая схема J. Tromp / P. Pons): столбец c занимает
биты c*7 .. c*7+5 (снизу вверх), бит c*7+6 — «сторожевой» и всегда пуст.
Благодаря сторожевому биту проверка «4 в ряд» сводится к четырём сдвигам.

Нумерация игроков: player 0 = mark 1 (ходит первым), player 1 = mark 2.
Формат Kaggle: observation.board — список из 42 чисел построчно СВЕРХУ вниз,
0 = пусто, 1/2 = фишки игроков.
"""
from __future__ import annotations

import numpy as np

ROWS, COLS, INAROW = 6, 7, 4
H = ROWS + 1  # высота столбца в битборде, включая сторожевой бит
N_CELLS = ROWS * COLS

_BOTTOM = [1 << (c * H) for c in range(COLS)]
_TOP = [1 << (c * H + ROWS - 1) for c in range(COLS)]
_COL_MASK = [((1 << ROWS) - 1) << (c * H) for c in range(COLS)]
# бит -> (row_from_top, col) для конвертации в матрицу
_BIT_TO_RC = {c * H + r: (ROWS - 1 - r, c) for c in range(COLS) for r in range(ROWS)}


def has_won(bb: int) -> bool:
    """Есть ли в битборде 4 фишки в ряд (вертикаль, горизонталь, 2 диагонали)."""
    for shift in (1, H, H - 1, H + 1):
        m = bb & (bb >> shift)
        if m & (m >> (2 * shift)):
            return True
    return False


def _all_windows() -> list[int]:
    """Все 69 «окон» из 4 клеток, в которых возможна победа (битовые маски)."""
    wins = []
    dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]  # (dc, dr): гориз., верт., диагонали
    for c in range(COLS):
        for r in range(ROWS):
            for dc, dr in dirs:
                cells = [(c + k * dc, r + k * dr) for k in range(INAROW)]
                if all(0 <= cc < COLS and 0 <= rr < ROWS for cc, rr in cells):
                    wins.append(sum(1 << (cc * H + rr) for cc, rr in cells))
    return wins


WINDOWS = _all_windows()
assert len(WINDOWS) == 69


class Board:
    """Позиция Connect Four. Мутабельна: play/undo для быстрого поиска."""

    __slots__ = ("bb", "heights", "n_moves")

    def __init__(self) -> None:
        self.bb = [0, 0]
        self.heights = [c * H for c in range(COLS)]  # индекс следующего свободного бита
        self.n_moves = 0

    # ------------------------------------------------------------------ basics
    @property
    def to_move(self) -> int:
        """Индекс игрока, который сейчас ходит (0 или 1)."""
        return self.n_moves & 1

    @property
    def mask(self) -> int:
        return self.bb[0] | self.bb[1]

    def copy(self) -> "Board":
        b = Board.__new__(Board)
        b.bb = self.bb[:]
        b.heights = self.heights[:]
        b.n_moves = self.n_moves
        return b

    def is_legal(self, col: int) -> bool:
        return 0 <= col < COLS and self.heights[col] < col * H + ROWS

    def legal_moves(self) -> list[int]:
        return [c for c in range(COLS) if self.heights[c] < c * H + ROWS]

    def legal_mask(self) -> np.ndarray:
        return np.array([self.heights[c] < c * H + ROWS for c in range(COLS)], dtype=bool)

    def play(self, col: int) -> None:
        self.bb[self.n_moves & 1] |= 1 << self.heights[col]
        self.heights[col] += 1
        self.n_moves += 1

    def undo(self, col: int) -> None:
        self.n_moves -= 1
        self.heights[col] -= 1
        self.bb[self.n_moves & 1] ^= 1 << self.heights[col]

    def is_full(self) -> bool:
        return self.n_moves >= N_CELLS

    def winner(self) -> int | None:
        """0/1 — индекс победителя, None — победителя нет (пока или ничья)."""
        if has_won(self.bb[0]):
            return 0
        if has_won(self.bb[1]):
            return 1
        return None

    def is_winning_move(self, col: int, player: int | None = None) -> bool:
        """Выиграет ли `player` (по умолчанию — тот, кто ходит), сыграв в `col`."""
        if player is None:
            player = self.n_moves & 1
        return has_won(self.bb[player] | (1 << self.heights[col]))

    def winning_moves(self, player: int | None = None) -> list[int]:
        """Все столбцы, ход в которые немедленно выигрывает для `player`.

        Для соперника (player != to_move) это «угрозы»: клетки, куда он выиграет,
        если ему позволить туда сыграть прямо сейчас.
        """
        if player is None:
            player = self.n_moves & 1
        return [c for c in self.legal_moves() if self.is_winning_move(c, player)]

    # ------------------------------------------------------------ conversions
    def to_grid(self) -> np.ndarray:
        """Матрица 6x7 в формате Kaggle (строка 0 — верхняя): 0/1/2."""
        g = np.zeros((ROWS, COLS), dtype=np.int8)
        for p in (0, 1):
            bb = self.bb[p]
            while bb:
                low = bb & -bb
                r, c = _BIT_TO_RC[low.bit_length() - 1]
                g[r, c] = p + 1
                bb ^= low
        return g

    def to_kaggle_board(self) -> list[int]:
        return self.to_grid().reshape(-1).tolist()

    def planes(self) -> np.ndarray:
        """Признаки для нейросети с точки зрения ходящего игрока: (2, 6, 7) float32.

        Канал 0 — фишки ходящего игрока, канал 1 — фишки соперника.
        Каноническая перспектива позволяет одной сети играть за обе стороны.
        """
        g = self.to_grid()
        me = self.to_move + 1
        opp = 3 - me
        return np.stack([(g == me), (g == opp)]).astype(np.float32)

    @classmethod
    def from_kaggle(cls, board: list[int], mark: int | None = None) -> "Board":
        """Восстановить позицию из observation.board (порядок ходов неважен)."""
        b = cls()
        g = np.asarray(board, dtype=np.int8).reshape(ROWS, COLS)
        for c in range(COLS):
            for r_top in range(ROWS - 1, -1, -1):  # снизу вверх
                v = int(g[r_top, c])
                if v == 0:
                    break
                b.bb[v - 1] |= 1 << b.heights[c]
                b.heights[c] += 1
                b.n_moves += 1
        if mark is not None and b.to_move != mark - 1:
            raise ValueError(f"mark={mark} не соответствует очереди хода на доске")
        return b

    @classmethod
    def from_moves(cls, moves: list[int]) -> "Board":
        b = cls()
        for m in moves:
            if not b.is_legal(m):
                raise ValueError(f"нелегальный ход {m}")
            b.play(m)
        return b

    def __str__(self) -> str:
        sym = {0: ".", 1: "X", 2: "O"}
        rows = [" ".join(sym[int(v)] for v in row) for row in self.to_grid()]
        return "\n".join(rows + ["-" * 13, " ".join(str(c) for c in range(COLS))])
