"""Q-сеть (MLP) и агент на её основе.

Инференс агента сделан на чистом numpy (те же формулы, что в submission/main.py),
поэтому локальная оценка проверяет ровно то поведение, которое уйдёт на Kaggle.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .agents import Agent
from .board import COLS, ROWS, Board


class QNet(nn.Module):
    """(2,6,7) -> Q(s, a) для 7 столбцов.

    conv: свёрточные слои 3x3 (padding=1, ReLU) -> flatten; затем полносвязные `hidden` -> 7.
    conv=[] даёт обычный MLP.
    """

    def __init__(self, hidden=(256, 256), conv=()) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        ch = 2
        for k in conv:
            layers += [nn.Conv2d(ch, k, 3, padding=1), nn.ReLU()]
            ch = k
        layers.append(nn.Flatten())
        d = ch * ROWS * COLS
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, COLS))
        self.net = nn.Sequential(*layers)
        self.hidden, self.conv = list(hidden), list(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def numpy_weights(net: QNet) -> list[tuple[str, np.ndarray, np.ndarray]]:
    out = []
    for m in net.net:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            kind = "conv" if isinstance(m, nn.Conv2d) else "linear"
            out.append((kind, m.weight.detach().cpu().numpy().astype(np.float32),
                        m.bias.detach().cpu().numpy().astype(np.float32)))
    return out


def numpy_forward(weights, x: np.ndarray) -> np.ndarray:
    """Тот же forward на numpy (используется и в submission/main.py)."""
    h = x.astype(np.float32)
    for i, (kind, w, b) in enumerate(weights):
        if kind == "conv":
            hp = np.pad(h, ((0, 0), (1, 1), (1, 1)))
            win = np.lib.stride_tricks.sliding_window_view(hp, (3, 3), axis=(1, 2))  # (C,6,7,3,3)
            h = np.einsum("chwij,ocij->ohw", win, w, optimize=True) + b[:, None, None]
        else:
            h = w @ h.reshape(-1) + b
        if i < len(weights) - 1:
            h = np.maximum(h, 0.0)
    return h


class DQNAgent(Agent):
    """Жадная политика argmax_a Q(s, a) по ЛЕГАЛЬНЫМ ходам (action masking)."""

    name = "dqn"

    def __init__(self, net: QNet, name: str = "dqn") -> None:
        self.weights = numpy_weights(net)
        self.name = name

    def q_values(self, board: Board) -> np.ndarray:
        return numpy_forward(self.weights, board.planes())

    def act(self, board: Board, rng: np.random.Generator) -> int:
        q = self.q_values(board)
        q[~board.legal_mask()] = -np.inf
        return int(np.argmax(q))

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "DQNAgent":
        net = load_qnet(path)
        return cls(net, name=f"dqn[{Path(path).parent.name}/{Path(path).stem}]")


def save_checkpoint(path: str | Path, net: QNet, **meta) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "hidden": net.hidden, "conv": net.conv, **meta}, path)


def load_qnet(path: str | Path) -> QNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    net = QNet(ckpt["hidden"], ckpt.get("conv", []))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net
