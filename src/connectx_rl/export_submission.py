"""Экспорт чекпойнта в самодостаточный Kaggle-сабмишен (submission/main.py).

Файл содержит веса (base64 npz) и numpy-forward, идентичный model.numpy_forward.
После экспорта: (1) проверяем совпадение ходов с DQNAgent на случайных позициях,
(2) прогоняем партии в официальной среде kaggle_environments.
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import json
import hashlib
from pathlib import Path

import numpy as np
import torch

from .agents import RandomAgent
from .board import Board
from .model import DQNAgent, load_qnet, numpy_weights
from .utils import set_torch_threads

TEMPLATE = '''"""ConnectX DQN baseline — автосгенерированный сабмишен (см. src/connectx_rl/export_submission.py)."""
import base64
import io

import numpy as np

_W = "{weights}"
_KINDS = {kinds}
_DATA = np.load(io.BytesIO(base64.b64decode(_W)))
WEIGHTS = [(k, _DATA["w%d" % i], _DATA["b%d" % i]) for i, k in enumerate(_KINDS)]


def _forward(x):
    h = x.astype(np.float32)
    for i, (kind, w, b) in enumerate(WEIGHTS):
        if kind == "conv":
            hp = np.pad(h, ((0, 0), (1, 1), (1, 1)))
            win = np.lib.stride_tricks.sliding_window_view(hp, (3, 3), axis=(1, 2))
            h = np.einsum("chwij,ocij->ohw", win, w, optimize=True) + b[:, None, None]
        else:
            h = w @ h.reshape(-1) + b
        if i < len(WEIGHTS) - 1:
            h = np.maximum(h, 0.0)
    return h


def agent(observation, configuration):
    rows, cols = configuration.rows, configuration.columns
    g = np.asarray(observation.board, dtype=np.int8).reshape(rows, cols)
    me = observation.mark
    x = np.stack([g == me, g == 3 - me]).astype(np.float32)
    q = _forward(x)
    q[g[0] != 0] = -np.inf  # маска нелегальных ходов (заполненные столбцы)
    return int(np.argmax(q))
'''


def export(ckpt: str, out: str) -> Path:
    net = load_qnet(ckpt)
    ws = numpy_weights(net)
    buf = io.BytesIO()
    np.savez_compressed(buf, **{f"w{i}": w for i, (_, w, _) in enumerate(ws)},
                        **{f"b{i}": b for i, (_, _, b) in enumerate(ws)})
    code = TEMPLATE.format(weights=base64.b64encode(buf.getvalue()).decode(), kinds=[k for k, _, _ in ws])
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code)
    return path


def check(ckpt: str, path: Path, n_positions: int = 300, n_games: int = 4) -> dict:
    set_torch_threads(1)
    spec = importlib.util.spec_from_file_location("submission_main", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    net = load_qnet(ckpt)
    ref = DQNAgent(net)
    rng = np.random.default_rng(0)

    class O(dict):
        __getattr__ = dict.__getitem__

    cfg = O(rows=6, columns=7, inarow=4)
    agree = total = 0
    max_q_error = 0.0
    for _ in range(n_positions):  # случайные позиции из random-партий
        b = Board()
        for _ in range(int(rng.integers(0, 30))):
            if b.is_full() or b.winner() is not None:
                break
            b.play(RandomAgent().act(b, rng))
        if b.is_full() or b.winner() is not None:
            continue
        total += 1
        with torch.no_grad():
            q_torch = net(torch.from_numpy(b.planes())[None])[0].numpy()
        q_numpy = ref.q_values(b)
        max_q_error = max(max_q_error, float(np.max(np.abs(q_torch - q_numpy))))
        np.testing.assert_allclose(q_torch, q_numpy, rtol=1e-4, atol=1e-5)
        agree += mod.agent(O(board=b.to_kaggle_board(), mark=b.to_move + 1), cfg) == ref.act(b, rng)
    res = dict(positions=total, agreement=agree / max(1, total),
               max_torch_numpy_q_error=max_q_error,
               checkpoint=str(ckpt), checkpoint_sha256=hashlib.sha256(Path(ckpt).read_bytes()).hexdigest(),
               submission_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               size_kb=round(path.stat().st_size / 1024, 1))
    try:
        from kaggle_environments import evaluate

        rewards = []
        for opp in ("random", "negamax"):
            r = evaluate("connectx", [mod.agent, opp], num_episodes=n_games)
            r += [x[::-1] for x in evaluate("connectx", [opp, mod.agent], num_episodes=n_games)]
            rewards.append(dict(opponent=opp, rewards_agent=[x[0] for x in r]))
        res["kaggle_env_games"] = rewards
        res["kaggle_env_errors"] = sum(x is None for d in rewards for x in d["rewards_agent"])
    except ImportError:
        res["kaggle_env_games"] = "kaggle_environments не установлен — пропущено"
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="submission/main.py")
    p.add_argument("--check-out", default=None, help="JSON validation report (default: next to submission)")
    args = p.parse_args()
    path = export(args.ckpt, args.out)
    res = check(args.ckpt, path)
    print(json.dumps(res, ensure_ascii=False))
    assert res["agreement"] == 1.0, "сабмишен расходится с локальным агентом!"
    assert res.get("kaggle_env_errors") == 0, "Kaggle environment check failed or was skipped"
    check_path = Path(args.check_out) if args.check_out else path.with_suffix(".validation.json")
    check_path.parent.mkdir(parents=True, exist_ok=True)
    check_path.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
