"""ДЗ2, раунд 2: признаки угроз и штраф за тактическую ошибку."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from connectx_rl.agents import RandomAgent
from connectx_rl.board import COLS, WINDOWS, Board, bits_to_grid, completing_cells
from connectx_rl.export_submission import check, export
from connectx_rl.model import DQNAgent, QNet, load_qnet, save_checkpoint
from connectx_rl.train import ReplayBuffer, tactical_error, train
from connectx_rl.utils import load_config

ROOT = Path(__file__).resolve().parents[1]


def random_position(rng, max_plies=30):
    b = Board()
    for _ in range(int(rng.integers(0, max_plies))):
        if b.is_full() or b.winner() is not None:
            break
        b.play(RandomAgent().act(b, rng))
    return b


def brute_completing(bb: int, mask: int) -> int:
    out = 0
    for w in WINDOWS:
        if w & mask & ~bb:  # в окне есть чужая фишка
            continue
        missing = w & ~bb
        if missing and missing & (missing - 1) == 0:  # ровно одна пустая клетка
            out |= missing
    return out


@pytest.mark.parametrize("seed", range(40))
def test_completing_cells_matches_brute_force_over_windows(seed):
    rng = np.random.default_rng(seed)
    b = random_position(rng, 36)
    for p in (0, 1):
        assert completing_cells(b.bb[p], b.mask) == brute_completing(b.bb[p], b.mask)


def test_threat_planes_contain_playable_wins_and_hanging_threats():
    rng = np.random.default_rng(1)
    for _ in range(200):
        b = random_position(rng)
        if b.winner() is not None or b.is_full():
            continue
        x = b.planes("threats")
        assert x.shape == (4, 6, 7) and x.dtype == np.float32
        np.testing.assert_array_equal(x[:2], b.planes())
        me = b.to_move
        for ch, player in ((2, me), (3, 1 - me)):
            grid = x[ch]
            playable = {c for c in range(COLS) if b.is_legal(c) and grid[5 - (b.heights[c] - c * 7), c] == 1}
            assert playable == set(b.winning_moves(player))
            assert set(zip(*np.nonzero(grid))) == set(zip(*np.nonzero(bits_to_grid(b.threat_cells(player)))))
    with pytest.raises(ValueError):
        b.planes("unknown")


def test_tactical_error_matches_evaluation_definition():
    # X: 3,2,1 в нижнем ряду; O: 0,5,6. Ходит X, единственная победа — столбец 4.
    b = Board.from_moves([3, 0, 2, 5, 1, 6])
    assert b.to_move == 0 and b.winning_moves() == [4]
    assert tactical_error(b, 4) is None and tactical_error(b, 0) == "missed_win"
    # Ходит O, у X ровно одна угроза (4): всё, кроме 4, — пропуск блока.
    b = Board.from_moves([3, 0, 2, 5, 1])
    assert b.to_move == 1 and b.winning_moves(0) == [4] and not b.winning_moves(1)
    assert tactical_error(b, 4) is None and tactical_error(b, 6) == "missed_block"
    # Две угрозы сразу (0 и 4): закрыть обе нельзя, это не считается пропуском блока.
    b = Board.from_moves([3, 3, 2, 2, 1])
    assert set(b.winning_moves(0)) == {0, 4}
    assert tactical_error(b, 5) is None
    # Угроз нет.
    assert tactical_error(Board.from_moves([3, 3]), 0) is None


def test_shaping_penalises_missed_block_even_when_random_opponent_does_not(tmp_path):
    cfg = load_config(ROOT / "configs/baseline.yaml")
    cfg["model"]["hidden"] = [8]
    cfg["train"].update(episodes=40, val_every=20, log_every=20, buffer_size=5000, warmup_transitions=5000,
                        batch_size=8, shaping={"missed_win": 0.5, "missed_block": 0.5})
    cfg["eval"]["val"]["games_per_opponent"] = 2
    import connectx_rl.train as tr
    captured = []
    orig_add = tr.ReplayBuffer.add

    def spy(self, s, a, r, s2, mask2, done):
        captured.append((float(r), s2 is None))
        orig_add(self, s, a, r, s2, mask2, done)

    tr.ReplayBuffer.add = spy
    try:
        res = train(cfg, 5, tmp_path / "run")
    finally:
        tr.ReplayBuffer.add = orig_add
    rewards = {r for r, _ in captured}
    assert rewards <= {-1.0, -0.5, 0.0, 1.0}
    assert -0.5 in rewards, "штраф ни разу не применился за 40 партий против random"
    assert any(r == -0.5 and not terminal for r, terminal in captured), "штраф должен быть и на нетерминальных переходах"
    hist = __import__("json").load(open(tmp_path / "run/history.json"))
    assert all(0 <= row["train_missed_block_rate"] <= 1 for row in hist if row["train_missed_block_rate"] is not None)


def test_training_without_shaping_keeps_outcome_only_rewards(tmp_path):
    cfg = load_config(ROOT / "configs/baseline.yaml")
    cfg["model"]["hidden"] = [8]
    cfg["train"].update(episodes=20, val_every=10, log_every=10, buffer_size=5000, warmup_transitions=5000, batch_size=8)
    cfg["eval"]["val"]["games_per_opponent"] = 2
    import connectx_rl.train as tr
    captured = []
    orig_add = tr.ReplayBuffer.add
    tr.ReplayBuffer.add = lambda self, s, a, r, s2, m, d: (captured.append(float(r)), orig_add(self, s, a, r, s2, m, d))
    try:
        train(cfg, 5, tmp_path / "run")
    finally:
        tr.ReplayBuffer.add = orig_add
    assert set(captured) <= {-1.0, 0.0, 1.0}


def test_bad_shaping_config_is_rejected(tmp_path):
    cfg = load_config(ROOT / "configs/baseline.yaml")
    cfg["train"].update(episodes=10, val_every=10, log_every=10, shaping={"blunder": 0.5})
    with pytest.raises(ValueError):
        train(cfg, 0, tmp_path / "a")
    cfg["train"]["shaping"] = {"missed_win": -0.1}
    with pytest.raises(ValueError):
        train(cfg, 0, tmp_path / "b")


def test_threat_features_train_checkpoint_and_export_round_trip(tmp_path):
    cfg = load_config(ROOT / "configs/baseline.yaml")
    cfg["model"].update(hidden=[8], features="threats")
    cfg["train"].update(episodes=20, val_every=10, log_every=10, buffer_size=200, warmup_transitions=16, batch_size=8)
    cfg["eval"]["val"]["games_per_opponent"] = 2
    res = train(cfg, 3, tmp_path / "run")
    assert res["updates"] > 0
    net = load_qnet(tmp_path / "run/best.pt")
    assert net.features == "threats" and net.net[1].in_features == 4 * 42
    agent = DQNAgent(net)
    rng = np.random.default_rng(0)
    b = random_position(rng)
    with torch.no_grad():
        q_torch = net(torch.from_numpy(b.planes("threats"))[None])[0].numpy()
    np.testing.assert_allclose(agent.q_values(b), q_torch, rtol=1e-4, atol=1e-5)
    path = export(str(tmp_path / "run/best.pt"), str(tmp_path / "main.py"))
    assert '_FEATURES = "threats"' in path.read_text()
    res = check(str(tmp_path / "run/best.pt"), path, n_positions=150, n_games=1)
    assert res["agreement"] == 1.0


def test_replay_buffer_shape_follows_features():
    assert ReplayBuffer(4, 4).s.shape == (4, 4, 6, 7)
    assert ReplayBuffer(4).s.shape == (4, 2, 6, 7)
