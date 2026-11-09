"""Checks of TD semantics, training reproducibility and deployable inference."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from connectx_rl.board import Board
from connectx_rl.export_submission import export
from connectx_rl.model import DQNAgent, QNet, load_qnet, numpy_forward, numpy_weights, save_checkpoint
from connectx_rl.train import ReplayBuffer, sgd_step, train
from connectx_rl.utils import load_config, set_torch_threads


@pytest.mark.parametrize("conv", [[], [4, 8]])
def test_torch_numpy_and_export_match_on_legal_positions(tmp_path, conv):
    set_torch_threads(1)
    torch.manual_seed(123)
    net = QNet([16], conv)
    ckpt = tmp_path / "model.pt"
    save_checkpoint(ckpt, net)
    path = export(str(ckpt), str(tmp_path / "submission.py"))
    spec = importlib.util.spec_from_file_location("test_submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = DQNAgent(net)
    rng = np.random.default_rng(123)

    class Attr(dict):
        __getattr__ = dict.__getitem__

    positions = [Board(), Board.from_moves([0, 0, 0, 0, 0, 0])]
    for _ in range(20):
        b = Board()
        for _ in range(int(rng.integers(1, 25))):
            c = int(rng.choice(b.legal_moves()))
            if b.is_winning_move(c):
                break
            b.play(c)
        positions.append(b)
    for b in positions:
        with torch.no_grad():
            q = net(torch.from_numpy(b.planes())[None])[0].numpy()
        np.testing.assert_allclose(q, numpy_forward(numpy_weights(net), b.planes()), rtol=1e-4, atol=1e-6)
        q[~b.legal_mask()] = -np.inf
        expected = int(q.argmax())
        assert agent.act(b, rng) == expected
        assert module.agent(Attr(board=b.to_kaggle_board(), mark=b.to_move + 1),
                            Attr(rows=6, columns=7)) == expected
        assert b.is_legal(expected)


class FixedQ(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.q = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, states):
        return self.q.expand(len(states), -1)


@pytest.mark.parametrize("double,done,reward,expected_target", [
    (True, False, 0.0, 0.36),  # online selects col 1; target evaluates it
    (False, False, 0.0, 0.72),  # target selects col 0
    (True, True, -1.0, -1.0),  # terminal loss must not bootstrap
    (True, True, 1.0, 1.0),
])
def test_td_targets_mask_illegal_actions_and_terminal_states(double, done, reward, expected_target):
    online = FixedQ([0.1, 0.2, 100.0, 100.0, 100.0, 100.0, 100.0])
    target = FixedQ([0.8, 0.4, 100.0, 100.0, 100.0, 100.0, 100.0])
    states = torch.zeros(1, 2, 6, 7)
    mask = torch.tensor([[not done, not done, False, False, False, False, False]])
    batch = (states, torch.tensor([0]), torch.tensor([reward]), states, mask, torch.tensor([float(done)]))
    opt = torch.optim.SGD(online.parameters(), lr=0.0)
    loss = sgd_step(online, target, opt, batch,
                    dict(double_dqn=double, gamma=0.9, target_clip=1.0, grad_clip=10.0))
    error = abs(0.1 - expected_target)
    expected_loss = 0.5 * error ** 2 if error < 1.0 else error - 0.5
    assert loss == pytest.approx(expected_loss)


def test_replay_terminal_transition_clears_overwritten_state():
    replay = ReplayBuffer(1)
    s = np.ones((2, 6, 7), dtype=np.int8)
    replay.add(s, 1, 0, s, np.ones(7, dtype=bool), 0)
    replay.add(s, 2, -1, None, None, 1)
    _, action, reward, next_state, mask, done = replay.sample(1, np.random.default_rng(0))
    assert action.item() == 2 and reward.item() == -1 and done.item() == 1
    assert not next_state.any() and not mask.any()


def test_short_training_is_reproducible_and_refuses_overwrite(tmp_path):
    cfg = load_config(Path(__file__).parents[1] / "configs/baseline.yaml")
    cfg["model"]["hidden"] = [8]
    cfg["train"].update(episodes=20, val_every=10, log_every=10, buffer_size=100,
                        warmup_transitions=8, batch_size=8)
    cfg["eval"]["val"]["games_per_opponent"] = 4
    first, second = tmp_path / "one", tmp_path / "two"
    a, b = train(cfg, 23, first), train(cfg, 23, second)
    assert a["updates"] > 0
    assert a["best_val_macro"] == b["best_val_macro"]
    assert a["best_episode"] == b["best_episode"]
    for key, value in load_qnet(first / "last.pt").state_dict().items():
        torch.testing.assert_close(value, load_qnet(second / "last.pt").state_dict()[key], rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        train(cfg, 23, first)
