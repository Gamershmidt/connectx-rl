"""Обучение baseline: Double DQN с маскированием нелегальных ходов.

Постановка MDP с точки зрения агента:
  s_t   — позиция перед ходом агента (каноническая перспектива: свои/чужие фишки);
  a_t   — столбец 0..6 (только легальные; маска применяется и в выборе действия, и в max_a' Q);
  r_t   — +1 победа, -1 поражение, 0 ничья/игра продолжается (награда только в конце партии);
  s_t+1 — позиция после хода агента И ответа соперника (соперник — часть среды).

Источник идей: DQN (Mnih et al., 2015), Double DQN (van Hasselt et al., 2016).
Реализация — собственная.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .agents import make_agent
from .board import COLS, ROWS, Board
from .evaluate import game_rngs, play_game, summarize
from .model import DQNAgent, QNet, save_checkpoint
from .utils import dump_json, env_info, load_config, set_seed, set_torch_threads


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.cap = capacity
        self.s = np.zeros((capacity, 2, ROWS, COLS), dtype=np.int8)
        self.s2 = np.zeros((capacity, 2, ROWS, COLS), dtype=np.int8)
        self.mask2 = np.zeros((capacity, COLS), dtype=bool)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.i = 0
        self.size = 0

    def add(self, s, a, r, s2, mask2, done) -> None:
        i = self.i
        self.s[i], self.a[i], self.r[i], self.done[i] = s, a, r, done
        if s2 is not None:
            self.s2[i], self.mask2[i] = s2, mask2
        else:
            self.s2[i], self.mask2[i] = 0, False
        self.i = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, size=n)
        t = torch.from_numpy
        return (t(self.s[idx]).float(), t(self.a[idx]), t(self.r[idx]), t(self.s2[idx]).float(),
                t(self.mask2[idx]), t(self.done[idx]))


def epsilon(ep: int, c: dict) -> float:
    frac = min(1.0, ep / max(1, c["eps_decay_episodes"]))
    return c["eps_start"] + frac * (c["eps_end"] - c["eps_start"])


def select_action(net: QNet, board: Board, eps: float, rng: np.random.Generator) -> int:
    legal = board.legal_moves()
    if rng.random() < eps:
        return legal[int(rng.integers(len(legal)))]
    with torch.no_grad():
        q = net(torch.from_numpy(board.planes())[None])[0].numpy()
    q[~board.legal_mask()] = -np.inf
    return int(np.argmax(q))


def sgd_step(net, target, opt, batch, c) -> float:
    s, a, r, s2, mask2, done = batch
    q = net(s).gather(1, a[:, None]).squeeze(1)
    with torch.no_grad():
        q2_t = target(s2)
        if c["double_dqn"]:
            q2_o = net(s2).masked_fill(~mask2, -1e9)
            a2 = q2_o.argmax(1, keepdim=True)
            q2 = q2_t.gather(1, a2).squeeze(1)
        else:
            q2 = q2_t.masked_fill(~mask2, -1e9).max(1).values
        q2 = torch.where(done > 0, torch.zeros_like(q2), q2)
        y = r + c["gamma"] * q2
        if c.get("target_clip"):  # ценность в игре с нулевой суммой лежит в [-1, 1]
            y = y.clamp(-c["target_clip"], c["target_clip"])
    loss = F.smooth_l1_loss(q, y)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite DQN loss; training aborted")
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), c["grad_clip"])
    opt.step()
    return float(loss.item())


def validate(net: QNet, val_cfg: dict) -> dict:
    agent = DQNAgent(net)
    recs = []
    for opp_spec in val_cfg["opponents"]:
        opp = make_agent(opp_spec)
        for g in range(val_cfg["games_per_opponent"]):
            r = play_game(agent, opp, g % 2 == 0, game_rngs(val_cfg["seed"], opp_spec, g),
                          val_cfg.get("opening_plies", 2), track=False)
            r["opponent"] = opp_spec
            recs.append(r)
    return summarize(recs, n_boot=200)


def train(cfg: dict, seed: int, out_dir: Path) -> dict:
    set_torch_threads(1)
    set_seed(seed)
    c = cfg["train"]
    if (min(c["episodes"], c["val_every"], c["log_every"]) <= 0
            or c["episodes"] < c["val_every"] or c["val_every"] % c["log_every"]):
        raise ValueError("episodes must reach val_every; val_every must be a multiple of log_every")
    if not 0 <= c["gamma"] <= 1 or c["buffer_size"] < c["warmup_transitions"]:
        raise ValueError("Invalid discount or replay buffer/warmup configuration")
    existing = [p.name for p in out_dir.glob("*") if p.suffix in {".pt", ".json"}]
    if existing:
        raise FileExistsError(f"Training output already exists: {out_dir}; choose a new --out/experiment")
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    net = QNet(cfg["model"]["hidden"], cfg["model"].get("conv", []))
    target = copy.deepcopy(net)
    opt = torch.optim.Adam(net.parameters(), lr=c["lr"])
    buf = ReplayBuffer(c["buffer_size"])
    opponents = [make_agent(o) for o in c["opponents"]]
    opp_rng = np.random.default_rng([seed, 7])

    history, window = [], []
    losses: list[float] = []
    n_steps = n_updates = 0
    best_val, best_ep = -1.0, -1
    t0 = time.time()
    for ep in range(1, c["episodes"] + 1):
        eps = epsilon(ep, c)
        opp = opponents[int(rng.integers(len(opponents)))]
        board = Board()
        me = int(rng.integers(2))  # сторона агента: 0 — первый ход, 1 — второй
        if me == 1:
            board.play(opp.act(board, opp_rng))
        outcome = 0.0
        while True:
            s = board.planes().astype(np.int8)
            a = select_action(net, board, eps, rng)
            n_steps += 1
            if board.is_winning_move(a):
                board.play(a)
                buf.add(s, a, 1.0, None, None, 1.0)
                outcome = 1.0
                break
            board.play(a)
            if board.is_full():
                buf.add(s, a, 0.0, None, None, 1.0)
                break
            o = opp.act(board, opp_rng)
            o_wins = board.is_winning_move(o)
            board.play(o)
            if o_wins:
                buf.add(s, a, -1.0, None, None, 1.0)
                outcome = -1.0
                break
            if board.is_full():
                buf.add(s, a, 0.0, None, None, 1.0)
                break
            buf.add(s, a, 0.0, board.planes().astype(np.int8), board.legal_mask(), 0.0)
            if buf.size >= c["warmup_transitions"] and n_steps % c["train_every"] == 0:
                losses.append(sgd_step(net, target, opt, buf.sample(c["batch_size"], rng), c))
                n_updates += 1
                if n_updates % c["target_update_every"] == 0:
                    target.load_state_dict(net.state_dict())
        window.append((outcome, board.n_moves))

        if ep % c["log_every"] == 0:
            outs = np.array([w[0] for w in window])
            row = dict(episode=ep, steps=n_steps, updates=n_updates, eps=round(eps, 4),
                       train_win=float((outs == 1).mean()), train_loss_rate=float((outs == -1).mean()),
                       train_len=float(np.mean([w[1] for w in window])),
                       loss=float(np.mean(losses)) if losses else None, time_s=round(time.time() - t0, 1))
            window, losses = [], []
            if ep % c["val_every"] == 0:
                v = validate(net, cfg["eval"]["val"])
                row["val_macro"] = v["macro_score"]
                row.update({f"val_{k}": m["score"] for k, m in v["per_opponent"].items()})
                if v["macro_score"] > best_val:
                    best_val, best_ep = v["macro_score"], ep
                    save_checkpoint(out_dir / "best.pt", net, episode=ep, val=v["macro_score"], seed=seed)
            history.append(row)
            print(json.dumps(row), flush=True)
    save_checkpoint(out_dir / "last.pt", net, episode=c["episodes"], seed=seed)
    summary = dict(seed=seed, best_val_macro=best_val, best_episode=best_ep,
                   train_time_s=round(time.time() - t0, 1), steps=n_steps, updates=n_updates, env=env_info())
    dump_json(history, out_dir / "history.json")
    dump_json(summary, out_dir / "train_summary.json")
    dump_json(cfg, out_dir / "config.json")
    return summary


def apply_overrides(cfg: dict, items: list[str]) -> None:
    for item in items:
        key, val = item.split("=", 1)
        *path, last = key.split(".")
        d = cfg
        for k in path:
            d = d[k]
        d[last] = yaml.safe_load(val)


def main() -> None:
    p = argparse.ArgumentParser(description="Обучение DQN baseline для ConnectX")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", default=None, help="папка запуска (по умолчанию runs/<experiment>/seed<k>)")
    p.add_argument("--episodes", type=int, default=None, help="переопределить число эпизодов (для smoke-теста)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="переопределить параметры конфига, напр. train.lr=0.0001 model.hidden=[128]")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.episodes is not None:
        cfg["train"]["episodes"] = args.episodes
    apply_overrides(cfg, args.set)
    out = Path(args.out or f"runs/{cfg['experiment']}/seed{args.seed}")
    print(json.dumps(train(cfg, args.seed, out)))


if __name__ == "__main__":
    main()
