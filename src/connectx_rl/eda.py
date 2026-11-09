"""EDA для RL-задачи: «данных» нет, они порождаются средой. Изучаем свойства среды и
сложность пула соперников:

1. random vs random: преимущество первого хода, длина партии, доля ничьих, разреженность награды;
2. tactical vs tactical: то же для «осмысленной» игры;
3. турнир эталонных агентов (каждый с каждым) — калибровка сложности train/val/test соперников.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from . import plotting
from .agents import make_agent
from .board import Board
from .evaluate import game_rngs, play_game, run_matches, summarize
from .utils import dump_json, load_config


def self_play_stats(spec: str, n: int, seed: int) -> dict:
    a, b = make_agent(spec), make_agent(spec)
    res, lengths, branching, first_moves = [], [], [], Counter()
    for g in range(n):
        rngs = game_rngs(seed, spec, g)
        r = play_game(a, b, True, rngs, opening_plies=0, track=False)
        res.append(r["result"])  # с точки зрения первого игрока
        lengths.append(r["length"])
        first_moves[r["moves"][0]] += 1
        bd = Board()
        for m in r["moves"]:
            branching.append(len(bd.legal_moves()))
            bd.play(m)
    res = np.array(res)
    lengths = np.array(lengths)
    # From P1's perspective there are ceil(length/2) transitions per game.
    # Each decisive game has one nonzero terminal reward; draws have none.
    # Pool transitions, rather than averaging reciprocal game lengths.
    transitions = int(np.ceil(lengths / 2).sum())
    nonzero_rewards = int((res != 0.5).sum())
    nonzero_frac = nonzero_rewards / transitions
    return dict(
        games=n,
        p1_win=float((res == 1).mean()), p2_win=float((res == 0).mean()), draw=float((res == 0.5).mean()),
        length_mean=float(lengths.mean()), length_median=float(np.median(lengths)),
        length_min=int(lengths.min()), length_max=int(lengths.max()),
        branching_mean=float(np.mean(branching)),
        nonzero_reward_transition_frac=nonzero_frac,
        p1_transitions=transitions,
        nonzero_terminal_rewards=nonzero_rewards,
        first_move_counts=dict(sorted(first_moves.items())),
        lengths=lengths.tolist(),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--out", default="results/eda")
    p.add_argument("--games", type=int, default=20000)
    p.add_argument("--rr-games", type=int, default=200, help="партий на пару в турнире эталонов")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stats = {s: self_play_stats(s, args.games if s == "random" else args.games // 4, seed=42)
             for s in ("random", "tactical")}
    for s, st in stats.items():
        print(f"{s} vs {s}: P1={st['p1_win']:.3f} P2={st['p2_win']:.3f} D={st['draw']:.3f} "
              f"len={st['length_mean']:.1f} branching={st['branching_mean']:.2f} "
              f"nonzero_r={st['nonzero_reward_transition_frac']:.3f}")

    # турнир эталонов: строка = агент, столбец = соперник, значение = score агента
    pool = list(dict.fromkeys(cfg["eval"]["test"]["opponents"]))
    matrix = {}
    for a in pool:
        recs = run_matches(a, pool, args.rr_games, seed=7, opening_plies=2, workers=args.workers)
        sm = summarize(recs, n_boot=500)
        matrix[a] = {o: dict(score=m["score"], ci95=m["ci95"]) for o, m in sm["per_opponent"].items()}
        print(a, {o: round(v["score"], 3) for o, v in matrix[a].items()})
    dump_json(dict(protocol=dict(self_play_seed=42, round_robin_seed=7, opening_plies=2,
                                random_games=args.games, tactical_games=args.games // 4,
                                opponents=pool),
                   self_play={k: {kk: vv for kk, vv in v.items() if kk != "lengths"} for k, v in stats.items()},
                   reference_round_robin=matrix, rr_games_per_pair=args.rr_games), out / "eda.json")

    # ---- графики
    plotting.setup()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.arange(6.5, 43.5, 1)
    for i, s in enumerate(stats):
        ax.hist(stats[s]["lengths"], bins=bins, density=True, histtype="step", linewidth=2,
                color=plotting.SERIES[i], label=f"{s} vs {s}")
    ax.set_xlabel("длина партии, полуходов")
    ax.set_ylabel("доля партий")
    ax.set_title("Распределение длины партии")
    ax.legend()
    plotting.save(fig, out / "game_length.png")

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    m = np.array([[matrix[a][o]["score"] for o in pool] for a in pool])
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(pool)), pool, rotation=30, ha="right")
    ax.set_yticks(range(len(pool)), pool)
    ax.grid(False)
    for i in range(len(pool)):
        for j in range(len(pool)):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                    color="white" if m[i, j] > 0.6 else plotting.TEXT, fontsize=9)
    ax.set_xlabel("соперник")
    ax.set_ylabel("агент")
    ax.set_title("Турнир эталонов: score агента (строка)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plotting.save(fig, out / "reference_round_robin.png")


if __name__ == "__main__":
    main()
