"""Протокол оценки: серия партий агента против пула соперников.

Ключевые свойства протокола:
* баланс сторон — ровно половина партий агент ходит первым (чётный game_id), половина вторым;
* случайный дебют — первые `opening_plies` полуходов делаются случайно (одинаково для всех
  оцениваемых агентов), иначе детерминированный агент против детерминированного соперника
  сыграл бы N одинаковых партий и доверительный интервал был бы фиктивным;
* common random numbers — seed каждой партии зависит только от (seed сплита, соперник, game_id),
  поэтому разные агенты играют ОДИНАКОВЫЕ дебюты против одинаково «настроенных» соперников ->
  сравнение агентов парное;
* независимые потоки случайности для дебюта, соперника и агента;
* нелегальный ход = поражение (как в правилах Kaggle).

Метрика партии: score = 1 (победа) / 0.5 (ничья) / 0 (поражение) с точки зрения агента.
"""
from __future__ import annotations

import argparse
import json
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import NormalDist

import numpy as np

from .agents import Agent, make_agent, safe_moves
from .board import Board
from .utils import load_config, set_torch_threads


def game_rngs(seed: int, opponent: str, game_id: int):
    """Три независимых генератора (дебют, соперник, агент) для конкретной партии."""
    ss = np.random.SeedSequence([seed, zlib.crc32(opponent.encode()), game_id])
    return [np.random.default_rng(s) for s in ss.spawn(3)]


def play_game(
    agent: Agent,
    opponent: Agent,
    agent_first: bool,
    rngs,
    opening_plies: int = 0,
    track: bool = True,
    max_examples: int = 2,
) -> dict:
    open_rng, opp_rng, agent_rng = rngs
    agent.reset()
    opponent.reset()
    board = Board()
    history: list[int] = []
    for _ in range(opening_plies):  # дебют: случайные ходы, не завершающие партию
        moves = [c for c in board.legal_moves() if not board.is_winning_move(c)]
        history.append(moves[int(open_rng.integers(len(moves)))])
        board.play(history[-1])
    opening = board.n_moves
    me = 0 if agent_first else 1
    st = dict(win_opps=0, missed_wins=0, block_opps=0, missed_blocks=0, safe_opps=0, blunders=0)
    examples: list[dict] = []
    illegal = None
    result = 0.5
    while True:
        player = board.to_move
        if player == me:
            col = agent.act(board, agent_rng)
            if not board.is_legal(col):
                illegal, result = "agent", 0.0
                break
            if track:
                _track(board, col, st, examples, history, max_examples)
        else:
            col = opponent.act(board, opp_rng)
            if not board.is_legal(col):
                illegal, result = "opponent", 1.0
                break
        won = board.is_winning_move(col)
        board.play(col)
        history.append(col)
        if won:
            result = 1.0 if player == me else 0.0
            break
        if board.is_full():
            result = 0.5
            break
    return dict(
        result=result,
        agent_first=agent_first,
        length=board.n_moves,
        opening=opening,
        illegal=illegal,
        moves=history,
        examples=examples,
        **st,
    )


def _track(board: Board, col: int, st: dict, examples: list, history: list, max_examples: int) -> None:
    """Классификация тактических ошибок агента в текущей позиции (до хода)."""
    me = board.to_move
    wins = board.winning_moves(me)
    kind, correct = None, None
    if wins:
        st["win_opps"] += 1
        if col not in wins:
            st["missed_wins"] += 1
            kind, correct = "missed_win", wins
    else:
        threats = board.winning_moves(1 - me)
        if len(threats) == 1:  # блокируемая угроза
            st["block_opps"] += 1
            if col not in threats:
                st["missed_blocks"] += 1
                kind, correct = "missed_block", threats
        elif not threats:
            safe = safe_moves(board)
            if safe and len(safe) < len(board.legal_moves()):  # есть шанс «подставиться»
                st["safe_opps"] += 1
                if col not in safe:
                    st["blunders"] += 1
                    kind, correct = "blunder", safe
    if kind and len(examples) < max_examples:
        examples.append(dict(kind=kind, board=board.to_kaggle_board(), mark=me + 1, move=col, correct=correct))


# ----------------------------------------------------------------- parallel runner
def _worker(args) -> list[dict]:
    agent_spec, opp_spec, game_ids, seed, opening_plies = args
    set_torch_threads(1)
    agent, opponent = make_agent(agent_spec), make_agent(opp_spec)
    out = []
    for g in game_ids:
        rec = play_game(agent, opponent, agent_first=(g % 2 == 0), rngs=game_rngs(seed, opp_spec, g),
                        opening_plies=opening_plies)
        rec.update(opponent=opp_spec, game_id=g)
        out.append(rec)
    return out


def run_matches(
    agent_spec: str,
    opponents: list[str],
    games_per_opponent: int,
    seed: int,
    opening_plies: int = 2,
    workers: int = 1,
    chunk: int = 20,
) -> list[dict]:
    assert games_per_opponent % 2 == 0, "нужно чётное число партий для баланса сторон"
    tasks = [
        (agent_spec, opp, list(range(i, min(i + chunk, games_per_opponent))), seed, opening_plies)
        for opp in opponents
        for i in range(0, games_per_opponent, chunk)
    ]
    if workers <= 1:
        results = [_worker(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_worker, tasks))
    return [r for chunk_res in results for r in chunk_res]


# ------------------------------------------------------------------------ metrics
def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def wilson_interval(wins: int, n: int, alpha: float = 0.05) -> list[float]:
    """Wilson interval for a binomial win probability; nonzero width at 0/n or n/n.

    Applied separately to each side for a fixed trained policy and opponent.
    It is not a confidence interval for the draw-adjusted score or for unseen opponents.
    """
    if n <= 0 or not 0 <= wins <= n:
        raise ValueError("Wilson interval requires 0 <= wins <= n and n > 0")
    z = NormalDist().inv_cdf(1 - alpha / 2)
    p, z2 = wins / n, z * z
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    radius = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return [float(max(0, center - radius)), float(min(1, center + radius))]


def stratified_score_samples(scores: np.ndarray, first: np.ndarray, n_boot: int,
                             rng: np.random.Generator) -> np.ndarray:
    """Resample games within each seat, retaining the evaluation's exact 50/50 balance."""
    groups = [scores[first], scores[~first]]
    if not len(groups[0]) or len(groups[0]) != len(groups[1]):
        raise ValueError("evaluation requires an equal nonzero number of games in each seat")
    return np.mean([g[rng.integers(0, len(g), size=(n_boot, len(g)))].mean(axis=1)
                    for g in groups], axis=0)


def _rate(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def summarize(records: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """Score unchanged; bootstrap samples retain the fixed opponent/seat proportions."""
    if not records:
        raise ValueError("cannot summarize an empty evaluation")
    by_opp: dict[str, list[dict]] = {}
    for r in records:
        by_opp.setdefault(r["opponent"], []).append(r)
    per_opp = {}
    score_samples = []
    rng = np.random.default_rng(seed)
    for opp, rs in by_opp.items():
        s = np.array([r["result"] for r in rs], dtype=float)
        if not np.isin(s, [0.0, 0.5, 1.0]).all():
            raise ValueError("game results must be 0, 0.5, or 1")
        first = np.array([r["agent_first"] for r in rs], dtype=bool)
        tot = {k: sum(r[k] for r in rs) for k in ("win_opps", "missed_wins", "block_opps", "missed_blocks",
                                                   "safe_opps", "blunders")}
        samples = stratified_score_samples(s, first, n_boot, rng)
        lo, hi = np.quantile(samples, [0.025, 0.975]).tolist()
        per_opp[opp] = dict(
            n=len(s),
            score=float(s.mean()),
            ci95=[lo, hi],
            win=float((s == 1).mean()),
            draw=float((s == 0.5).mean()),
            loss=float((s == 0).mean()),
            score_first=float(s[first].mean()),
            score_second=float(s[~first].mean()),
            win_ci95_by_side={label: wilson_interval(int((s[mask] == 1).sum()), int(mask.sum()))
                              for label, mask in (("first", first), ("second", ~first))},
            mean_length=float(np.mean([r["length"] for r in rs])),
            illegal_rate=float(np.mean([r["illegal"] == "agent" for r in rs])),
            missed_win_rate=_rate(tot["missed_wins"], tot["win_opps"]),
            missed_block_rate=_rate(tot["missed_blocks"], tot["block_opps"]),
            blunder_rate=_rate(tot["blunders"], tot["safe_opps"]),
            **tot,
        )
        score_samples.append(samples)
    boots = np.mean(score_samples, axis=0)
    macro = float(np.mean([v["score"] for v in per_opp.values()]))
    return dict(
        macro_score=macro,
        macro_ci95=[float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        per_opponent=per_opp,
    )


def evaluate_agent(agent_spec: str, split_cfg: dict, workers: int = 1, keep_records: bool = False) -> dict:
    recs = run_matches(
        agent_spec,
        split_cfg["opponents"],
        split_cfg["games_per_opponent"],
        split_cfg["seed"],
        split_cfg.get("opening_plies", 2),
        workers=workers,
    )
    summ = summarize(recs)
    summ["agent"] = agent_spec
    summ["examples"] = [dict(opponent=r["opponent"], game_id=r["game_id"], **e) for r in recs for e in r["examples"]]
    if keep_records:
        summ["records"] = [{k: v for k, v in r.items() if k != "examples"} for r in recs]
    return summ


def main() -> None:
    p = argparse.ArgumentParser(description="Оценка агента на val/test пуле соперников")
    p.add_argument("--config", required=True)
    p.add_argument("--agent", required=True, help="random | tactical | lookahead | kaggle_negamax | dqn:<ckpt>")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--opening-plies", type=int, default=None, help="переопределить длину случайного дебюта")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    split = dict(cfg["eval"][args.split])
    if args.opening_plies is not None:
        split["opening_plies"] = args.opening_plies
    res = evaluate_agent(args.agent, split, workers=args.workers, keep_records=True)
    res["split"] = args.split
    res["split_cfg"] = split
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"[{args.split}] {args.agent}: macro score = {res['macro_score']:.3f} "
          f"(95% CI {res['macro_ci95'][0]:.3f}..{res['macro_ci95'][1]:.3f})")
    for opp, m in res["per_opponent"].items():
        print(f"  vs {opp:15s} score={m['score']:.3f} [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}] "
              f"W/D/L={m['win']:.2f}/{m['draw']:.2f}/{m['loss']:.2f} first={m['score_first']:.2f} "
              f"second={m['score_second']:.2f}")


if __name__ == "__main__":
    main()
