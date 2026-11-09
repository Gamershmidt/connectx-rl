"""Checks for evaluation validity, paired uncertainty, and incomplete-artifact rejection."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from connectx_rl.agents import Agent, RandomAgent
from connectx_rl.evaluate import game_rngs, play_game, summarize, wilson_interval
from connectx_rl.report import checked_evaluation, hier_bootstrap, load, load_inputs, paired_comparison


def records(n=20, opponent="random", values=None, opening=2):
    values = values if values is not None else [float((g // 2) % 2) for g in range(n)]
    return [dict(opponent=opponent, game_id=g, result=float(v), agent_first=g % 2 == 0,
                 opening=opening, moves=[3, 2][:opening], length=10, illegal=None,
                 win_opps=1, missed_wins=0, block_opps=1, missed_blocks=0, safe_opps=1, blunders=0)
            for g, v in enumerate(values)]


def result(rs, split=None, agent="random"):
    return dict(**summarize(rs, n_boot=100), records=rs, examples=[], split="test", agent=agent,
                split_cfg=split or dict(opponents=["random"], games_per_opponent=len(rs), seed=2000, opening_plies=2))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_draw_score_and_side_balance():
    s = summarize(records(4, values=[1, 0, 0.5, 0.5]), n_boot=100)
    assert s["macro_score"] == 0.5
    r = s["per_opponent"]["random"]
    assert (r["win"], r["draw"], r["loss"]) == (0.25, 0.5, 0.25)
    assert (r["score_first"], r["score_second"]) == (0.75, 0.25)


def test_fixed_seat_mix_is_preserved_in_bootstrap():
    # Every first-player game wins and every second-player game loses: the fixed
    # balanced protocol has no game-sampling uncertainty despite different seats.
    s = summarize(records(20, values=[1, 0] * 10))
    assert s["macro_ci95"] == [0.5, 0.5]
    assert s["per_opponent"]["random"]["ci95"] == [0.5, 0.5]


def test_zero_observed_wins_does_not_mean_zero_probability():
    ci = wilson_interval(0, 200)
    assert ci[0] == pytest.approx(0, abs=1e-12)
    assert ci[1] == pytest.approx(0.0188453264, abs=1e-9)
    perfect = wilson_interval(200, 200)
    assert perfect[0] == pytest.approx(1 - ci[1])
    assert perfect[1] == pytest.approx(1)


@pytest.mark.parametrize("values", [[], [1, 0, 1], [1, 0, 2, 0]])
def test_invalid_evaluation_rejected(values):
    with pytest.raises(ValueError):
        summarize(records(len(values), values=values))


def test_identical_paired_agents_have_exactly_zero_difference():
    r = result(records())
    paired = paired_comparison([r] * 3, [copy.deepcopy(r)], ["random"], n_boot=200)
    assert paired["mean"] == 0
    assert paired["ci95"] == [0, 0]


def test_shared_games_are_not_treated_as_independent_across_model_copies():
    r = result(records(200))
    single = hier_bootstrap([r], ["random"], n_boot=1500)
    repeated = hier_bootstrap([r] * 5, ["random"], n_boot=1500)
    # Duplicating a model evaluated on the SAME games supplies no extra evidence.
    assert (repeated[1] - repeated[0]) / (single[1] - single[0]) == pytest.approx(1, rel=0.12)


def test_paired_comparison_uses_draw_adjusted_score():
    left = result(records(4, values=[1, 0.5, 1, 0.5]))
    right = result(records(4, values=[0.5, 0, 0.5, 0]))
    paired = paired_comparison([left], [right], ["random"], n_boot=100)
    assert paired["mean"] == 0.5
    assert paired["ci95"] == [0.5, 0.5]


@pytest.mark.parametrize("mismatch", ["id", "seat", "opening"])
def test_pairing_rejects_different_cases(mismatch):
    left = result(records())
    right = copy.deepcopy(left)
    r = right["records"][0]
    if mismatch == "id":
        r["game_id"] = 1000
    elif mismatch == "seat":
        r["agent_first"] = False
    else:
        r["moves"][0] = 1
    with pytest.raises(ValueError):
        paired_comparison([left], [right], ["random"], n_boot=10)


@pytest.mark.parametrize("damage", ["config", "duplicate", "score", "agent"])
def test_report_rejects_stale_or_corrupt_evaluations(tmp_path, damage):
    r = result(records())
    expected = copy.deepcopy(r["split_cfg"])
    if damage == "config":
        r["split_cfg"]["seed"] += 1
    elif damage == "duplicate":
        r["records"][0]["game_id"] = r["records"][1]["game_id"]
    elif damage == "score":
        r["macro_score"] += 0.1
    else:
        r["agent"] = "tactical"
    p = tmp_path / "test.json"
    write(p, r)
    with pytest.raises(ValueError):
        checked_evaluation(p, expected, "random")


def test_report_recalculation_keeps_raw_evaluation_untouched(tmp_path):
    r = result(records())
    p = tmp_path / "test.json"
    write(p, r)
    before = p.read_bytes()
    updated = checked_evaluation(p, r["split_cfg"], "random")
    assert updated["macro_score"] == r["macro_score"]
    assert p.read_bytes() == before


def test_missing_file_is_not_silently_skipped(tmp_path):
    with pytest.raises(FileNotFoundError, match="required report input"):
        load(tmp_path / "missing.json")


def test_complete_run_required_even_when_main_result_exists(tmp_path):
    cfg = dict(experiment="check", seeds=[0], model=dict(hidden=[4]),
               train=dict(episodes=4, log_every=2),
               eval=dict(test=dict(opponents=["random"], games_per_opponent=4, seed=2000, opening_plies=2)))
    runs, results, refs = (tmp_path / name for name in ("runs", "results", "references"))
    run = runs / "check" / "seed0"
    write(run / "config.json", cfg)
    write(run / "train_summary.json", dict(seed=0, best_episode=2, best_val_macro=0.75))
    write(run / "history.json", [dict(episode=2, val_macro=0.75), dict(episode=4, val_macro=0.5)])
    for name in ("best.pt", "last.pt"):
        (run / name).touch()
    for prefix, ckpt, opening in (("test_seed", "best.pt", 2), ("test_last_seed", "last.pt", 2),
                                   ("test_open0_seed", "best.pt", 0)):
        split = {**cfg["eval"]["test"], "opening_plies": opening}
        write(results / "check" / f"{prefix}0.json", result(records(4, opening=opening), split, f"dqn:{run / ckpt}"))
    for name in ("random", "tactical", "lookahead"):
        write(refs / f"test_{name}.json", result(records(4), cfg["eval"]["test"], name))
    loaded = load_inputs(cfg, runs, results, refs)
    assert len(loaded[0]) == 1
    (results / "check" / "test_last_seed0.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_inputs(cfg, runs, results, refs)


def test_illegal_action_is_a_loss_and_rng_replays_same_game():
    class IllegalAgent(Agent):
        def act(self, board, rng):
            return 7

    illegal = play_game(IllegalAgent(), RandomAgent(), True, game_rngs(0, "random", 0), opening_plies=2)
    assert illegal["illegal"] == "agent" and illegal["result"] == 0
    args = (RandomAgent(), RandomAgent(), True)
    first = play_game(*args, game_rngs(1000, "random", 8), opening_plies=2)
    again = play_game(*args, game_rngs(1000, "random", 8), opening_plies=2)
    assert first == again
