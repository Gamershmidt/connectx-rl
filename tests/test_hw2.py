"""Проверки ДЗ2: плечи отличаются от baseline только заявленными ключами, парные контрасты корректны,
переиспользование чекпойнтов требует идентичного конфига обучения, сбой одного обучения останавливает остальные."""
from __future__ import annotations

import copy
import subprocess
import sys
import time
from pathlib import Path

import pytest

from connectx_rl.evaluate import summarize
from connectx_rl.hw2 import (Layout, apply_dotted, arm_config, check_pairing, contrast, run_bounded, seed_state,
                             strip_test, verdict)
from connectx_rl.report import paired_comparison, result_arrays
from connectx_rl.utils import dump_json, load_config

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/hw2/experiments.yaml"
REGISTRIES = [REGISTRY, ROOT / "configs/hw2_round2/experiments.yaml"]


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


@pytest.mark.parametrize("registry", REGISTRIES, ids=lambda p: p.parent.name)
def test_arms_differ_from_baseline_only_by_registered_overrides(registry):
    reg = load_config(registry)
    base = load_config(ROOT / reg["base_config"])
    for arm in reg["arms"]:
        cfg = arm_config(base, reg, arm, smoke=False)
        diff = {k: v for k, v in flatten(cfg).items() if flatten(base).get(k) != v}
        diff.pop("experiment", None)
        for k in list(diff):
            if k.startswith("eval.test."):
                diff.pop(k)
        assert diff == flatten(dict(arm.get("overrides") or {})), arm["name"]
        assert cfg["eval"]["val"] == base["eval"]["val"]
        assert cfg["eval"]["test"] == reg["test"]
        assert cfg["train"]["episodes"] == base["train"]["episodes"]
        assert cfg["seeds"] == base["seeds"]


@pytest.mark.parametrize("registry", REGISTRIES, ids=lambda p: p.parent.name)
def test_factorial_design_is_complete_and_test_seed_is_fresh(registry):
    reg = load_config(registry)
    base = load_config(ROOT / reg["base_config"])
    lay = Layout(registry, smoke=False)
    assert set(lay.by_factor()) == {(False, False), (True, False), (False, True), (True, True)}
    assert reg["test"]["seed"] not in {base["eval"]["test"]["seed"], base["eval"]["val"]["seed"]}
    names = {a["name"] for a in reg["arms"]}
    for ab in reg["ablations"]:
        assert {ab["full"], ab["ablated"]} <= names
    assert len(reg["ablations"]) >= 2


@pytest.mark.parametrize("registry", REGISTRIES, ids=lambda p: p.parent.name)
def test_materialized_configs_match_registry(registry):
    lay = Layout(registry, smoke=False)
    for arm, cfg in lay.cfgs.items():
        path = lay.cfg_path(arm)
        assert path.is_file(), f"запустите python -m connectx_rl.hw2 --stage configs ({path})"
        assert load_config(path) == cfg


def test_reuse_accepts_new_test_split_but_rejects_other_training(tmp_path):
    reg = load_config(REGISTRY)
    base = load_config(ROOT / reg["base_config"])
    cfg = arm_config(base, reg, reg["arms"][0], smoke=False)
    saved = copy.deepcopy(cfg)
    saved["eval"]["test"] = base["eval"]["test"]  # обучен в ДЗ1 с прежним тестом
    run = tmp_path / "seed0"
    run.mkdir()
    dump_json(saved, run / "config.json")
    for name in ("history.json", "train_summary.json", "best.pt", "last.pt"):
        (run / name).write_text("x")
    assert strip_test(saved) == strip_test(cfg)
    assert seed_state(cfg, tmp_path, 0, reuse=True) == "complete"
    with pytest.raises(ValueError, match="другим конфигом"):
        seed_state(cfg, tmp_path, 0, reuse=False)
    other = copy.deepcopy(cfg)
    other["train"]["gamma"] = 0.99
    with pytest.raises(ValueError, match="другим конфигом"):
        seed_state(other, tmp_path, 0, reuse=True)
    (run / "best.pt").unlink()
    with pytest.raises(ValueError, match="Частичный"):
        seed_state(cfg, tmp_path, 0, reuse=True)
    assert seed_state(cfg, tmp_path, 1, reuse=True) == "absent"


def test_apply_dotted_sets_nested_keys_without_touching_siblings():
    cfg = dict(model=dict(hidden=[256, 256]), train=dict(opponents=["random"], lr=0.1))
    apply_dotted(cfg, {"train.opponents": ["random", "tactical"], "model.conv": [16]})
    assert cfg == dict(model=dict(hidden=[256, 256], conv=[16]), train=dict(opponents=["random", "tactical"], lr=0.1))


def records(values, opponent="random", opening=2):
    return [dict(opponent=opponent, game_id=g, result=float(v), agent_first=g % 2 == 0, opening=opening,
                 moves=[3, 2][:opening], length=10, illegal=None, win_opps=1, missed_wins=0, block_opps=1,
                 missed_blocks=0, safe_opps=1, blunders=0) for g, v in enumerate(values)]


def result(values, **kw):
    rs = records(values, **kw)
    return dict(**summarize(rs, n_boot=50), records=rs, examples=[], split="test", agent="x",
                split_cfg=dict(opponents=["random"], games_per_opponent=len(rs), seed=3000, opening_plies=2))


def test_contrast_matches_paired_comparison_and_is_zero_for_identical_arms():
    a = [result([1, 0, 1, 1, 0, 0.5, 1, 0]), result([1, 1, 1, 0, 0, 0, 1, 0])]
    b = [result([0, 0, 1, 0.5, 0, 0, 1, 1]), result([1, 0, 0, 0, 0.5, 0, 1, 0])]
    arrays = {"a": result_arrays(a, ["random"]), "b": result_arrays(b, ["random"])}
    c = contrast(arrays, {"a": 1, "b": -1}, n_boot=300)
    ref = paired_comparison(a, b, ["random"], n_boot=300)
    assert c["mean"] == pytest.approx(ref["mean"])
    assert c["per_seed"] == pytest.approx(ref["per_seed"])
    assert c["ci95"] == pytest.approx(ref["ci95"])
    same = contrast({"a": arrays["a"], "b": arrays["a"]}, {"a": 1, "b": -1}, n_boot=100)
    assert same["mean"] == 0 and same["ci95"] == [0, 0]
    # взаимодействие (a+b) − a − b + 0 для аддитивных эффектов равно нулю
    inter = contrast({"ab": arrays["a"], "a": arrays["a"], "b": arrays["b"], "0": arrays["b"]},
                     {"ab": 1, "a": -1, "b": -1, "0": 1}, n_boot=100)
    assert inter["mean"] == pytest.approx(0) and inter["ci95"] == pytest.approx([0, 0])


def test_check_pairing_rejects_different_openings():
    arms = {"a": dict(tests=[result([1, 0, 1, 0])]), "b": dict(tests=[result([0, 0, 1, 0])])}
    check_pairing(arms)
    arms["b"]["tests"][0]["records"][0]["moves"][0] = 5
    with pytest.raises(ValueError, match="разные партии"):
        check_pairing(arms)


def test_verdict_requires_ci_above_zero_and_error_reduction():
    assert verdict(dict(ci95=[0.01, 0.05]), 0.8, 0.5, True).startswith("✔")
    assert verdict(dict(ci95=[0.01, 0.05]), 0.5, 0.8, True).startswith("◐")
    assert verdict(dict(ci95=[-0.01, 0.05]), 0.8, 0.5, True).startswith("○")
    assert verdict(dict(ci95=[-0.05, -0.01]), None, None, False).startswith("✘")
    assert verdict(dict(ci95=[0.01, 0.05]), None, None, False).startswith("✔")


def test_run_bounded_limits_parallelism_and_cancels_on_failure(tmp_path):
    marker = tmp_path / "order.txt"
    script = "import sys,time; from pathlib import Path; p=Path(sys.argv[1]); p.open('a').write(sys.argv[2]+' start\\n'); time.sleep(0.6); p.open('a').write(sys.argv[2]+' end\\n')"
    jobs = [(f"j{i}", [sys.executable, "-c", script, str(marker), f"j{i}"], tmp_path / f"j{i}.log") for i in range(2)]
    run_bounded(jobs, max_parallel=1)
    lines = marker.read_text().splitlines()
    assert lines == ["j0 start", "j0 end", "j1 start", "j1 end"]
    sentinel = tmp_path / "late"
    jobs = [("slow", [sys.executable, "-c", "import sys,time; from pathlib import Path; time.sleep(4); Path(sys.argv[1]).write_text('x')", str(sentinel)], tmp_path / "slow.log"),
            ("bad", [sys.executable, "-c", "raise SystemExit(3)"], tmp_path / "bad.log")]
    t = time.monotonic()
    with pytest.raises(subprocess.CalledProcessError) as exc:
        run_bounded(jobs, max_parallel=2)
    assert exc.value.returncode == 3 and time.monotonic() - t < 3
    assert not sentinel.exists()


def test_appendix_markdown_keeps_selected_sections_and_rewrites_paths():
    from connectx_rl.hw2 import APPENDIX_SECTIONS, appendix_markdown
    md = ["# Отчёт", "## Гипотезы", "текст", "## Результаты по запускам", "![a](figures/x.png)",
          "### Подраздел", "см. [summary.json](summary.json)", "## Парные сравнения", "скрыто"]
    out = appendix_markdown(md, "results/hw2")
    assert "Гипотезы" not in out and "скрыто" not in out
    assert "### Результаты по запускам" in out and "#### Подраздел" in out
    assert "(results/hw2/figures/x.png)" in out and "(results/hw2/summary.json)" in out
    assert "Результаты по запускам" in APPENDIX_SECTIONS


def test_homework2_has_live_appendix():
    text = Path("homework2.md").read_text()
    from connectx_rl.hw2 import APPENDIX_END, APPENDIX_START, APPENDIX_SECTIONS
    body = text.split(APPENDIX_START)[1].split(APPENDIX_END)[0]
    for name in APPENDIX_SECTIONS:
        assert f"### {name}" in body


def test_round2_test_seed_differs_from_round1_and_all_earlier_seeds():
    r1, r2 = load_config(REGISTRIES[0]), load_config(REGISTRIES[1])
    base = load_config(ROOT / r1["base_config"])
    used = {base["eval"]["test"]["seed"], base["eval"]["val"]["seed"], r1["test"]["seed"]}
    assert r2["test"]["seed"] not in used
    assert r2["name"] != r1.get("name", "hw2")
