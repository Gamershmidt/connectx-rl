"""Regressions for preserving published runs and detecting incomplete training."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from connectx_rl.model import QNet, save_checkpoint
from connectx_rl.pipeline import check_destinations, effective_config, run_parallel, validate_training
from connectx_rl.utils import dump_json, load_config


def config():
    return load_config(Path(__file__).resolve().parents[1] / "configs/baseline.yaml")


def completed_run(tmp_path):
    cfg = effective_config(config(), experiment="unit_training", smoke=True)
    cfg["seeds"] = [0]
    cfg["model"]["hidden"] = [4]
    run = tmp_path / "runs" / cfg["experiment"]
    seed = run / "seed0"
    seed.mkdir(parents=True)
    dump_json(cfg, seed / "config.json")
    dump_json([dict(episode=500, val_macro=0.6), dict(episode=1000, val_macro=0.5)], seed / "history.json")
    dump_json(dict(seed=0, best_episode=500, best_val_macro=0.6), seed / "train_summary.json")
    net = QNet(cfg["model"]["hidden"])
    save_checkpoint(seed / "best.pt", net, seed=0, episode=500, val=0.6)
    save_checkpoint(seed / "last.pt", net, seed=0, episode=1000)
    return cfg, run


def test_smoke_isolated_without_mutating_baseline():
    original = config()
    snapshot = copy.deepcopy(original)
    smoke = effective_config(original, experiment=None, smoke=True)
    assert original == snapshot
    assert smoke["experiment"].startswith("smoke_")
    assert smoke["experiment"] != original["experiment"]
    assert smoke["seeds"] == [0, 1]
    assert smoke["train"]["episodes"] == 1000
    assert smoke["eval"]["test"]["games_per_opponent"] == 20
    assert smoke["eval"]["test"]["opponents"] == original["eval"]["test"]["opponents"]


@pytest.mark.parametrize("name", ["../baseline_dqn", "/tmp/run", "bad/name", "", "x" * 101])
def test_experiment_cannot_escape_artifact_directories(name):
    with pytest.raises(ValueError, match="simple name"):
        effective_config(config(), experiment=name, smoke=False)


def test_existing_results_protected_even_with_reuse(tmp_path):
    cfg, run = completed_run(tmp_path)
    result = tmp_path / "results" / cfg["experiment"]
    result.mkdir(parents=True)
    artifact = result / "test_seed0.json"
    artifact.write_text("original result")
    with pytest.raises(FileExistsError, match="Results already exist"):
        check_destinations(cfg, run, result, reuse_trained=True)
    assert artifact.read_text() == "original result"


def test_existing_training_requires_explicit_reuse(tmp_path):
    cfg, run = completed_run(tmp_path)
    with pytest.raises(FileExistsError, match="Training artifacts"):
        check_destinations(cfg, run, tmp_path / "results", reuse_trained=False)
    check_destinations(cfg, run, tmp_path / "results", reuse_trained=True)
    assert not (tmp_path / "results").exists()


def test_stale_best_is_not_completed_training(tmp_path):
    cfg, run = completed_run(tmp_path)
    (run / "seed0" / "train_summary.json").unlink()
    with pytest.raises(ValueError, match="Incomplete seed"):
        validate_training(cfg, run)


def test_config_mismatch_rejects_reuse(tmp_path):
    cfg, run = completed_run(tmp_path)
    cfg["train"]["gamma"] = 0.99
    with pytest.raises(ValueError, match="Config mismatch"):
        validate_training(cfg, run)


def test_truncated_history_rejects_reuse(tmp_path):
    cfg, run = completed_run(tmp_path)
    dump_json([dict(episode=500, val_macro=0.6)], run / "seed0" / "history.json")
    with pytest.raises(ValueError, match="episode count"):
        validate_training(cfg, run)


def test_checkpoint_from_wrong_seed_rejected(tmp_path):
    cfg, run = completed_run(tmp_path)
    save_checkpoint(run / "seed0" / "best.pt", QNet([4]), seed=1, episode=500, val=0.6)
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_training(cfg, run)


def test_nonfinite_checkpoint_rejected(tmp_path):
    cfg, run = completed_run(tmp_path)
    net = QNet([4])
    with torch.no_grad():
        next(net.parameters()).fill_(float("nan"))
    save_checkpoint(run / "seed0" / "best.pt", net, seed=0, episode=500, val=0.6)
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_training(cfg, run)


def test_worker_failure_cancels_other_training(tmp_path):
    sentinel = tmp_path / "should_not_finish"
    jobs = [
        ("slow", [sys.executable, "-c", "import sys,time; from pathlib import Path; time.sleep(4); Path(sys.argv[1]).write_text('bad')", str(sentinel)], tmp_path / "slow.log"),
        ("failure", [sys.executable, "-c", "raise SystemExit(7)"], tmp_path / "failure.log"),
    ]
    observed = []
    start = time.monotonic()
    with pytest.raises(subprocess.CalledProcessError) as exc:
        run_parallel(jobs, lambda name, cmd, code: observed.append((name, code)))
    assert exc.value.returncode == 7
    assert time.monotonic() - start < 3
    assert ("failure", 7) in observed
    assert not sentinel.exists()
