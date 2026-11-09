"""Run the complete baseline in an isolated, immutable experiment directory.

The pipeline never trusts file existence as evidence of a completed training run
and never reuses evaluation caches. Existing experiments remain reviewable.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import yaml

from .utils import dump_json, env_info, load_config


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def effective_config(config: dict, *, experiment: str | None, smoke: bool) -> dict:
    cfg = copy.deepcopy(config)
    if experiment is None and smoke:
        experiment = "smoke_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    if experiment is not None:
        cfg["experiment"] = experiment
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", str(cfg["experiment"])):
        raise ValueError("experiment must be a simple name: letters, digits, '_' or '-' (max 100)")
    if not cfg.get("seeds") or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("seeds must be nonempty and unique")
    if smoke:
        cfg["seeds"] = [0, 1]
        cfg["train"].update(episodes=1000, log_every=500, val_every=500,
                            warmup_transitions=256, buffer_size=5000,
                            batch_size=64, eps_decay_episodes=600)
        for split in ("val", "test"):
            cfg["eval"][split]["games_per_opponent"] = 20
    return cfg


def validate_training(cfg: dict, run_dir: Path) -> dict:
    """Verify exact effective config, final history, and both checkpoint identities."""
    import torch
    from .model import QNet

    verified = {}
    for seed in cfg["seeds"]:
        directory = run_dir / f"seed{seed}"
        needed = [directory / name for name in ("config.json", "history.json", "train_summary.json", "best.pt", "last.pt")]
        missing = [str(p) for p in needed if not p.is_file()]
        if missing:
            raise ValueError(f"Incomplete seed {seed}: missing {', '.join(missing)}")
        saved_cfg = json.loads((directory / "config.json").read_text())
        history = json.loads((directory / "history.json").read_text())
        summary = json.loads((directory / "train_summary.json").read_text())
        if saved_cfg != cfg:
            raise ValueError(f"Config mismatch for seed {seed}; refusing stale checkpoint")
        if not history or history[-1]["episode"] != cfg["train"]["episodes"]:
            raise ValueError(f"Training for seed {seed} did not reach configured episode count")
        if summary["seed"] != seed:
            raise ValueError(f"Summary seed mismatch: {seed}")
        vals = [r for r in history if "val_macro" in r]
        if not vals:
            raise ValueError(f"No validation observations for seed {seed}")
        best_row = max(vals, key=lambda r: r["val_macro"])
        if (summary["best_episode"] != best_row["episode"]
                or abs(summary["best_val_macro"] - best_row["val_macro"]) > 1e-10):
            raise ValueError(f"Best validation checkpoint inconsistent with history: seed {seed}")
        for name, episode in (("best", summary["best_episode"]), ("last", cfg["train"]["episodes"])):
            path = directory / f"{name}.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint.get("seed") != seed or checkpoint.get("episode") != episode:
                raise ValueError(f"Checkpoint identity mismatch: {path}")
            if checkpoint.get("hidden") != cfg["model"]["hidden"] or checkpoint.get("conv", []) != cfg["model"].get("conv", []):
                raise ValueError(f"Checkpoint architecture mismatch: {path}")
            if name == "best" and abs(checkpoint.get("val", -1) - summary["best_val_macro"]) > 1e-10:
                raise ValueError(f"Checkpoint validation score mismatch: {path}")
            net = QNet(cfg["model"]["hidden"], cfg["model"].get("conv", []))
            net.load_state_dict(checkpoint["state_dict"], strict=True)
            if not all(torch.isfinite(v).all() for v in checkpoint["state_dict"].values()):
                raise ValueError(f"Nonfinite weights in {path}")
        verified[str(seed)] = {"best_sha256": file_sha256(directory / "best.pt"),
                               "last_sha256": file_sha256(directory / "last.pt"),
                               "best_val_macro": summary["best_val_macro"]}
    return verified


def check_destinations(cfg: dict, run_dir: Path, result_dir: Path, reuse_trained: bool) -> None:
    if result_dir.exists() and any(result_dir.iterdir()):
        raise FileExistsError(f"Results already exist in {result_dir}. Use --experiment NEW_NAME; nothing was overwritten.")
    if run_dir.exists() and any(run_dir.iterdir()) and not reuse_trained:
        raise FileExistsError(f"Training artifacts already exist in {run_dir}. Use --experiment NEW_NAME; nothing was overwritten.")
    if reuse_trained:
        validate_training(cfg, run_dir)


def run_parallel(jobs: list[tuple[str, list[str], Path]], on_result=None) -> None:
    """Poll every worker; cancel surviving workers as soon as any process fails."""
    processes = []
    files = []
    try:
        for name, cmd, path in jobs:
            log = path.open("w")
            files.append(log)
            print(f"  {name}: log -> {path}", flush=True)
            processes.append((name, cmd, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)))
        pending = set(range(len(processes)))
        while pending:
            for i in list(pending):
                name, cmd, proc = processes[i]
                code = proc.poll()
                if code is None:
                    continue
                pending.remove(i)
                if on_result:
                    on_result(name, cmd, code)
                if code:
                    raise subprocess.CalledProcessError(code, cmd)
            if pending:
                time.sleep(0.1)
    finally:
        for _, _, proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for _, _, proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for f in files:
            f.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", nargs="?", default="configs/baseline.yaml")
    p.add_argument("--experiment", help="new experiment name; never overwrites existing results")
    p.add_argument("--smoke", action="store_true", default=os.environ.get("SMOKE") == "1")
    p.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "8")))
    p.add_argument("--reuse-trained", action="store_true",
                   help="reuse fully validated training runs; results directory must still be empty")
    args = p.parse_args()
    if args.workers < 1:
        p.error("workers must be positive")
    cfg = effective_config(load_config(args.config), experiment=args.experiment, smoke=args.smoke)
    exp = cfg["experiment"]
    run_dir, res_dir = Path("runs") / exp, Path("results") / exp
    # No directories or logs are created before overwrite/reuse checks pass.
    check_destinations(cfg, run_dir, res_dir, args.reuse_trained)
    run_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "effective_config.yaml"
    if config_path.exists():
        if load_config(config_path) != cfg:
            raise ValueError(f"Existing effective config differs: {config_path}")
    else:
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    manifest_path = run_dir / "pipeline_manifest.json"
    if manifest_path.exists():
        manifest_path = run_dir / f"pipeline_manifest_{time.time_ns()}.json"
    manifest = dict(experiment=exp, status="running", started_at_utc=timestamp(), smoke=args.smoke,
                    config=cfg, config_sha256=file_sha256(config_path), source_config=str(Path(args.config)),
                    reused_training=args.reuse_trained, environment=env_info(), steps=[],
                    source_sha256={str(f): file_sha256(f) for f in sorted(Path("src/connectx_rl").glob("*.py"))})
    dump_json(manifest, manifest_path)

    def record(name, cmd, code):
        manifest["steps"].append(dict(name=name, command=cmd, returncode=code, finished_at_utc=timestamp()))
        dump_json(manifest, manifest_path)

    def execute(name, *cmd):
        command = [sys.executable, *map(str, cmd)]
        print(f"== {name}", flush=True)
        code = subprocess.run(command).returncode
        record(name, command, code)
        if code:
            raise subprocess.CalledProcessError(code, command)

    try:
        execute("Tests", "-m", "pytest", "-q", "tests")
        eda_args = ["-m", "connectx_rl.eda", "--config", config_path, "--workers", args.workers,
                    "--out", res_dir / "eda"]
        if args.smoke:
            eda_args += ["--games", 200, "--rr-games", 20]
        execute("EDA", *eda_args)
        if not args.reuse_trained:
            jobs = [(f"train_seed{s}", [sys.executable, "-m", "connectx_rl.train", "--config", str(config_path), "--seed", str(s)],
                     run_dir / f"train_seed{s}.log") for s in cfg["seeds"]]
            print("== Training", flush=True)
            run_parallel(jobs, record)
        manifest["training"] = validate_training(cfg, run_dir)
        dump_json(manifest, manifest_path)

        def evaluate(agent: str, output: Path, opening_plies: int | None = None):
            cmd = ["-m", "connectx_rl.evaluate", "--config", config_path, "--agent", agent,
                   "--split", "test", "--workers", args.workers, "--out", output]
            if opening_plies is not None:
                cmd += ["--opening-plies", opening_plies]
            execute(f"Evaluate {output.name}", *cmd)

        references = res_dir / "reference"
        for agent in ("random", "tactical", "lookahead"):
            evaluate(agent, references / f"test_{agent}.json")
        for seed in cfg["seeds"]:
            ck = run_dir / f"seed{seed}"
            evaluate(f"dqn:{ck / 'best.pt'}", res_dir / f"test_seed{seed}.json")
            evaluate(f"dqn:{ck / 'last.pt'}", res_dir / f"test_last_seed{seed}.json")
            evaluate(f"dqn:{ck / 'best.pt'}", res_dir / f"test_open0_seed{seed}.json", 0)
        execute("Report", "-m", "connectx_rl.report", "--config", config_path, "--references", references)
        best = max(cfg["seeds"], key=lambda s: manifest["training"][str(s)]["best_val_macro"])
        manifest["submission_seed"] = best
        execute("Submission and validation", "-m", "connectx_rl.export_submission", "--ckpt", run_dir / f"seed{best}" / "best.pt",
                "--out", res_dir / "submission" / "main.py")
        manifest["status"] = "complete"
        manifest["output_sha256"] = {str(f): file_sha256(f) for f in sorted(res_dir.rglob("*")) if f.is_file()}
        print(f"Complete. Report: {res_dir / 'report.md'}", flush=True)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["finished_at_utc"] = timestamp()
        dump_json(manifest, manifest_path)


if __name__ == "__main__":
    main()
