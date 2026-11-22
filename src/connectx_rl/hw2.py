"""ДЗ2: улучшения baseline, трекинг экспериментов и абляции (факторный план 2x2).

Раунд задаётся реестром (--registry): configs/hw2/experiments.yaml (раунд 1, факторы mix/cnn)
или configs/hw2_round2/experiments.yaml (раунд 2, факторы threats/shaping).

Стадии: configs -> train -> evaluate -> report. Каждая стадия идемпотентна:
готовые и проверенные артефакты не пересчитываются, частичные или не совпадающие
с конфигом артефакты приводят к ошибке (ничего не перезаписывается молча).

Честность сравнения обеспечивается кодом, а не договорённостью:
* все плечи наследуют configs/baseline.yaml, меняются только ключи из реестра;
* валидация (seed 1000) и тест (seed 3000) одинаковы для всех плеч и проверяются
  при загрузке каждого результата (checked_evaluation);
* тест запускается только после завершения ВСЕХ обучений, один раз на best.pt,
  который выбран внутри train.py исключительно по валидации;
* все сравнения парные: одни и те же дебюты и RNG соперника для каждой партии.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from . import plotting
from .evaluate import evaluate_agent
from .model import QNet
from .pipeline import file_sha256, validate_training
from .report import (ERR_KEYS, block_bootstrap, board_ascii, checked_evaluation, hier_bootstrap, load,
                     result_arrays)
from .utils import dump_json, env_info, load_config

ERR_NUM_DEN = {"missed_win_rate": ("missed_wins", "win_opps"), "missed_block_rate": ("missed_blocks", "block_opps"),
               "blunder_rate": ("blunders", "safe_opps")}
ERR_NAMES = {"missed_win_rate": "пропуск победы в 1 ход", "missed_block_rate": "пропуск блока угрозы",
             "blunder_rate": "подстава"}
OPP_NAMES = {"random": "случайный", "tactical": "тактик", "lookahead": "поиск на 3 хода", "kaggle_negamax": "агент Kaggle"}
REFERENCES = ("random", "tactical", "lookahead")


# ----------------------------------------------------------------------------- configs
def apply_dotted(cfg: dict, overrides: dict) -> None:
    for key, val in (overrides or {}).items():
        *path, last = key.split(".")
        d = cfg
        for k in path:
            d = d[k]
        d[last] = copy.deepcopy(val)


def smoke_adjust(cfg: dict) -> None:
    """Те же уменьшения, что в pipeline.effective_config(smoke=True)."""
    cfg["seeds"] = [0, 1]
    cfg["train"].update(episodes=1000, log_every=500, val_every=500, warmup_transitions=256,
                        buffer_size=5000, batch_size=64, eps_decay_episodes=600)
    for split in ("val", "test"):
        cfg["eval"][split]["games_per_opponent"] = 20


def arm_config(base: dict, registry: dict, arm: dict, smoke: bool) -> dict:
    cfg = copy.deepcopy(base)
    cfg["experiment"] = arm["name"]
    apply_dotted(cfg, arm.get("overrides"))
    cfg["eval"]["test"] = copy.deepcopy(registry["test"])
    if smoke:
        smoke_adjust(cfg)
    return cfg


def strip_test(cfg: dict) -> dict:
    """Конфиг без eval.test: всё, что определяет обучение и выбор чекпойнта."""
    c = copy.deepcopy(cfg)
    c["eval"].pop("test", None)
    return c


class Layout:
    """Пути артефактов. Реальный запуск: runs/<arm>, results/<name>/<arm>. Smoke: отдельные корни."""

    def __init__(self, registry_path: Path, smoke: bool):
        self.registry_path = Path(registry_path)
        self.registry = load_config(self.registry_path)
        self.base = load_config(self.registry["base_config"])
        self.smoke = smoke
        self.name = self.registry.get("name", "hw2")
        self.design = {"factors": ["mix", "cnn"], "labels": {}, "interaction": "Взаимодействие факторов",
                       **(self.registry.get("design") or {})}
        if len(self.design["factors"]) != 2:
            raise ValueError("design.factors: ожидаются ровно два фактора (план 2×2)")
        self.report = {"file": "homework2.md", "tag": "", **(self.registry.get("report") or {})}
        tag = f"smoke_{self.name}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") if smoke else None
        self.run_root = Path("runs") / tag if smoke else Path("runs")
        self.res_root = Path("results") / tag if smoke else Path("results") / self.name
        self.cfg_dir = self.run_root / "configs" if smoke else self.registry_path.parent
        self.arms = {a["name"]: a for a in self.registry["arms"]}
        self.cfgs = {n: arm_config(self.base, self.registry, a, smoke) for n, a in self.arms.items()}

    def run_dir(self, arm: str) -> Path:
        reuse = self.arms[arm].get("reuse_runs")
        if reuse and not self.smoke:
            return Path(reuse)
        return self.run_root / arm

    def cfg_path(self, arm: str) -> Path:
        return self.cfg_dir / f"{arm}.yaml"

    def res_dir(self, arm: str) -> Path:
        return self.res_root / arm

    @property
    def test_cfg(self) -> dict:
        return self.cfgs[next(iter(self.cfgs))]["eval"]["test"]

    @property
    def opponents(self) -> list[str]:
        return list(self.test_cfg["opponents"])

    def by_factor(self) -> dict:
        """Плечи факторного плана 2×2: (уровень фактора 1, уровень фактора 2) -> имя плеча."""
        f1, f2 = self.design["factors"]
        return {(a["factors"][f1], a["factors"][f2]): a["name"] for a in self.registry["arms"]
                if isinstance(a["factors"][f1], bool) and isinstance(a["factors"][f2], bool)}


def write_configs(lay: Layout) -> None:
    lay.cfg_dir.mkdir(parents=True, exist_ok=True)
    for arm, cfg in lay.cfgs.items():
        header = (f"# Сгенерировано: python -m connectx_rl.hw2 --stage configs (из {lay.registry_path}).\n"
                  f"# Плечо {arm}: {lay.arms[arm]['label']}\n# Отличия от configs/baseline.yaml: {lay.arms[arm]['change']}\n")
        text = header + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
        path = lay.cfg_path(arm)
        if path.exists() and path.read_text() != text:
            print(f"  configs: {path} обновлён (реестр изменился)")
        path.write_text(text)
    print(f"configs: {len(lay.cfgs)} файлов в {lay.cfg_dir}")


def check_configs_materialized(lay: Layout) -> None:
    for arm, cfg in lay.cfgs.items():
        path = lay.cfg_path(arm)
        if not path.is_file() or load_config(path) != cfg:
            raise ValueError(f"{path} отсутствует или не совпадает с реестром: запустите --stage configs")


# ------------------------------------------------------------------------------ train
def seed_state(cfg: dict, run_dir: Path, seed: int, reuse: bool) -> str:
    """'complete' | 'absent' | исключение для частичного/несовпадающего запуска."""
    d = run_dir / f"seed{seed}"
    if not d.exists() or not any(d.iterdir()):
        return "absent"
    needed = ["config.json", "history.json", "train_summary.json", "best.pt", "last.pt"]
    if not all((d / n).is_file() for n in needed):
        raise ValueError(f"Частичный запуск {d}: удалите папку вручную, чтобы переобучить")
    saved = load(d / "config.json")
    if reuse:
        if strip_test(saved) != strip_test(cfg):
            raise ValueError(f"Переиспользуемый запуск {d} обучен с другим конфигом (кроме eval.test)")
    elif saved != cfg:
        raise ValueError(f"Запуск {d} обучен с другим конфигом; выберите другое имя плеча")
    return "complete"


def validate_arm_training(lay: Layout, arm: str) -> dict:
    """Полная проверка чекпойнтов (как в pipeline), с поправкой на eval.test у переиспользуемого baseline."""
    cfg, run_dir = lay.cfgs[arm], lay.run_dir(arm)
    reuse = bool(lay.arms[arm].get("reuse_runs")) and not lay.smoke
    if reuse:
        saved = load(run_dir / f"seed{cfg['seeds'][0]}" / "config.json")
        if strip_test(saved) != strip_test(cfg):
            raise ValueError(f"{run_dir}: конфиг обучения отличается от реестра")
        return validate_training(saved, run_dir)
    return validate_training(cfg, run_dir)


def run_bounded(jobs: list[tuple[str, list[str], Path]], max_parallel: int, on_done=None) -> None:
    """Запуск подпроцессов не более max_parallel одновременно; любой сбой останавливает остальных."""
    pending, running = list(jobs), []
    try:
        while pending or running:
            while pending and len(running) < max_parallel:
                name, cmd, log_path = pending.pop(0)
                log = log_path.open("w")
                print(f"  start {name}: log -> {log_path}", flush=True)
                running.append((name, cmd, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
            for item in list(running):
                name, cmd, proc, log = item
                code = proc.poll()
                if code is None:
                    continue
                running.remove(item)
                log.close()
                print(f"  done  {name}: code={code} ({datetime.now().strftime('%H:%M:%S')})", flush=True)
                if on_done:
                    on_done(name, cmd, code)
                if code:
                    raise subprocess.CalledProcessError(code, cmd)
            time.sleep(0.5)
    finally:
        for name, cmd, proc, log in running:
            if proc.poll() is None:
                proc.terminate()
        for name, cmd, proc, log in running:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log.close()


def train_stage(lay: Layout, jobs: int) -> None:
    check_configs_materialized(lay)
    todo = []
    for arm, cfg in lay.cfgs.items():
        run_dir = lay.run_dir(arm)
        reuse = bool(lay.arms[arm].get("reuse_runs")) and not lay.smoke
        for seed in cfg["seeds"]:
            state = seed_state(cfg, run_dir, seed, reuse)
            if state == "complete":
                continue
            if reuse:
                raise FileNotFoundError(f"Переиспользуемый запуск отсутствует: {run_dir / f'seed{seed}'}")
            todo.append((arm, seed))
    # Длинные задания (CNN) первыми, чтобы сократить общее время.
    todo.sort(key=lambda t: (not lay.cfgs[t[0]]["model"].get("conv"), t[0], t[1]))
    print(f"train: {len(todo)} запусков обучения, параллельно до {jobs}")
    job_list = []
    for arm, seed in todo:
        run_dir = lay.run_dir(arm)
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "connectx_rl.train", "--config", str(lay.cfg_path(arm)), "--seed", str(seed),
               "--out", str(run_dir / f"seed{seed}")]
        job_list.append((f"{arm}/seed{seed}", cmd, run_dir / f"train_seed{seed}.log"))
    t0 = time.time()
    run_bounded(job_list, jobs)
    print(f"train: готово за {(time.time() - t0) / 60:.1f} мин")
    for arm in lay.cfgs:
        validate_arm_training(lay, arm)
    print("train: все чекпойнты проверены")


# --------------------------------------------------------------------------- evaluate
def evaluate_one(agent_spec: str, split: dict, out: Path, workers: int, expected_agent: str) -> dict:
    if out.is_file():
        return checked_evaluation(out, split, expected_agent)
    res = evaluate_agent(agent_spec, split, workers=workers, keep_records=True)
    res["split"], res["split_cfg"] = "test", split
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"  eval {agent_spec}: macro={res['macro_score']:.4f}", flush=True)
    return checked_evaluation(out, split, expected_agent)


def evaluate_stage(lay: Layout, workers: int) -> None:
    check_configs_materialized(lay)
    for arm in lay.cfgs:  # тест только после завершения всех обучений
        validate_arm_training(lay, arm)
    split = copy.deepcopy(lay.test_cfg)
    lay.res_root.mkdir(parents=True, exist_ok=True)
    for ref in REFERENCES:
        evaluate_one(ref, split, lay.res_root / "reference" / f"test_{ref}.json", workers, ref)
    for arm, cfg in lay.cfgs.items():
        for seed in cfg["seeds"]:
            spec = f"dqn:{lay.run_dir(arm) / f'seed{seed}' / 'best.pt'}"
            evaluate_one(spec, split, lay.res_dir(arm) / f"test_seed{seed}.json", workers, spec)
    print("evaluate: готово")


# ----------------------------------------------------------------------------- report
def load_arm(lay: Layout, arm: str) -> dict:
    cfg, run_dir, split = lay.cfgs[arm], lay.run_dir(arm), lay.test_cfg
    tests, trains, hists = [], [], []
    for seed in cfg["seeds"]:
        d = run_dir / f"seed{seed}"
        spec = f"dqn:{d / 'best.pt'}"
        tests.append(checked_evaluation(lay.res_dir(arm) / f"test_seed{seed}.json", split, spec))
        tr, hist = load(d / "train_summary.json"), load(d / "history.json")
        if tr["seed"] != seed or hist[-1]["episode"] != cfg["train"]["episodes"]:
            raise ValueError(f"неполный запуск {d}")
        best = max((r for r in hist if "val_macro" in r), key=lambda r: r["val_macro"])
        if best["episode"] != tr["best_episode"]:
            raise ValueError(f"best.pt выбран не по валидации: {d}")
        trains.append(tr)
        hists.append(hist)
    n_params = sum(p.numel() for p in QNet(cfg["model"]["hidden"], cfg["model"].get("conv", []), cfg["model"].get("features", "plain")).parameters())
    return dict(name=arm, cfg=cfg, tests=tests, trains=trains, hists=hists, n_params=n_params)


def micro_rate(results: list[dict], key: str, opponents: list[str]) -> float | None:
    num_key, den_key = ERR_NUM_DEN[key]
    num = sum(t["per_opponent"][o][num_key] for t in results for o in opponents)
    den = sum(t["per_opponent"][o][den_key] for t in results for o in opponents)
    return num / den if den else None


def aggregate(a: dict, opponents: list[str]) -> dict:
    tests = a["tests"]
    macro = np.array([t["macro_score"] for t in tests])
    out = dict(name=a["name"], n_seeds=len(tests), n_params=a["n_params"],
               macro_mean=float(macro.mean()), macro_std=float(macro.std(ddof=1)) if len(macro) > 1 else 0.0,
               macro_per_seed=macro.tolist(), macro_ci95=hier_bootstrap(tests, opponents),
               per_opponent={o: dict(mean=float(np.mean([t["per_opponent"][o]["score"] for t in tests])),
                                     std=float(np.std([t["per_opponent"][o]["score"] for t in tests], ddof=1)) if len(tests) > 1 else 0.0,
                                     first=float(np.mean([t["per_opponent"][o]["score_first"] for t in tests])),
                                     second=float(np.mean([t["per_opponent"][o]["score_second"] for t in tests])),
                                     mean_length=float(np.mean([t["per_opponent"][o]["mean_length"] for t in tests])))
                             for o in opponents},
               illegal_rate=float(np.mean([t["per_opponent"][o]["illegal_rate"] for t in tests for o in opponents])),
               val_best_mean=float(np.mean([tr["best_val_macro"] for tr in a["trains"]])),
               val_best_per_seed=[tr["best_val_macro"] for tr in a["trains"]],
               best_episode_mean=float(np.mean([tr["best_episode"] for tr in a["trains"]])),
               train_time_min=float(np.mean([tr["train_time_s"] for tr in a["trains"]]) / 60),
               steps_mean=float(np.mean([tr["steps"] for tr in a["trains"]])),
               updates_mean=float(np.mean([tr["updates"] for tr in a["trains"]])))
    for k, _ in ERR_KEYS:
        out[k] = micro_rate(tests, k, opponents)
        out[k + "_by_opponent"] = {o: micro_rate(tests, k, [o]) for o in opponents}
    return out


def contrast(arrays: dict[str, dict], coefs: dict[str, float], n_boot: int = 2000, seed: int = 0) -> dict:
    """Линейная комбинация плеч, парная по номеру сида и по партии (дебют + RNG соперника).

    coefs={'A': 1, 'B': -1} даёт A минус B. ДИ — перекрёстный бутстреп по сидам и партиям
    (report.block_bootstrap), применённый к поигровым разностям.
    """
    keys = list(next(iter(arrays.values())).keys())
    diffs = {k: sum(c * arrays[a][k] for a, c in coefs.items()) for k in keys}
    by_seed = np.mean([v.mean(axis=1) for v in diffs.values()], axis=0)
    return dict(mean=float(by_seed.mean()), std=float(by_seed.std(ddof=1)) if len(by_seed) > 1 else 0.0,
                per_seed=by_seed.tolist(), ci95=block_bootstrap(diffs, n_boot, seed), coefs=coefs)


def check_pairing(arms: dict[str, dict]) -> None:
    """Все плечи сыграли одни и те же партии: те же ID, стороны и дебюты."""
    ref = None
    for a in arms.values():
        for t in a["tests"]:
            key = {(r["opponent"], r["game_id"]): (r["agent_first"], tuple(r["moves"][:r["opening"]])) for r in t["records"]}
            if ref is None:
                ref = key
            elif key != ref:
                raise ValueError("плечи сыграли разные партии: парное сравнение невозможно")


def fmt(v, digits=3, sign=False) -> str:
    if v is None:
        return "—"
    s = f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}"
    return s.replace(".", ",").replace("-", "−")


def thousands(n) -> str:
    return f"{int(round(n)):,}".replace(",", " ")


def pct(v) -> str:
    return "—" if v is None else f"{100 * v:.1f}%".replace(".", ",")


def ci_str(ci, digits=3, sign=False) -> str:
    return f"[{fmt(ci[0], digits, sign)}; {fmt(ci[1], digits, sign)}]"


def verdict(c: dict, err_before, err_after, need_error: bool) -> str:
    lo, hi = c["ci95"]
    if lo > 0:
        if not need_error:
            return "✔ ДИ > 0"
        if err_before is not None and err_after is not None and err_after < err_before:
            return "✔ ДИ > 0 и ошибка снизилась"
        return "◐ ДИ > 0, но ошибка не снизилась"
    if hi < 0:
        return "✘ значимо хуже"
    return "○ шум: ДИ включает 0"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def report_stage(lay: Layout) -> None:
    check_configs_materialized(lay)
    opps = lay.opponents
    arms = {n: load_arm(lay, n) for n in lay.cfgs}
    check_pairing(arms)
    refs = {r: checked_evaluation(lay.res_root / "reference" / f"test_{r}.json", lay.test_cfg, r) for r in REFERENCES}
    agg = {n: aggregate(a, opps) for n, a in arms.items()}
    arrays = {n: result_arrays(a["tests"], opps) for n, a in arms.items()}
    reg = lay.registry
    res_dir, fig_dir = lay.res_root, lay.res_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- сравнения
    comparisons = {}
    for imp in reg["improvements"]:
        c = contrast(arrays, {imp["new"]: 1, imp["ref"]: -1})
        ek = imp.get("expected_error")
        c.update(kind="improvement", name=imp["name"], left=imp["new"], right=imp["ref"], expected_error=ek,
                 error_before=agg[imp["ref"]][ek] if ek else None, error_after=agg[imp["new"]][ek] if ek else None)
        c["verdict"] = verdict(c, c["error_before"], c["error_after"], bool(ek))
        comparisons[f"imp:{imp['new']}"] = c
    for ab in reg["ablations"]:
        c = contrast(arrays, {ab["full"]: 1, ab["ablated"]: -1})
        c.update(kind="ablation", name=ab["name"], left=ab["full"], right=ab["ablated"], removed=ab["removed"],
                 expected_error=None, error_before=None, error_after=None)
        c["verdict"] = verdict(c, None, None, False)
        comparisons[f"abl:{ab['ablated']}"] = c
    # взаимодействие факторов: (оба − первый) − (второй − baseline)
    by_factor = lay.by_factor()
    if len(by_factor) == 4:
        inter = contrast(arrays, {by_factor[(True, True)]: 1, by_factor[(True, False)]: -1,
                                  by_factor[(False, True)]: -1, by_factor[(False, False)]: 1})
        inter.update(kind="interaction", name=lay.design["interaction"],
                     left="", right="", expected_error=None, error_before=None, error_after=None)
        inter["verdict"] = "○ аддитивно (ДИ включает 0)" if inter["ci95"][0] <= 0 <= inter["ci95"][1] else "✔ взаимодействие есть"
        comparisons["interaction"] = inter
    for name in arms:  # каждое плечо против случайного игрока (санити)
        pass
    ref_arrays = {r: result_arrays([refs[r]], opps) for r in refs}

    # ---------------------------------------------------------------- summary.json
    train_keys = list(dict.fromkeys(k for a in arms.values() for k in a["cfg"]["train"]))  # объединение ключей всех плеч
    summary = dict(test_cfg=lay.test_cfg, seeds=lay.cfgs[next(iter(arms))]["seeds"], arms=agg,
                   references={r: dict(macro=refs[r]["macro_score"], macro_ci95=refs[r]["macro_ci95"],
                                       per_opponent={o: refs[r]["per_opponent"][o]["score"] for o in opps},
                                       **{k: micro_rate([refs[r]], k, opps) for k, _ in ERR_KEYS}) for r in refs},
                   comparisons=comparisons, registry=reg, environment=env_info(),
                   checkpoints={n: {str(s): file_sha256(lay.run_dir(n) / f"seed{s}" / "best.pt") for s in a["cfg"]["seeds"]}
                                for n, a in arms.items()},
                   fairness=dict(
                       val_identical=len({json.dumps(a["cfg"]["eval"]["val"], sort_keys=True) for a in arms.values()}) == 1,
                       test_identical=len({json.dumps(t["split_cfg"], sort_keys=True) for a in arms.values() for t in a["tests"]}) == 1,
                       val_seed=lay.cfgs[next(iter(arms))]["eval"]["val"]["seed"], test_seed=lay.test_cfg["seed"],
                       hw1_test_seed=lay.base["eval"]["test"]["seed"],
                       shared_train_keys=[k for k in train_keys if len({json.dumps(a["cfg"]["train"].get(k)) for a in arms.values()}) == 1],
                       differing_train_keys=[k for k in train_keys if len({json.dumps(a["cfg"]["train"].get(k)) for a in arms.values()}) > 1],
                       checkpoint_rule="best.pt = максимум val_macro по истории обучения (проверено для каждого сида)"))
    dump_json(summary, res_dir / "summary.json")

    # ---------------------------------------------------------------- CSV-трекинг
    exp_rows = []
    for a in reg["arms"]:
        g = agg[a["name"]]
        exp_rows.append(dict(experiment=a["name"], id=a["id"], hypothesis=a["hypothesis"], change=a["change"],
                             **{f: a["factors"][f] for f in lay.design["factors"]}, seeds=" ".join(map(str, lay.cfgs[a["name"]]["seeds"])),
                             n_params=g["n_params"], train_opponents=" ".join(lay.cfgs[a["name"]]["train"]["opponents"]),
                             model_conv=json.dumps(lay.cfgs[a["name"]]["model"].get("conv", [])), model_hidden=json.dumps(lay.cfgs[a["name"]]["model"]["hidden"]),
                             episodes=lay.cfgs[a["name"]]["train"]["episodes"], val_seed=lay.cfgs[a["name"]]["eval"]["val"]["seed"],
                             test_seed=lay.test_cfg["seed"], val_best_mean=round(g["val_best_mean"], 4),
                             test_macro_mean=round(g["macro_mean"], 4), test_macro_std=round(g["macro_std"], 4),
                             test_macro_ci_lo=round(g["macro_ci95"][0], 4), test_macro_ci_hi=round(g["macro_ci95"][1], 4),
                             **{f"test_{o}": round(g["per_opponent"][o]["mean"], 4) for o in opps},
                             **{k: (None if g[k] is None else round(g[k], 4)) for k, _ in ERR_KEYS},
                             train_time_min=round(g["train_time_min"], 1), updates_mean=round(g["updates_mean"])))
    write_csv(res_dir / "experiments.csv", exp_rows)
    run_rows, game_rows = [], []
    for n, a in arms.items():
        for seed, t, tr in zip(a["cfg"]["seeds"], a["tests"], a["trains"]):
            run_rows.append(dict(experiment=n, seed=seed, best_episode=tr["best_episode"], val_best=round(tr["best_val_macro"], 4),
                                 test_macro=round(t["macro_score"], 4), **{f"test_{o}": round(t["per_opponent"][o]["score"], 4) for o in opps},
                                 **{k: (None if micro_rate([t], k, opps) is None else round(micro_rate([t], k, opps), 4)) for k, _ in ERR_KEYS},
                                 steps=tr["steps"], updates=tr["updates"], train_time_s=tr["train_time_s"],
                                 best_sha256=summary["checkpoints"][n][str(seed)][:12]))
            for r in t["records"]:
                game_rows.append(dict(experiment=n, seed=seed, opponent=r["opponent"], game_id=r["game_id"], agent_first=r["agent_first"],
                                      result=r["result"], length=r["length"], missed_wins=r["missed_wins"], win_opps=r["win_opps"],
                                      missed_blocks=r["missed_blocks"], block_opps=r["block_opps"], blunders=r["blunders"],
                                      safe_opps=r["safe_opps"], moves=" ".join(map(str, r["moves"]))))
    write_csv(res_dir / "runs.csv", run_rows)
    write_csv(res_dir / "games.csv", game_rows)
    write_csv(res_dir / "comparisons.csv", [dict(key=k, kind=c["kind"], name=c["name"], left=c["left"], right=c["right"],
                                                 mean=round(c["mean"], 4), std=round(c["std"], 4), ci_lo=round(c["ci95"][0], 4),
                                                 ci_hi=round(c["ci95"][1], 4), per_seed=" ".join(f"{v:+.4f}" for v in c["per_seed"]),
                                                 expected_error=c["expected_error"], error_before=c["error_before"],
                                                 error_after=c["error_after"], verdict=c["verdict"]) for k, c in comparisons.items()])

    # ---------------------------------------------------------------- Markdown
    md = []
    md.append("# ДЗ2: отчёт по экспериментам, абляциям и честному сравнению\n")
    md.append("Отчёт собран автоматически командой `python -m connectx_rl.hw2 --stage report` из сохранённых партий. "
              "Метрика та же, что в ДЗ1: 1 очко за победу, 0,5 за ничью, 0 за поражение, среднее по четырём соперникам (macro score). "
              f"Тест ДЗ2: seed {lay.test_cfg['seed']}, {lay.test_cfg['games_per_opponent']} партий с каждым соперником, "
              f"{lay.test_cfg['opening_plies']} случайных первых хода, половина партий первым ходом. "
              "Знак ± обозначает стандартное отклонение по запускам обучения, в квадратных скобках 95% доверительный интервал "
              "(перекрёстный бутстреп по запускам и партиям, 2000 повторений).\n")

    md.append("## Гипотезы\n")
    for a in reg["arms"]:
        md.append(f"- **{a['id']} ({a['name']})** — {a['label']}. Изменение: {a['change']}. {a['hypothesis']}")
    md.append("")

    md.append("## Таблица экспериментов\n")
    md.append("| Эксперимент | Гипотеза | Что изменено относительно baseline | Сиды | Параметров | Val (лучшая, среднее) | **Test macro** ± std | 95% ДИ | Время обучения, мин/сид | Шагов SGD, тыс. |")
    md.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for a in reg["arms"]:
        g = agg[a["name"]]
        md.append(f"| `{a['name']}` | {a['label']} | {a['change']} | {' '.join(map(str, lay.cfgs[a['name']]['seeds']))} | {thousands(g['n_params'])} | "
                  f"{fmt(g['val_best_mean'])} | **{fmt(g['macro_mean'])} ± {fmt(g['macro_std'])}** | {ci_str(g['macro_ci95'])} | "
                  f"{fmt(g['train_time_min'], 1)} | {g['updates_mean'] / 1000:.0f} |")
    md.append("\nВремя обучения настенное: оно зависит от числа одновременно запущенных процессов и простоя машины, поэтому сравнивать плечи по нему нельзя. Стоимость одного шага SGD измерена отдельно: у CNN она примерно в 9 раз выше, чем у MLP.\n")
    md.append("Полная версия с конфигом каждого плеча: [experiments.csv](experiments.csv). "
              "Каждый запуск отдельно (сид, эпизод лучшего чекпойнта, валидация, тест, ошибки, число шагов): [runs.csv](runs.csv). "
              "Все партии теста (исход, длина, ходы): [games.csv](games.csv).\n")

    md.append("## Тест: результат против каждого соперника\n")
    md.append("| Агент | " + " | ".join(OPP_NAMES[o] for o in opps) + " | **Среднее** |")
    md.append("|---|" + "---:|" * (len(opps) + 1))
    for r, v in summary["references"].items():
        md.append(f"| {OPP_NAMES[r]} (эталон) | " + " | ".join(fmt(v["per_opponent"][o]) for o in opps) + f" | {fmt(v['macro'])} {ci_str(v['macro_ci95'])} |")
    for a in reg["arms"]:
        g = agg[a["name"]]
        md.append(f"| **`{a['name']}`** | " + " | ".join(f"{fmt(g['per_opponent'][o]['mean'])} ± {fmt(g['per_opponent'][o]['std'])}" for o in opps)
                  + f" | **{fmt(g['macro_mean'])} ± {fmt(g['macro_std'])}** {ci_str(g['macro_ci95'])} |")
    md.append("\n![Macro score по экспериментам](figures/macro_by_arm.png)\n")
    md.append("![Результат против каждого соперника](figures/per_opponent.png)\n")

    md.append("## Результаты по запускам\n")
    md.append("| Эксперимент | Сид | Эпизод лучшего чекпойнта | Val | Test macro | " + " | ".join(OPP_NAMES[o] for o in opps) + " | Пропуск блока | Пропуск победы | Обучение, с |")
    md.append("|---|---:|---:|---:|---:|" + "---:|" * len(opps) + "---:|---:|---:|")
    for n, a in arms.items():
        for seed, t, tr in zip(a["cfg"]["seeds"], a["tests"], a["trains"]):
            md.append(f"| `{n}` | {seed} | {thousands(tr['best_episode'])} | {fmt(tr['best_val_macro'])} | {fmt(t['macro_score'])} | "
                      + " | ".join(fmt(t["per_opponent"][o]["score"]) for o in opps)
                      + f" | {pct(micro_rate([t], 'missed_block_rate', opps))} | {pct(micro_rate([t], 'missed_win_rate', opps))} | {tr['train_time_s']:.0f} |")
    md.append("\n![Валидация и тест по запускам](figures/val_vs_test.png)\n")

    md.append("## Парные сравнения: улучшения и абляции\n")
    md.append("Все плечи сыграли одни и те же тестовые партии (одинаковые дебюты и случайность соперника), пары строятся по партии и по номеру сида. "
              "Разность положительна, если левое плечо сильнее. Критерий из ДЗ1: нижняя граница 95% ДИ больше нуля и соответствующий тип ошибок стал реже.\n")
    md.append("| Сравнение | Левое − правое | Разность | 95% ДИ | По сидам | Ошибка: было → стало | Вердикт |")
    md.append("|---|---|---:|---:|---|---|---|")
    for k, c in comparisons.items():
        err = f"{ERR_NAMES[c['expected_error']]}: {pct(c['error_before'])} → {pct(c['error_after'])}" if c["expected_error"] else "—"
        lr = f"`{c['left']}` − `{c['right']}`" if c["left"] else "(a+b) − a − b + 0"
        md.append(f"| {c['name']} | {lr} | {fmt(c['mean'], 3, True)} | {ci_str(c['ci95'], 3, True)} | "
                  f"{', '.join(fmt(v, 3, True) for v in c['per_seed'])} | {err} | {c['verdict']} |")
    md.append("\n![Парные разности](figures/paired_effects.png)\n")
    md.append("![Факторный план 2×2](figures/factorial.png)\n")

    md.append("## Тактические ошибки\n")
    md.append("Доля считается от числа позиций, где такая ошибка была возможна (микро-усреднение по всем сидам и соперникам).\n")
    md.append("| Агент | " + " | ".join(ERR_NAMES[k] for k, _ in ERR_KEYS) + " |")
    md.append("|---|" + "---:|" * len(ERR_KEYS))
    for r, v in summary["references"].items():
        md.append(f"| {OPP_NAMES[r]} (эталон) | " + " | ".join(pct(v[k]) for k, _ in ERR_KEYS) + " |")
    for a in reg["arms"]:
        md.append(f"| **`{a['name']}`** | " + " | ".join(pct(agg[a["name"]][k]) for k, _ in ERR_KEYS) + " |")
    md.append("\n![Тактические ошибки](figures/tactical_errors.png)\n")
    md.append("Пропуск блока по соперникам (строка — эксперимент):\n")
    md.append("| Эксперимент | " + " | ".join(OPP_NAMES[o] for o in opps) + " |")
    md.append("|---|" + "---:|" * len(opps))
    for a in reg["arms"]:
        md.append(f"| `{a['name']}` | " + " | ".join(pct(agg[a["name"]]["missed_block_rate_by_opponent"][o]) for o in opps) + " |")
    md.append("\n![Пропуск блока по соперникам](figures/missed_block_heatmap.png)\n")

    md.append("## Сторона и длина партии\n")
    md.append("| Эксперимент | Соперник | Первым | Вторым | Средняя длина, ходов |")
    md.append("|---|---|---:|---:|---:|")
    for a in reg["arms"]:
        for o in opps:
            p = agg[a["name"]]["per_opponent"][o]
            md.append(f"| `{a['name']}` | {OPP_NAMES[o]} | {fmt(p['first'])} | {fmt(p['second'])} | {fmt(p['mean_length'], 1)} |")
    md.append("")

    md.append("## Кривые обучения\n")
    md.append("Среднее по сидам, полоса — минимум и максимум. Валидация одинакова для всех плеч: 200 партий против random и 200 против tactical, seed 1000. "
              "Доля побед в обучении у плеч со смесью соперников считается против смеси, поэтому с baseline напрямую не сравнима.\n")
    md.append("![Кривые обучения](figures/learning_curves.png)\n")

    md.append("## Примеры ошибок\n")
    best_arm = max(agg, key=lambda n: agg[n]["macro_mean"])
    md.append("Столбцы 0–6 слева направо, точка — пустая клетка.\n")
    md.append(f"### Лучшее плечо: `{best_arm}`\n")
    kind_names = {"missed_win": "Пропущенная победа", "missed_block": "Пропущенный блок", "blunder": "Подстава"}
    kind_rate = {"missed_win": "missed_win_rate", "missed_block": "missed_block_rate", "blunder": "blunder_rate"}

    def example(arm: str, kind: str, prefer: str = "random") -> dict | None:
        found = [e for t in arms[arm]["tests"] for e in t["examples"] if e["kind"] == kind]
        found.sort(key=lambda e: (e["opponent"] != prefer, e["game_id"]))
        return found[0] if found else None

    def show(e: dict) -> None:
        md.append(f"**{kind_names[e['kind']]}.** Соперник «{OPP_NAMES[e['opponent']]}», партия {e['game_id']}.\n```\n"
                  + board_ascii(e["board"], e["move"], e["correct"], e["mark"]) + "\n```\n")

    for kind in kind_names:
        e = example(best_arm, kind)
        if e:
            show(e)

    base = reg["arms"][0]["name"]
    md.append("### Новые плечи: ошибка, которая стала чаще всего\n")
    md.append(f"Для каждого плеча показана ошибка, доля которой выросла сильнее всего относительно `{base}`. Это разбор того, что именно сломалось.\n")
    for a in reg["arms"][1:]:
        n = a["name"]
        deltas = {k: (agg[n][r] or 0) - (agg[base][r] or 0) for k, r in kind_rate.items()}
        kind = max(deltas, key=deltas.get)
        if deltas[kind] <= 0:
            md.append(f"`{n}`: ни один тип ошибок не стал чаще, чем у `{base}`.\n")
            continue
        md.append(f"`{n}`: {ERR_NAMES[kind_rate[kind]]} {pct(agg[base][kind_rate[kind]])} → {pct(agg[n][kind_rate[kind]])}.\n")
        e = example(n, kind)
        if e:
            show(e)

    md.append("## Автоматические проверки честности\n")
    f = summary["fairness"]
    md.append(f"- Конфиг валидации одинаков у всех плеч: **{'да' if f['val_identical'] else 'НЕТ'}** (seed {f['val_seed']}).")
    md.append(f"- Конфиг теста одинаков у всех оценок: **{'да' if f['test_identical'] else 'НЕТ'}** (seed {f['test_seed']}; тест ДЗ1 имел seed {f['hw1_test_seed']}, валидация seed {f['val_seed']}).")
    md.append(f"- Ключи `train`, одинаковые у всех плеч: {', '.join(f'`{k}`' for k in f['shared_train_keys'])}.")
    md.append(f"- Ключи `train`, различающиеся между плечами: {', '.join(f'`{k}`' for k in f['differing_train_keys']) or 'нет'}.")
    md.append(f"- {f['checkpoint_rule']}.")
    md.append("- Одни и те же партии у всех плеч (ID, сторона, дебют): проверено функцией `check_pairing`.")
    md.append("- SHA-256 всех оценённых чекпойнтов записаны в [summary.json](summary.json).\n")
    (res_dir / "report.md").write_text("\n".join(md) + "\n")
    sync_homework_appendix(lay, md)

    # ---------------------------------------------------------------- графики
    make_figures(lay, reg, arms, agg, refs, comparisons, fig_dir)
    print(f"report: {res_dir / 'report.md'}")


APPENDIX_SECTIONS = ["Тест: результат против каждого соперника", "Результаты по запускам", "Тактические ошибки",
                     "Примеры ошибок", "Автоматические проверки честности"]
APPENDIX_START, APPENDIX_END = "<!-- autoreport:start -->", "<!-- autoreport:end -->"


def appendix_markdown(md: list[str], res_rel: str) -> str:
    """Выбранные разделы автоотчёта с путями, переписанными относительно корня репозитория."""
    out, keep = [], False
    for line in "\n".join(md).split("\n"):
        if line.startswith("## "):
            keep = line[3:].strip() in APPENDIX_SECTIONS
            if keep:
                out.append("#" + line)
                continue
        elif line.startswith("### ") and keep:
            out.append("#" + line)
            continue
        if keep:
            out.append(line.replace("(figures/", f"({res_rel}/figures/")
                       .replace("(summary.json)", f"({res_rel}/summary.json)")
                       .replace("(runs.csv)", f"({res_rel}/runs.csv)")
                       .replace("(games.csv)", f"({res_rel}/games.csv)")
                       .replace("(experiments.csv)", f"({res_rel}/experiments.csv)"))
    return "\n".join(out).strip() + "\n"


def appendix_markers(tag: str = "") -> tuple[str, str]:
    t = f":{tag}" if tag else ""
    return f"<!-- autoreport{t}:start -->", f"<!-- autoreport{t}:end -->"


def sync_homework_appendix(lay: Layout, md: list[str], homework: Path | None = None) -> None:
    """Обновляет приложение в корневом отчёте между маркерами, если файл и маркеры есть."""
    homework = Path(lay.report["file"]) if homework is None else homework
    if lay.smoke or not homework.exists():
        return
    start, end = appendix_markers(lay.report["tag"])
    text = homework.read_text()
    if start not in text or end not in text:
        return
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    body = appendix_markdown(md, os.path.relpath(lay.res_root, Path(".")))
    homework.write_text(head + start + "\n" + body + end + tail)
    print(f"report: приложение обновлено в {homework}")


def make_figures(lay: Layout, reg: dict, arms: dict, agg: dict, refs: dict, comparisons: dict, fig_dir: Path) -> None:
    plotting.setup()
    import matplotlib.pyplot as plt

    opps = lay.opponents
    names = [a["name"] for a in reg["arms"]]
    colors = {n: plotting.SERIES[i] for i, n in enumerate(names)}
    labels = {a["name"]: f"{a['id']}\n{a['name']}" for a in reg["arms"]}

    # 1. macro score по плечам: ДИ + точки по сидам + эталоны
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, n in enumerate(names):
        g = agg[n]
        ax.errorbar(i, g["macro_mean"], yerr=[[g["macro_mean"] - g["macro_ci95"][0]], [g["macro_ci95"][1] - g["macro_mean"]]],
                    fmt="o", color=colors[n], capsize=5, markersize=8, zorder=3)
        ax.scatter(np.full(len(g["macro_per_seed"]), i) + np.linspace(-0.12, 0.12, len(g["macro_per_seed"])), g["macro_per_seed"],
                   color=colors[n], alpha=0.5, s=18, zorder=2)
    for r, ls in (("random", ":"), ("tactical", "--")):
        ax.axhline(refs[r]["macro_score"], color=plotting.TEXT2, lw=1, ls=ls)
        ax.text(len(names) - 0.55, refs[r]["macro_score"] + 0.006, f"эталон {OPP_NAMES[r]}", color=plotting.TEXT2, fontsize=8, ha="right")
    ax.set_xticks(range(len(names)), [labels[n] for n in names])
    ax.set_ylabel("test macro score")
    ax.set_title(f"Тест (seed {lay.test_cfg['seed']}): macro score по экспериментам (точки — сиды, усы — 95% ДИ)")
    plotting.save(fig, fig_dir / "macro_by_arm.png")

    # 2. по соперникам
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.8 / len(names)
    xs = np.arange(len(opps))
    for i, n in enumerate(names):
        g = agg[n]
        ax.bar(xs + (i - (len(names) - 1) / 2) * w, [g["per_opponent"][o]["mean"] for o in opps], width=w * 0.9, color=colors[n],
               label=f"{reg['arms'][i]['id']}: {n}", yerr=[g["per_opponent"][o]["std"] for o in opps], capsize=2,
               error_kw=dict(ecolor=plotting.TEXT2, lw=1))
    ax.scatter(xs, [refs["tactical"]["per_opponent"][o]["score"] for o in opps], marker="_", s=400, color=plotting.TEXT, zorder=4, label="эталон тактик")
    ax.set_xticks(xs, [OPP_NAMES[o] for o in opps])
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Тест (seed {lay.test_cfg['seed']}): score против каждого соперника (среднее ± std по сидам)")
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    plotting.save(fig, fig_dir / "per_opponent.png")

    # 3. кривые обучения
    panels = [("val_macro", "val macro (random+tactical)"), ("val_random", "val vs random"), ("val_tactical", "val vs tactical"), ("loss", "Huber loss (среднее за 1000 партий)")]
    if any(r.get("train_missed_win_rate") is not None for n in names for h in arms[n]["hists"] for r in h):
        panels += [("train_missed_win_rate", "пропуск победы в обучении"), ("train_missed_block_rate", "пропуск блока в обучении")]
    ncol = min(4, len(panels))
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow), squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    for ax, (k, title) in zip(axes, panels):
        for n in names:
            curves = []
            for h in arms[n]["hists"]:
                pts = [(r["episode"], r[k]) for r in h if r.get(k) is not None]
                if pts:
                    x, y = zip(*pts)
                    curves.append(y)
            if curves:
                m = np.mean(curves, axis=0)
                ax.plot(x, m, color=colors[n], lw=2, label=n)
                ax.fill_between(x, np.min(curves, axis=0), np.max(curves, axis=0), color=colors[n], alpha=0.15, lw=0)
        ax.set_title(title)
        ax.set_xlabel("партия обучения")
        if k != "loss":
            ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("score")
    axes[0].legend(fontsize=8, loc="lower right")
    plotting.save(fig, fig_dir / "learning_curves.png")

    # 4. тактические ошибки
    fig, ax = plt.subplots(figsize=(8, 4.4))
    series = [(OPP_NAMES["random"] + " (эталон)", refs_err(refs["random"], opps), plotting.TEXT2)] + [(n, agg[n], colors[n]) for n in names]
    xs = np.arange(len(ERR_KEYS))
    w = 0.8 / len(series)
    for i, (label, src, color) in enumerate(series):
        ax.bar(xs + (i - (len(series) - 1) / 2) * w, [src[k] or 0 for k, _ in ERR_KEYS], width=w * 0.9, color=color, label=label)
    ax.set_xticks(xs, [ERR_NAMES[k] for k, _ in ERR_KEYS])
    ax.set_ylabel("доля ошибок")
    ax.set_title(f"Тактические ошибки на тесте (seed {lay.test_cfg['seed']}), чем ниже, тем лучше")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    plotting.save(fig, fig_dir / "tactical_errors.png")

    # 5. heatmap пропуска блока: плечо x соперник
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    m = np.array([[agg[n]["missed_block_rate_by_opponent"][o] or 0 for o in opps] for n in names])
    im = ax.imshow(m, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(opps)), [OPP_NAMES[o].replace(" ", "\n", 1) for o in opps])
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(opps)):
            ax.text(j, i, f"{100 * m[i, j]:.0f}%", ha="center", va="center", color="white" if m[i, j] > 0.55 else plotting.TEXT, fontsize=9)
    ax.set_title("Пропуск блока: плечо × соперник")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    plotting.save(fig, fig_dir / "missed_block_heatmap.png")

    # 6. парные разности (forest plot)
    keys = list(comparisons)
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(keys) + 1.8))
    for i, k in enumerate(keys):
        c = comparisons[k]
        color = {"improvement": plotting.SERIES[0], "ablation": plotting.SERIES[1], "interaction": plotting.SERIES[2]}[c["kind"]]
        ax.errorbar(c["mean"], i, xerr=[[c["mean"] - c["ci95"][0]], [c["ci95"][1] - c["mean"]]], fmt="o", color=color, capsize=4)
        ax.scatter(c["per_seed"], np.full(len(c["per_seed"]), i), color=color, alpha=0.4, s=14)
    ax.axvline(0, color=plotting.TEXT2, lw=1)
    ax.set_yticks(range(len(keys)), [comparisons[k]["name"] for k in keys], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("разность macro score: левое − правое (усы — 95% ДИ, точки — сиды)")
    ax.set_title(f"Парные сравнения на тесте (seed {lay.test_cfg['seed']})")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", color=plotting.SERIES[i], ls="", label=l) for i, l in
                       enumerate(("улучшение vs baseline", "абляция полной модели", "взаимодействие факторов"))], fontsize=8, loc="lower right")
    plotting.save(fig, fig_dir / "paired_effects.png")

    # 7. факторный план 2x2
    by_factor = lay.by_factor()
    if len(by_factor) == 4:
        f1, f2 = lay.design["factors"]
        lab = lay.design["labels"]
        l1 = lab.get(f1, [f"{f1}: нет", f"{f1}: да"])
        l2 = lab.get(f2, [f"{f2}: нет", f"{f2}: да"])
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        for lvl2, label, color in ((False, l2[0], plotting.SERIES[0]), (True, l2[1], plotting.SERIES[1])):
            ys = [agg[by_factor[(lvl1, lvl2)]]["macro_mean"] for lvl1 in (False, True)]
            es = [agg[by_factor[(lvl1, lvl2)]]["macro_std"] for lvl1 in (False, True)]
            ax.errorbar([0, 1], ys, yerr=es, marker="o", color=color, label=label, capsize=4)
        ax.set_xticks([0, 1], l1)
        ax.set_ylabel("test macro score (среднее ± std)")
        ax.set_title("Факторный план 2×2")
        ax.legend()
        plotting.save(fig, fig_dir / "factorial.png")

    # 8. валидация vs тест по запускам
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for n in names:
        ax.scatter(agg[n]["val_best_per_seed"], agg[n]["macro_per_seed"], color=colors[n], label=n, s=40)
    ax.set_xlabel("лучшая валидация (random+tactical, seed 1000)")
    ax.set_ylabel(f"тест (4 соперника, seed {lay.test_cfg['seed']})")
    ax.set_title("Каждая точка — один запуск обучения")
    ax.legend(fontsize=8)
    plotting.save(fig, fig_dir / "val_vs_test.png")


def refs_err(ref: dict, opps: list[str]) -> dict:
    return {k: micro_rate([ref], k, opps) for k, _ in ERR_KEYS}


# ------------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry", default="configs/hw2/experiments.yaml")
    p.add_argument("--stage", default="all", choices=["all", "configs", "train", "evaluate", "report"])
    p.add_argument("--smoke", action="store_true", default=os.environ.get("SMOKE") == "1",
                   help="уменьшенный прогон всех стадий в отдельных папках runs/smoke_<name>_*, results/smoke_<name>_*")
    p.add_argument("--jobs", type=int, default=int(os.environ.get("JOBS", os.cpu_count() or 4)), help="параллельных обучений")
    p.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "8")), help="процессов на оценку")
    args = p.parse_args()
    lay = Layout(Path(args.registry), args.smoke)
    stages = ["configs", "train", "evaluate", "report"] if args.stage == "all" else [args.stage]
    if args.smoke and args.stage != "all":
        p.error("--smoke выполняет все стадии сразу (--stage all)")
    for stage in stages:
        print(f"== {stage} ({datetime.now().strftime('%H:%M:%S')})", flush=True)
        {"configs": lambda: write_configs(lay), "train": lambda: train_stage(lay, args.jobs),
         "evaluate": lambda: evaluate_stage(lay, args.workers), "report": lambda: report_stage(lay)}[stage]()


if __name__ == "__main__":
    main()
