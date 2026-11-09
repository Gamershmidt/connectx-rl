"""Агрегация результатов baseline по сидам: таблицы, бутстрап-ДИ, графики, примеры ошибок."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import plotting
from .board import Board
from .evaluate import summarize
from .utils import dump_json, load_config

ERR_KEYS = [("missed_win_rate", "пропуск победы в 1 ход"), ("missed_block_rate", "пропуск блока угрозы"),
            ("blunder_rate", "подстава (ход под угрозу)")]


def load(path: Path):
    """A report is an all-or-nothing artifact: never silently omit an absent seed."""
    if not path.is_file():
        raise FileNotFoundError(f"missing required report input: {path}")
    return json.loads(path.read_text())


def result_arrays(results: list[dict], opponents: list[str]) -> dict:
    """Align shared evaluation game IDs across training seeds in each opponent/seat stratum."""
    if not results:
        raise ValueError("at least one seed result is required")
    arrays = {}
    for opp in opponents:
        for first in (True, False):
            maps = [{r["game_id"]: r["result"] for r in res["records"]
                     if r["opponent"] == opp and r["agent_first"] == first} for res in results]
            ids = sorted(maps[0])
            if not ids or any(sorted(m) != ids for m in maps):
                raise ValueError(f"unaligned game IDs for {opp}, agent_first={first}")
            arrays[(opp, first)] = np.array([[m[i] for i in ids] for m in maps], dtype=float)
    return arrays


def block_bootstrap(arrays: dict, n_boot: int = 2000, seed: int = 0) -> list[float]:
    """Crossed resampling: training seeds plus shared game IDs, stratified by opponent/seat.

    One game-ID sample is applied to every sampled model in a stratum. This preserves
    the correlation induced by testing all models on the same random openings/RNG seeds.
    For paired comparisons, arrays already contain within-game score differences.
    """
    rng = np.random.default_rng(seed)
    n_seeds = next(iter(arrays.values())).shape[0]
    out = np.empty(n_boot)
    for b in range(n_boot):
        model_ids = rng.integers(0, n_seeds, n_seeds)
        vals = [a[np.ix_(model_ids, rng.integers(0, a.shape[1], a.shape[1]))].mean()
                for a in arrays.values()]
        out[b] = np.mean(vals)
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def hier_bootstrap(seed_results: list[dict], opponents: list[str], n_boot: int = 2000, seed: int = 0):
    return block_bootstrap(result_arrays(seed_results, opponents), n_boot, seed)


def paired_comparison(left: list[dict], right: list[dict], opponents: list[str],
                      n_boot: int = 2000, seed: int = 0, same_openings: bool = True) -> dict:
    """Mean left-minus-right difference, paired on training seed and evaluation game ID.

    A single right-hand reference policy is broadcast across trained left-hand policies;
    it is never treated as five independently evaluated reference agents.
    """
    if len(right) not in (1, len(left)):
        raise ValueError("paired comparison requires matching model seeds or one reference")
    for i, lres in enumerate(left):
        rres = right[0 if len(right) == 1 else i]
        lm = {(r["opponent"], r["game_id"]): r for r in lres["records"]}
        rm = {(r["opponent"], r["game_id"]): r for r in rres["records"]}
        if lm.keys() != rm.keys():
            raise ValueError("paired comparison has different evaluation game IDs")
        for key, l in lm.items():
            r = rm[key]
            if l["agent_first"] != r["agent_first"]:
                raise ValueError("paired comparison has mismatched seats")
            if same_openings and (l["opening"] != r["opening"]
                                  or l["moves"][:l["opening"]] != r["moves"][:r["opening"]]):
                raise ValueError("paired comparison has mismatched openings")
    la, ra = result_arrays(left, opponents), result_arrays(right, opponents)
    diffs = {k: la[k] - ra[k] for k in la}
    by_seed = np.mean([v.mean(axis=1) for v in diffs.values()], axis=0)
    return dict(mean=float(by_seed.mean()), std=float(by_seed.std(ddof=1)) if len(by_seed) > 1 else 0.0,
                per_seed=by_seed.tolist(), ci95=block_bootstrap(diffs, n_boot, seed),
                same_openings=same_openings)


def checked_evaluation(path: Path, split_cfg: dict, expected_agent: str | None = None) -> dict:
    res = load(path)
    if res.get("split") != "test" or res.get("split_cfg") != split_cfg:
        raise ValueError(f"evaluation/config mismatch: {path}")
    if expected_agent is not None:
        actual = res.get("agent", "")
        if expected_agent.startswith("dqn:"):
            matches = actual.startswith("dqn:") and Path(actual[4:]).resolve() == Path(expected_agent[4:]).resolve()
        else:
            matches = actual == expected_agent
        if not matches:
            raise ValueError(f"wrong evaluated agent in {path}: {actual}")
    records = res.get("records", [])
    opponents = split_cfg["opponents"]
    n = split_cfg["games_per_opponent"]
    if len(records) != len(opponents) * n or n <= 0 or n % 2:
        raise ValueError(f"incomplete evaluation: {path}")
    if {r["opponent"] for r in records} != set(opponents):
        raise ValueError(f"wrong opponent pool: {path}")
    for opp in opponents:
        rs = [r for r in records if r["opponent"] == opp]
        if sorted(r["game_id"] for r in rs) != list(range(n)):
            raise ValueError(f"missing/duplicate game IDs for {opp}: {path}")
        if any(r["agent_first"] != (r["game_id"] % 2 == 0)
               or r["opening"] != split_cfg.get("opening_plies", 2) for r in rs):
            raise ValueError(f"wrong seat or opening protocol: {path}")
    recalculated = summarize(records)
    if not np.isclose(res.get("macro_score", np.nan), recalculated["macro_score"], atol=1e-12):
        raise ValueError(f"stored macro score disagrees with game records: {path}")
    for opp in opponents:
        if not np.isclose(res["per_opponent"][opp]["score"], recalculated["per_opponent"][opp]["score"], atol=1e-12):
            raise ValueError(f"stored opponent score disagrees with records: {path}")
    # Recalculate uncertainty from original records without mutating the input JSON.
    return {**res, **recalculated}


def load_inputs(cfg: dict, runs: Path, results: Path, references: Path):
    exp, seeds = cfg["experiment"], cfg["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("report needs a nonempty set of unique training seeds")
    tests, lasts, open0, trains, hists = [], [], [], [], []
    for s in seeds:
        run_dir = runs / exp / f"seed{s}"
        if load(run_dir / "config.json") != cfg:
            raise ValueError(f"saved training config differs from report config: {run_dir}")
        train, hist = load(run_dir / "train_summary.json"), load(run_dir / "history.json")
        if train.get("seed") != s or not hist or hist[-1]["episode"] != cfg["train"]["episodes"]:
            raise ValueError(f"incomplete or mismatched training run: {run_dir}")
        episodes = [r["episode"] for r in hist]
        if episodes != list(range(cfg["train"]["log_every"], cfg["train"]["episodes"] + 1, cfg["train"]["log_every"])):
            raise ValueError(f"incomplete training history: {run_dir}")
        best = max((r for r in hist if "val_macro" in r), key=lambda r: r["val_macro"])
        if best["episode"] != train["best_episode"] or not np.isclose(best["val_macro"], train["best_val_macro"]):
            raise ValueError(f"best checkpoint was not selected by validation: {run_dir}")
        for ckpt in ("best.pt", "last.pt"):
            if not (run_dir / ckpt).is_file():
                raise FileNotFoundError(f"missing checkpoint: {run_dir / ckpt}")
        for prefix, ckpt, split_cfg, target in (
            ("test_seed", "best.pt", cfg["eval"]["test"], tests),
            ("test_last_seed", "last.pt", cfg["eval"]["test"], lasts),
            ("test_open0_seed", "best.pt", {**cfg["eval"]["test"], "opening_plies": 0}, open0),
        ):
            target.append(checked_evaluation(results / exp / f"{prefix}{s}.json", split_cfg,
                                             f"dqn:{run_dir / ckpt}"))
        trains.append(train)
        hists.append(hist)
    refs = {a: checked_evaluation(references / f"test_{a}.json", cfg["eval"]["test"], a)
            for a in ("random", "tactical", "lookahead")}
    return tests, lasts, open0, trains, hists, refs


def board_ascii(board: list[int], move: int, correct: list[int], mark: int) -> str:
    b = Board.from_kaggle(board)
    sym = {0: ".", 1: "X", 2: "O"}
    rows = [" ".join(sym[int(v)] for v in row) for row in b.to_grid()]
    marker = ["  "] * 7
    marker[move] = "^ "
    for c in correct:
        marker[c] = "* " if c != move else "^*"
    return "\n".join(rows + ["0 1 2 3 4 5 6", "".join(marker).rstrip(),
                             f"агент играет {sym[mark]}; ^ — ход агента, * — правильный ход"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--runs", default="runs")
    p.add_argument("--results", default="results")
    p.add_argument("--references", "--reference-dir", dest="references", default=None,
                   help="reference evaluations directory (default: <results>/reference)")
    args = p.parse_args()
    cfg = load_config(args.config)
    exp = cfg["experiment"]
    seeds = cfg["seeds"]
    res_dir = Path(args.results) / exp
    fig_dir = res_dir / "figures"
    opponents = cfg["eval"]["test"]["opponents"]

    tests, lasts, open0, trains, hists, refs = load_inputs(
        cfg, Path(args.runs), Path(args.results), Path(args.references) if args.references else Path(args.results) / "reference")
    fig_dir.mkdir(parents=True, exist_ok=True)

    def agg(results: list[dict]) -> dict:
        macro = np.array([t["macro_score"] for t in results])
        per = {o: np.array([t["per_opponent"][o]["score"] for t in results]) for o in opponents}
        out = dict(n_seeds=len(results), macro_mean=float(macro.mean()), macro_std=float(macro.std(ddof=1)) if len(macro) > 1 else 0.0,
                   macro_per_seed=macro.tolist(),
                   per_opponent={o: dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0)
                                 for o, v in per.items()})
        for side in ("score_first", "score_second"):
            out[side] = {o: float(np.mean([t["per_opponent"][o][side] for t in results])) for o in opponents}
        for k, _ in ERR_KEYS:
            # микро-усреднение по всем возможностям ошибиться
            num_key, den_key = {"missed_win_rate": ("missed_wins", "win_opps"),
                                "missed_block_rate": ("missed_blocks", "block_opps"),
                                "blunder_rate": ("blunders", "safe_opps")}[k]
            num = sum(t["per_opponent"][o][num_key] for t in results for o in opponents)
            den = sum(t["per_opponent"][o][den_key] for t in results for o in opponents)
            out[k] = num / den if den else None
        out["mean_length"] = {o: float(np.mean([t["per_opponent"][o]["mean_length"] for t in results])) for o in opponents}
        out["illegal_rate"] = float(np.mean([t["per_opponent"][o]["illegal_rate"] for t in results for o in opponents]))
        return out

    summary = dict(experiment=exp, seeds=seeds, test_cfg=cfg["eval"]["test"])
    summary["dqn_best"] = agg(tests)
    summary["dqn_best"]["macro_ci95_hier"] = hier_bootstrap(tests, opponents)
    # средняя ширина ДИ по партиям внутри одного сида vs разброс между сидами (RQ2)
    summary["dqn_best"]["within_seed_ci_halfwidth"] = float(np.mean([(t["macro_ci95"][1] - t["macro_ci95"][0]) / 2 for t in tests]))
    if lasts:
        summary["dqn_last"] = agg(lasts)
    if open0:
        summary["dqn_best_opening0"] = agg(open0)
    summary["references"] = {a: dict(macro=r["macro_score"], macro_ci95=r["macro_ci95"],
                                      per_opponent={o: r["per_opponent"][o]["score"] for o in opponents},
                                      **{k: _micro(r, k, opponents) for k, _ in ERR_KEYS})
                             for a, r in refs.items()}
    summary["train"] = trains
    val_scores = np.array([t["best_val_macro"] for t in trains])
    summary["validation"] = dict(mean=float(val_scores.mean()),
                                  std=float(val_scores.std(ddof=1)) if len(val_scores) > 1 else 0.0,
                                  per_seed=val_scores.tolist(),
                                  selected_submission_seed=seeds[int(val_scores.argmax())])
    summary["comparisons"] = {
        "dqn_minus_random": paired_comparison(tests, [refs["random"]], opponents),
        "best_minus_last": paired_comparison(tests, lasts, opponents),
        "opening0_minus_opening2": paired_comparison(open0, tests, opponents, same_openings=False),
    }
    summary["seed_table"] = [dict(seed=s, best_episode=tr["best_episode"], val_macro=tr["best_val_macro"],
                                       test_best=t["macro_score"], test_last=l["macro_score"],
                                       test_opening0=o["macro_score"], train_time_s=tr["train_time_s"])
                              for s, tr, t, l, o in zip(seeds, trains, tests, lasts, open0)]
    summary["win_ci95_by_seed_and_side"] = {
        str(s): {o: t["per_opponent"][o]["win_ci95_by_side"] for o in opponents}
        for s, t in zip(seeds, tests)}
    summary["uncertainty_method"] = {
        "macro": "crossed bootstrap: model seeds and shared game IDs within opponent x seat strata",
        "paired": "same resampled model/game indices for both agents; one reference realization shared across models",
        "n_boot": 2000, "bootstrap_seed": 0,
        "scope": "conditional on fixed opponent implementations and opening distribution; not unseen-opponent uncertainty",
        "zero_results": "bootstrap can degenerate; Wilson intervals for win probability reported separately by seed and seat",
    }
    dump_json(summary, res_dir / "summary.json")

    # ------------------------------------------------------------------ таблица
    d = summary["dqn_best"]
    lines = ["| Агент | " + " | ".join(opponents) + " | **macro score** |", "|---" * (len(opponents) + 2) + "|"]
    for a, r in summary["references"].items():
        lines.append(f"| {a} (эталон) | " + " | ".join(f"{r['per_opponent'][o]:.3f}" for o in opponents)
                     + f" | {r['macro']:.3f} [{r['macro_ci95'][0]:.3f}, {r['macro_ci95'][1]:.3f}] |")
    lines.append(f"| **DQN baseline** (best-val, {d['n_seeds']} сидов) | "
                 + " | ".join(f"{d['per_opponent'][o]['mean']:.3f} ± {d['per_opponent'][o]['std']:.3f}" for o in opponents)
                 + f" | **{d['macro_mean']:.3f} ± {d['macro_std']:.3f}** [{d['macro_ci95_hier'][0]:.3f}, {d['macro_ci95_hier'][1]:.3f}] |")
    if lasts:
        l = summary["dqn_last"]
        lines.append(f"| DQN (последний чекпойнт) | "
                     + " | ".join(f"{l['per_opponent'][o]['mean']:.3f} ± {l['per_opponent'][o]['std']:.3f}" for o in opponents)
                     + f" | {l['macro_mean']:.3f} ± {l['macro_std']:.3f} |")
    table = "\n".join(lines)

    seed_lines = ["| Train seed | Выбранный эпизод | Best val macro | Test best | Test last | Test opening=0 | Обучение, с |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary["seed_table"]:
        seed_lines.append(f"| {r['seed']} | {r['best_episode']} | {r['val_macro']:.4f} | {r['test_best']:.4f} | "
                          f"{r['test_last']:.4f} | {r['test_opening0']:.4f} | {r['train_time_s']:.1f} |")
    val = summary["validation"]
    val_text = (f"Среднее best-val: **{val['mean']:.4f} ± {val['std']:.4f}**. "
                f"Сабмишен выбирается по val: train seed **{val['selected_submission_seed']}**. "
                "Это оценка, использованная для выбора чекпойнта, поэтому она оптимистична; "
                "финальные выводы о качестве опираются на отдельный test. Val и test имеют разный пул соперников, "
                "поэтому разность их macro score не является оценкой переобучения.")

    comp = summary["comparisons"]
    comparison_lines = ["| Сравнение | Средняя разность macro score | 95% парный ДИ |",
                        "|---|---:|---:|"]
    for key, name in (("dqn_minus_random", "DQN − random"), ("best_minus_last", "best-val − last"),
                      ("opening0_minus_opening2", "opening=0 − opening=2")):
        c = comp[key]
        comparison_lines.append(f"| {name} | {c['mean']:+.4f} | [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] |")
    bl = comp["best_minus_last"]
    rq = [
        "**RQ1 — помогает ли выбор best-val по сравнению с последним чекпойнтом?** "
        f"Средняя парная разность {bl['mean']:+.4f}, 95% ДИ [{bl['ci95'][0]:+.4f}, {bl['ci95'][1]:+.4f}]. "
        + ("ДИ включает ноль: уверенного преимущества нет. " if bl["ci95"][0] <= 0 <= bl["ci95"][1]
           else "ДИ не включает ноль. ")
        + "Правило best-val сохраняется независимо от результатов этой test-диагностики.",
        "**RQ2 — насколько результат зависит от сида обучения?** "
        f"Test macro по сидам: {', '.join(f'{v:.4f}' for v in d['macro_per_seed'])}; "
        f"std={d['macro_std']:.4f}, диапазон={min(d['macro_per_seed']):.4f}–{max(d['macro_per_seed']):.4f}; "
        f"средняя полуширина ДИ по партиям внутри одного сида={d['within_seed_ci_halfwidth']:.4f}. "
        "Разброс оценивается по независимым запускам обучения, а не по повторам одного чекпойнта.",
        "**RQ3 — меняется ли оценка при отказе от случайного дебюта?** "
        f"При opening=0 macro={summary['dqn_best_opening0']['macro_mean']:.4f}, "
        f"при opening=2 macro={d['macro_mean']:.4f}; парная разность "
        f"{comp['opening0_minus_opening2']['mean']:+.4f}, 95% ДИ [{comp['opening0_minus_opening2']['ci95'][0]:+.4f}, "
        f"{comp['opening0_minus_opening2']['ci95'][1]:+.4f}]. "
        "Это разные распределения стартовых позиций, а не улучшение агента. "
        "Парность сохраняется по train seed и RNG seed партии; сами дебюты в этом сравнении различаются.",
    ]

    uncertainty = (
        "В таблице ± обозначает выборочное стандартное отклонение по train seeds; квадратные скобки — 95% ДИ. "
        "Для DQN и разностей используется 2000 bootstrap-реплик: ресэмплируются train seeds, затем общие game_id "
        "внутри каждой страты (соперник, сторона). Один набор game_id применяется ко всем моделям и обеим "
        "сторонам сравнения; доли соперников и первого/второго хода фиксированы. Для эталонов ресэмплируются "
        "только партии по тем же стратам. Это условная неопределённость для данного пула соперников; "
        "неопределённость переноса на другие стратегии не оценена. При пяти train seeds интервалы приблизительны. "
        "Bootstrap по одинаковым исходам даёт вырожденный ДИ и не доказывает нулевую вероятность победы. "
        "В summary.json дополнительно сохранены интервалы Wilson для win rate каждого train seed отдельно "
        "по сопернику и стороне; их нельзя подменять интервалами для score или объединять пять сидов "
        "как независимые партии."
    )
    zero_examples = []
    for o in opponents:
        rs = [r for r in tests[0]["records"] if r["opponent"] == o and r["agent_first"]]
        if rs and all(r["result"] == 0 for r in rs):
            ci = tests[0]["per_opponent"][o]["win_ci95_by_side"]["first"]
            zero_examples.append(f"Например, seed {seeds[0]} первым против {o}: 0/{len(rs)} побед; "
                                 f"95% Wilson для win rate [{ci[0]:.4f}, {ci[1]:.4f}].")

    side = ["| Соперник | DQN первым | DQN вторым | средняя длина партии |", "|---|---|---|---|"]
    for o in opponents:
        side.append(f"| {o} | {d['score_first'][o]:.3f} | {d['score_second'][o]:.3f} | {d['mean_length'][o]:.1f} |")

    err = ["| Тип ошибки | random | tactical | lookahead | DQN baseline |", "|---|---|---|---|---|"]
    for k, name in ERR_KEYS:
        row = [summary["references"].get(a, {}).get(k) for a in ("random", "tactical", "lookahead")] + [d[k]]
        err.append(f"| {name} | " + " | ".join("—" if v is None else f"{v:.3f}" for v in row) + " |")

    err_slices = ["| Соперник | Пропущенные победы / возможности | Пропущенные блоки / возможности | Подставы / возможности |",
                  "|---|---:|---:|---:|"]
    for o in opponents:
        counts = []
        for num, den in (("missed_wins", "win_opps"), ("missed_blocks", "block_opps"), ("blunders", "safe_opps")):
            n = sum(t["per_opponent"][o][num] for t in tests)
            total = sum(t["per_opponent"][o][den] for t in tests)
            counts.append(f"{n}/{total} ({n / total:.1%})" if total else "0/0 (нет возможностей)")
        err_slices.append(f"| {o} | " + " | ".join(counts) + " |")

    # примеры ошибок: по одному на каждый тип из seed0
    ex_md = []
    seen = set()
    for t in tests:
        for e in t["examples"]:
            if e["kind"] not in seen:
                seen.add(e["kind"])
                ex_md.append(f"**{e['kind']}** (соперник `{e['opponent']}`, партия {e['game_id']}):\n```\n"
                             + board_ascii(e["board"], e["move"], e["correct"], e["mark"]) + "\n```")
    # Пустая строка после каждой таблицы: иначе Markdown продолжает таблицу следующим абзацем.
    report = ["# Отчёт baseline ConnectX\n", "## Validation и выбор чекпойнта\n", "\n".join(seed_lines), "", val_text,
              "\n## Test: score (W=1, D=0.5, L=0)\n", table,
              "\n## Парные сравнения и исследовательские вопросы\n", "\n".join(comparison_lines), "", "\n\n".join(rq),
              "\n## Неопределённость оценки\n", uncertainty, "", " ".join(zero_examples),
              "\n## По сторонам\n", "\n".join(side),
              "\n## Тактические ошибки (доля от возможностей ошибиться)\n", "\n".join(err),
              "\n### Срез по соперникам\n", "\n".join(err_slices),
              "\n## Примеры ошибок\n", "\n\n".join(ex_md)]
    (res_dir / "report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))

    # ------------------------------------------------------------------ графики
    plotting.setup()
    import matplotlib.pyplot as plt

    val_keys = [("train_win", "доля побед в обучении\n(vs random, с ε-исследованием)")] + \
               [(f"val_{o}", f"val score vs {o}") for o in cfg["eval"]["val"]["opponents"]]
    fig, axes = plt.subplots(1, len(val_keys), figsize=(4 * len(val_keys), 3.4), sharey=True)
    for ax, (k, title) in zip(axes, val_keys):
        curves = []
        for h in hists:
            if not h:
                continue
            pts = [(r["episode"], r[k]) for r in h if k in r]
            x, y = zip(*pts)
            ax.plot(x, y, color=plotting.SERIES[0], alpha=0.25, linewidth=1)
            curves.append(y)
        if curves:
            m = np.mean(curves, axis=0)
            ax.plot(x, m, color=plotting.SERIES[0], linewidth=2, label="среднее по сидам")
        ax.set_title(title)
        ax.set_xlabel("эпизод")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("score / win rate")
    axes[0].legend(loc="lower right")
    plotting.save(fig, fig_dir / "learning_curves.png")

    agents = [(a, summary["references"][a]["per_opponent"], None) for a in ("random", "tactical") if a in summary["references"]]
    agents.append(("DQN baseline", {o: d["per_opponent"][o]["mean"] for o in opponents},
                   {o: d["per_opponent"][o]["std"] for o in opponents}))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = 0.8 / len(agents)
    xs = np.arange(len(opponents))
    for i, (name, vals, errs) in enumerate(agents):
        ax.bar(xs + (i - (len(agents) - 1) / 2) * w, [vals[o] for o in opponents], width=w * 0.92,
               color=plotting.SERIES[i], label=name,
               yerr=[errs[o] for o in opponents] if errs else None, capsize=3, error_kw=dict(ecolor=plotting.TEXT2, lw=1))
    ax.axhline(0.5, color=plotting.TEXT2, lw=1, ls="--")
    ax.set_xticks(xs, opponents)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Test: score против отложенного пула соперников (DQN: среднее ± std по сидам)")
    ax.legend(ncol=len(agents), loc="upper right")
    plotting.save(fig, fig_dir / "test_scores.png")

    fig, ax = plt.subplots(figsize=(7, 3.4))
    err_agents = [(a, summary["references"][a]) for a in ("random", "tactical") if a in summary["references"]] + [("DQN baseline", d)]
    xs = np.arange(len(ERR_KEYS))
    w = 0.8 / len(err_agents)
    for i, (name, src) in enumerate(err_agents):
        ax.bar(xs + (i - (len(err_agents) - 1) / 2) * w, [src[k] or 0 for k, _ in ERR_KEYS], width=w * 0.92,
               color=plotting.SERIES[i], label=name)
    ax.set_xticks(xs, [n for _, n in ERR_KEYS])
    ax.set_ylabel("доля ошибок")
    ax.set_title("Тактические ошибки на test")
    ax.legend()
    plotting.save(fig, fig_dir / "tactical_errors.png")


def _micro(r: dict, k: str, opponents: list[str]):
    num_key, den_key = {"missed_win_rate": ("missed_wins", "win_opps"), "missed_block_rate": ("missed_blocks", "block_opps"),
                        "blunder_rate": ("blunders", "safe_opps")}[k]
    num = sum(r["per_opponent"][o][num_key] for o in opponents)
    den = sum(r["per_opponent"][o][den_key] for o in opponents)
    return num / den if den else None


if __name__ == "__main__":
    main()
