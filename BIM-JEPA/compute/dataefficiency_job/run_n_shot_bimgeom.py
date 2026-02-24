import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import csv
from typing import List, Dict, Tuple
import math

METRIC_PATTERNS = {
    "overall_accuracy": re.compile(r"Overall Accuracy:\s+([0-9.]+)"),
    "mean_class_accuracy": re.compile(r"Mean Class Accuracy:\s+([0-9.]+)"),
    "precision": re.compile(r"Precision KATEX_INLINE_OPENMacroKATEX_INLINE_CLOSE:\s+([0-9.]+)"),
    "recall": re.compile(r"Recall KATEX_INLINE_OPENMacroKATEX_INLINE_CLOSE:\s+([0-9.]+)"),
    "f1_score": re.compile(r"F1 Score KATEX_INLINE_OPENMacroKATEX_INLINE_CLOSE:\s+([0-9.]+)"),
}

def parse_test_stdout(stdout: str) -> Dict[str, float]:
    out = {}
    for k, pat in METRIC_PATTERNS.items():
        m = pat.search(stdout)
        if m: out[k] = float(m.group(1))
    return out

def list_version_dirs(logs_root: Path) -> List[Path]:
    if not logs_root.exists(): return []
    return sorted([p for p in logs_root.iterdir() if p.is_dir() and p.name.startswith("version_")])

def newest_version_dir(logs_root: Path, before: List[Path]) -> Path:
    before_set = set(p.name for p in before)
    after = list_version_dirs(logs_root)
    new = [p for p in after if p.name not in before_set]
    if len(new) == 1: return new[0]
    if after: return max(after, key=lambda p: p.stat().st_mtime)
    raise RuntimeError(f"No lightning_logs found in {logs_root}")

def find_last_ckpt(version_dir: Path) -> Path:
    ckpt_dir = version_dir / "checkpoints"
    if ckpt_dir.exists():
        last_ckpt = ckpt_dir / "last.ckpt"
        if last_ckpt.exists(): return last_ckpt
        all_ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if all_ckpts: return all_ckpts[0]
    raise RuntimeError(f"No checkpoints found under {version_dir}")

def run_cmd(cmd: List[str], env=None) -> Tuple[int, str, str]:
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    return res.returncode, res.stdout, res.stderr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/BIM-JEPA/classification/bimgeom_n_shot.yaml")
    parser.add_argument("--artifacts_root", type=str, default="/home/users/nus/e1291616/jepa/artifacts")
    parser.add_argument("--n_shots", type=str, default="140,100,50,25,10,5")
    parser.add_argument("--seeds", type=str, default="42,1337,2024")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--python", type=str, default=sys.executable)
    args = parser.parse_args()

    n_shots = [int(x) for x in args.n_shots.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    artifacts_root = Path(args.artifacts_root).resolve()
    logs_root = artifacts_root / "lightning_logs"
    results_dir = artifacts_root / "n_shot_results_bimgeom"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_rows = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_csv = results_dir / f"bimgeom_n_shot_runs_{ts}.csv"

    for n in n_shots:
        for seed in seeds:
            tag = f"n_{n}_seed_{seed}"
            print(f"\n=== Running {tag} ===")
            before_dirs = list_version_dirs(logs_root)

            fit_cmd = [
                args.python, "-m", "bimjepa.tasks.classification", "fit",
                "--config", args.config,
                f"--seed_everything={seed}",
                f"--trainer.default_root_dir={str(artifacts_root)}",
                "--data.class_path=bimjepa.datasets.bimgeom_datamodule_n_shot.BIMGEOMDataModuleNShot",
                f"--data.init_args.samples_per_class={n}",
                f"--data.init_args.subset_seed={seed}",
                "--data.init_args.val_split_ratio=0.0",
            ]

            if args.dry_run:
                print("[DRY RUN] Would run fit:", " ".join(fit_cmd))
                continue

            rc, out, err = run_cmd(fit_cmd)
            if rc != 0:
                print(f"[ERROR][FIT] {tag} failed:\n{err}")
                continue
            
            try:
                version_dir = newest_version_dir(logs_root, before_dirs)
                ckpt_path = find_last_ckpt(version_dir)
            except Exception as e:
                print(f"[ERROR] Could not find checkpoint for {tag}: {e}")
                continue
            print(f"Checkpoint for {tag}: {ckpt_path}")

            test_cmd = [
                args.python, "-m", "bimjepa.tasks.classification", "test",
                "--config", args.config,
                f"--ckpt_path={str(ckpt_path)}",
                f"--seed_everything={seed}",
                f"--trainer.default_root_dir={str(artifacts_root)}",
                "--data.class_path=bimjepa.datasets.bimgeom_datamodule_n_shot.BIMGEOMDataModuleNShot",
                f"--data.init_args.samples_per_class={n}",
                f"--data.init_args.subset_seed={seed}",
                "--data.init_args.val_split_ratio=0.0",
            ]
            rc_test, out_test, err_test = run_cmd(test_cmd)
            if rc_test != 0:
                print(f"[ERROR][TEST] {tag} failed:\n{err_test}")
                continue

            metrics = parse_test_stdout(out_test)
            print(f"Parsed metrics for {tag}:", metrics)
            row = {"tag": tag, "n_shot": n, "seed": seed, "ckpt_path": str(ckpt_path), **metrics}
            results_rows.append(row)

            write_header = not run_csv.exists()
            with open(run_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=row.keys())
                if write_header: w.writeheader()
                w.writerow(row)

    print(f"\nDone. Wrote per-run results to: {run_csv}")

    by_shot: Dict[int, List[Dict[str, float]]] = {}
    for r in results_rows: by_shot.setdefault(r["n_shot"], []).append(r)

    agg_csv = results_dir / f"bimgeom_n_shot_agg_{ts}.csv"
    with open(agg_csv, "w", newline="") as f:
        fieldnames = ["n_shot", "n_seeds", "overall_accuracy_mean", "overall_accuracy_std",
                      "mean_class_accuracy_mean", "mean_class_accuracy_std",
                      "precision_mean", "precision_std", "recall_mean", "recall_std",
                      "f1_score_mean", "f1_score_std"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        def mean(xs): return sum(xs)/len(xs) if xs else float("nan")
        def std(xs):
            if len(xs) < 2: return float("nan")
            m = mean(xs)
            return math.sqrt(sum((x-m)**2 for x in xs)/(len(xs)-1))

        for n_shot, rows in sorted(by_shot.items()):
            def c(k): return [r.get(k, float("nan")) for r in rows if k in r]
            w.writerow({
                "n_shot": n_shot, "n_seeds": len(rows),
                "overall_accuracy_mean": mean(c("overall_accuracy")), "overall_accuracy_std": std(c("overall_accuracy")),
                "mean_class_accuracy_mean": mean(c("mean_class_accuracy")), "mean_class_accuracy_std": std(c("mean_class_accuracy")),
                "precision_mean": mean(c("precision")), "precision_std": std(c("precision")),
                "recall_mean": mean(c("recall")), "recall_std": std(c("recall")),
                "f1_score_mean": mean(c("f1_score")), "f1_score_std": std(c("f1_score")),
            })
    print(f"Aggregated results written to: {agg_csv}")

if __name__ == "__main__":
    main()