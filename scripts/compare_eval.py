#!/usr/bin/env python3
"""[Eval-2] A~E 평가 매트릭스 비교표 + 시각화.

CLAUDE.md 평가 매트릭스(A~E):
    A: cam_port(외부고정) / noise=none   — 기존 베이스라인
    B: cam_eih(eye-in-hand) / noise=none — eye-in-hand 전환 자체의 영향
    C: cam_eih / noise=low
    D: cam_eih / noise=medium            — D435 실측 기준
    E: cam_eih / noise=high

Usage:
    conda activate wemeet
    python scripts/compare_eval.py
입력: outputs/eval_A_eth_none.json ~ outputs/eval_E_eih_high.json
출력: outputs/eval_comparison.png (+ stdout 표)
"""
import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
OUT_PNG = os.path.join(OUTPUT_DIR, "eval_comparison.png")

RUNS = [
    ("A", "eval_A_eth_none.json",   "cam_port / none"),
    ("B", "eval_B_eih_none.json",   "cam_eih / none"),
    ("C", "eval_C_eih_low.json",    "cam_eih / low"),
    ("D", "eval_D_eih_medium.json", "cam_eih / medium"),
    ("E", "eval_E_eih_high.json",   "cam_eih / high"),
]

METRICS = [
    ("obj_trans_err_mm", "ob_in_cam trans (mm)"),
    ("obj_rot_err_deg", "ob_in_cam rot (deg)"),
    ("charge_port_approach_landing_pos_err_mm", "approach landing (mm)"),
    ("charge_port_insert_landing_pos_err_mm", "insert landing (mm)"),
    ("charge_port_insert_landing_rot_err_deg", "insert landing (deg)"),
]


def load_runs():
    runs = []
    for tag, fname, label in RUNS:
        path = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{tag} 결과 파일 없음: {path}")
        with open(path) as f:
            data = json.load(f)
        runs.append({"tag": tag, "label": label, "data": data})
    return runs


def print_table(runs):
    header = f"{'#':3s} {'config':18s}" + "".join(f"{m[1]:>24s}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for r in runs:
        agg = r["data"]["aggregate"]
        n = r["data"]["config"]["n"]
        seed = r["data"]["config"]["seed"]
        row = f"{r['tag']:3s} {r['label']:18s}"
        for key, _ in METRICS:
            m = agg[key]["mean"]
            mx = agg[key]["max"]
            row += f"{m:9.2f}/{mx:6.2f}max"
        print(row)
    print(f"\n(N={runs[0]['data']['config']['n']}, seed={runs[0]['data']['config']['seed']} 공통 / "
          f"각 셀 = mean/max)")


def save_png(runs, path):
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    tags = [r["tag"] for r in runs]
    colors = ["tab:gray", "tab:blue", "tab:green", "tab:orange", "tab:red"]

    for ax, (key, title) in zip(axes, METRICS):
        means = [r["data"]["aggregate"][key]["mean"] for r in runs]
        stds = [r["data"]["aggregate"][key]["std"] for r in runs]
        ax.bar(tags, means, yerr=stds, color=colors[:len(tags)], capsize=4)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("run")
        for i, m in enumerate(means):
            ax.text(i, m, f"{m:.2f}", ha="center", va="bottom", fontsize=8)

    labels = " | ".join(f"{r['tag']}={r['label']}" for r in runs)
    fig.suptitle(f"Eval matrix A-E (N={runs[0]['data']['config']['n']}, "
                f"seed={runs[0]['data']['config']['seed']})\n{labels}", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=120)
    print(f"\nsaved: {path}")


def main():
    runs = load_runs()
    print_table(runs)
    save_png(runs, OUT_PNG)


if __name__ == "__main__":
    main()
