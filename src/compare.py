"""Create a compact comparison chart from native-evaluation summary files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", help="Paths to *_summary.json files")
    parser.add_argument("--output", default="results/comparison.png")
    args = parser.parse_args()

    data = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.summaries]
    labels = [item["label"] for item in data]
    means = [item["mean_native_return"] for item in data]
    errors = [item["std_native_return"] for item in data]
    success = [100 * item["success_rate_return_ge_200"] for item in data]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, means, yerr=errors, capsize=5)
    axes[0].axhline(200, color="tab:green", linestyle="--", label="Solved threshold")
    axes[0].set_ylabel("Native evaluation return")
    axes[0].legend()
    axes[1].bar(labels, success, color="tab:orange")
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Success rate (%)")
    figure.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(f"Chart written to {output}")


if __name__ == "__main__":
    main()
