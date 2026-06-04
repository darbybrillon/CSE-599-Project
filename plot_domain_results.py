"""Plot cross-domain tokenizer results produced by run_domain_eval.py.

Reads the JSON summaries written to the results directory and renders, for each
vocab size:
  * In-domain compression ratio vs. training scale, one panel per domain.
  * In-domain k-gram conditional entropy vs. k, one panel per domain.
  * A domain-mismatch heatmap (train-domain x test-domain) of compression ratio.

Figures are written next to the existing c4 ``Graphs`` so the code/math panels
sit beside the originals. No GPU, no network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FAMILIES = ["bpe", "unigram", "wordpiece", "wordlevel", "superbpe"]
COLORS = {
    "bpe": "C0",
    "unigram": "C1",
    "wordpiece": "C2",
    "wordlevel": "C3",
    "superbpe": "C4",
}
MARKERS = {
    "bpe": "o",
    "unigram": "s",
    "wordpiece": "^",
    "wordlevel": "D",
    "superbpe": "P",
}
DOMAIN_LABEL = {"c4": "News (C4)", "codeparrot": "Code", "gsm8k": "Math (GSM8K)"}


def _scale_to_log10(scale: str) -> float:
    # Scale codes are 10^X sample counts encoded as "1 0 X" -> use trailing digits.
    digits = "".join(ch for ch in scale if ch.isdigit())
    return float(int(digits) - 100) if len(digits) == 3 else float(int(digits))


def plot_in_domain(summary: dict, out_dir: Path, vocab: int) -> list[Path]:
    domains = summary["domains"]
    written: list[Path] = []

    # Compression vs scale, one subplot per domain.
    fig, axes = plt.subplots(1, len(domains), figsize=(5 * len(domains), 4), squeeze=False)
    for ax, (domain, by_scale) in zip(axes[0], domains.items()):
        scales = sorted(by_scale.keys(), key=_scale_to_log10)
        xs = [_scale_to_log10(s) for s in scales]
        for fam in FAMILIES:
            ys = [
                by_scale[s][fam]["compression_ratio"]
                for s in scales
                if fam in by_scale[s]
            ]
            xf = [_scale_to_log10(s) for s in scales if fam in by_scale[s]]
            if ys:
                ax.plot(xf, ys, marker=MARKERS[fam], color=COLORS[fam], label=fam)
        ax.set_title(DOMAIN_LABEL.get(domain, domain))
        ax.set_xlabel(r"training data ($\log_{10}$ samples)")
        ax.set_ylabel("compression ratio (bytes/token)")
        ax.grid(True, alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"In-domain compression vs. scale (vocab {vocab})")
    fig.tight_layout()
    path = out_dir / f"{vocab}_in_domain_compression.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # k-gram entropy vs k at the largest available scale, one subplot per domain.
    fig, axes = plt.subplots(1, len(domains), figsize=(5 * len(domains), 4), squeeze=False)
    for ax, (domain, by_scale) in zip(axes[0], domains.items()):
        if not by_scale:
            continue
        top_scale = max(by_scale.keys(), key=_scale_to_log10)
        fam_metrics = by_scale[top_scale]
        for fam in FAMILIES:
            if fam not in fam_metrics:
                continue
            ent = fam_metrics[fam]["entropies_bits_per_token"]
            orders = sorted(ent.keys(), key=lambda k: int(k[1:]))
            ks = [int(k[1:]) for k in orders]
            ys = [ent[k] for k in orders]
            ax.plot(ks, ys, marker=MARKERS[fam], color=COLORS[fam], label=fam)
        ax.set_title(f"{DOMAIN_LABEL.get(domain, domain)} (10^{int(_scale_to_log10(top_scale))})")
        ax.set_xlabel("k-gram order")
        ax.set_ylabel("conditional entropy (bits/token)")
        ax.grid(True, alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"In-domain k-gram entropy (vocab {vocab})")
    fig.tight_layout()
    path = out_dir / f"{vocab}_in_domain_kgram_entropy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)
    return written


def plot_mismatch(summary: dict, out_dir: Path, family: str = "bpe") -> Path:
    vocab = summary["vocab"]
    scale = summary["scale"]
    matrix = summary["matrix"]
    train_domains = list(matrix.keys())
    test_domains = sorted({td for row in matrix.values() for td in row})

    grid = np.full((len(train_domains), len(test_domains)), np.nan)
    for i, tr in enumerate(train_domains):
        for j, te in enumerate(test_domains):
            cell = matrix.get(tr, {}).get(te)
            if cell and family in cell:
                grid[i, j] = cell[family]["compression_ratio"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, cmap="viridis")
    ax.set_xticks(range(len(test_domains)))
    ax.set_xticklabels([DOMAIN_LABEL.get(d, d) for d in test_domains], rotation=20)
    ax.set_yticks(range(len(train_domains)))
    ax.set_yticklabels([DOMAIN_LABEL.get(d, d) for d in train_domains])
    ax.set_xlabel("evaluated on (test domain)")
    ax.set_ylabel("tokenizer trained on")
    for i in range(len(train_domains)):
        for j in range(len(test_domains)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="w")
    fig.colorbar(im, ax=ax, label="compression ratio (bytes/token)")
    ax.set_title(f"Domain mismatch — {family}, vocab {vocab}, 10^{int(_scale_to_log10(scale))}")
    fig.tight_layout()
    path = out_dir / f"{vocab}_{scale}_mismatch_{family}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--vocabs", nargs="+", type=int, default=[16000, 64000])
    parser.add_argument("--mismatch-scale", default="106")
    parser.add_argument(
        "--mismatch-families",
        nargs="+",
        default=["bpe", "superbpe"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for vocab in args.vocabs:
        in_path = args.results / f"in_domain_{vocab}.json"
        if in_path.is_file():
            summary = json.loads(in_path.read_text(encoding="utf-8"))
            written += plot_in_domain(summary, args.out_dir, vocab)

        mm_path = args.results / f"mismatch_{vocab}_{args.mismatch_scale}.json"
        if mm_path.is_file():
            summary = json.loads(mm_path.read_text(encoding="utf-8"))
            for fam in args.mismatch_families:
                written.append(plot_mismatch(summary, args.out_dir, fam))

    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
