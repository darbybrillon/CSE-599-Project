#%%
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "evaluate.py").exists():
    ROOT = ROOT / "CSE-599-Project"

dataset = "gsm8k"
vocab_size = 64000


TEST_CORPUS = ROOT / f"data/{dataset}_106/test.jsonl"


STANDARD_PATTERN = f"huggingface_tokenizers_64000_{dataset}_*"

SUPERBPE_PATTERN = f"{vocab_size}_{dataset}_*"

FAMILIES = ["bpe", "unigram", "wordpiece", "wordlevel"]
COLORS   = {"bpe": "C0", "unigram": "C1", "wordpiece": "C2", "wordlevel": "C3"}
MARKERS  = {"bpe": "o",  "unigram": "s",  "wordpiece": "^",  "wordlevel": "D"}

TOKENIZER_DIRS = {
    "standard": Path(fr"C:\Users\darby\Downloads\CSE 599\Project\tokenizers\{dataset}"),
    "superbpe": Path(fr"C:\Users\darby\Downloads\CSE 599\Project\SuperBPE Tokenizers\{dataset}"),
}
SOURCE_LINESTYLES = {"standard": "-",  "superbpe": "-"}
SOURCE_LABELS     = {"standard": "Standard", "superbpe": "SuperBPE"}

#%%

# Helpers

def train_chars_for_scale(scale: str) -> int | None:
    train_path = ROOT / f"data/{dataset}_{scale}/train.jsonl"
    if not train_path.exists():
        return None
    return sum(len(json.loads(line)["text"]) for line in train_path.open(encoding="utf-8"))


def _scale_from_dir(tokenizer_dir: Path, source: str) -> str:
    name = tokenizer_dir.name
    if source == "superbpe":
        m = re.search(fr"_{dataset}_(.+)$", name)
    else:
        m = re.search(fr"_{dataset}_(.+)$", name)
    return m.group(1) if m else name


def _evaluate_superbpe_dir(tokenizer_dir: Path, texts: list[str]) -> dict:

    from superbpe.utils import construct_hf_tokenizer
    from transformers import AutoTokenizer

    construct_hf_tokenizer(tokenizer_dir)
    hf_tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "tokenizer.json"
        hf_tok.backend_tokenizer.save(str(tmp_path))

        from evaluate import evaluate_tokenizer
        metrics = evaluate_tokenizer(tmp_path, texts)

    return metrics


def ensure_summary_standard(tokenizer_dir: Path, test_path: Path, scale: str) -> Path:
    result_dir = ROOT / "results" / f"{dataset}_{scale}_test_standard"
    summary_path = result_dir / "summary.json"
    if summary_path.exists():
        return summary_path

    from evaluate import discover_tokenizer_paths, evaluate_tokenizer, load_jsonl_texts

    texts = load_jsonl_texts(test_path)
    summary: dict = {"test_corpus": str(test_path), "tokenizers": {}}
    for family, tok_path in discover_tokenizer_paths(tokenizer_dir).items():
        summary["tokenizers"][family] = evaluate_tokenizer(tok_path, texts)

    result_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def ensure_summary_superbpe(tokenizer_dir: Path, test_path: Path, scale: str) -> Path:
    result_dir = ROOT / "results" / f"{dataset}_{scale}_test_superbpe"
    summary_path = result_dir / "summary.json"
    if summary_path.exists():
        return summary_path

    from evaluate import load_jsonl_texts

    texts = load_jsonl_texts(test_path)
    metrics = _evaluate_superbpe_dir(tokenizer_dir, texts)

    summary: dict = {
        "test_corpus": str(test_path),
        "tokenizers": {"superbpe": metrics},
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


#%%

# DataFrame loader-

def load_metrics_df() -> pd.DataFrame:
    rows: list[dict] = []

    for source, base_dir in TOKENIZER_DIRS.items():
        if not base_dir.exists():
            print(f"Warning: directory not found, skipping — {base_dir}")
            continue

        pattern = SUPERBPE_PATTERN if source == "superbpe" else STANDARD_PATTERN
        tokenizer_dirs = sorted(base_dir.glob(pattern))
        if not tokenizer_dirs:
            print(f"Warning: no dirs matching '{pattern}' in {base_dir}")
            continue

        for tokenizer_dir in tokenizer_dirs:
            scale = _scale_from_dir(tokenizer_dir, source)
            train_chars = train_chars_for_scale(scale)

            if source == "superbpe":
                summary_path = ensure_summary_superbpe(tokenizer_dir, TEST_CORPUS, scale)
            else:
                summary_path = ensure_summary_standard(tokenizer_dir, TEST_CORPUS, scale)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))

            for family, metrics in summary["tokenizers"].items():
                row: dict = {
                    "source":           source,
                    "scale":            scale,
                    "train_chars":      train_chars,
                    "family":           family,
                    "compression_ratio": metrics["compression_ratio"],
                }
                for key, value in metrics["entropies_bits_per_token"].items():
                    row[f"entropy_{key}"] = value
                for key, value in metrics["entropy_rates_bits_per_char"].items():
                    row[f"rate_{key}"] = value
                rows.append(row)

    if not rows:
        raise FileNotFoundError("No tokenizer data found in either TOKENIZER_DIRS path.")


    all_families = FAMILIES + ["superbpe"]
    df = pd.DataFrame(rows)
    df["family"] = pd.Categorical(df["family"], categories=all_families, ordered=True)
    df["source"] = pd.Categorical(df["source"], categories=list(TOKENIZER_DIRS.keys()), ordered=True)
    return df.sort_values(["source", "scale", "family"]).reset_index(drop=True)


def use_log_train_axis(df: pd.DataFrame) -> bool:
    by_scale = df.groupby("scale", observed=True)["train_chars"].first()
    return bool(by_scale.notna().all() and by_scale.nunique() > 1)


df = load_metrics_df()
LOG_X = use_log_train_axis(df)
df

#%%

# Plot helpers


COLORS["superbpe"]  = "C4"
MARKERS["superbpe"] = "P"

all_families_in_df = list(df["family"].cat.categories)


def _x_label() -> str:
    return "Training chars (log scale)" if LOG_X else "Training scale ID"


def _plot_x(sub: pd.DataFrame):
    return sub["train_chars"] if LOG_X else sub["scale"].astype(str)


#%%

# Compression ratio

def plot_compression_ratio(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    n_scales = df["scale"].nunique()

    if n_scales > 1:
        for source in df["source"].cat.categories:
            for family in all_families_in_df:
                sub = df[(df["source"] == source) & (df["family"] == family)].sort_values("scale")
                if sub.empty:
                    continue
                label = f"{family}"
                ax.plot(
                    _plot_x(sub), sub["compression_ratio"],
                    marker=MARKERS.get(family, "x"),
                    color=COLORS.get(family, "gray"),
                    linestyle=SOURCE_LINESTYLES[source],
                    label=label,
                    linewidth=2,
                )
        if LOG_X:
            ax.set_xscale("log")
        ax.set_xlabel(_x_label())
    else:
        families_present = [f for f in all_families_in_df if not df[df["family"] == f].empty]
        sources = list(df["source"].cat.categories)
        x = range(len(families_present))
        width = 0.8 / len(sources)
        for i, source in enumerate(sources):
            sub = df[df["source"] == source].sort_values("family")
            vals = [
                sub.loc[sub["family"] == f, "compression_ratio"].iloc[0]
                if not sub.loc[sub["family"] == f].empty else 0
                for f in families_present
            ]
            offset = (i - len(sources) / 2 + 0.5) * width
            ax.bar(
                [p + offset for p in x], vals, width=width,
                color=[COLORS.get(f, "gray") for f in families_present],
                label=SOURCE_LABELS[source],
                alpha=0.7 if source == "superbpe" else 1.0,
                hatch="//" if source == "superbpe" else "",
            )
        ax.set_xticks(list(x))
        ax.set_xticklabels(families_present)
        ax.set_xlabel("Tokenizer family")

    ax.set_ylabel("Compression ratio (UTF-8 bytes / token)")
    ax.set_title("Compression ratio on held-out test")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()



plot_compression_ratio(df)
plt.show()

#%%

# k-gram entropies


def plot_kgram_entropies(df: pd.DataFrame) -> None:
    entropy_cols = [c for c in df.columns if c.startswith("entropy_H")]
    orders = sorted(entropy_cols, key=lambda c: int(re.search(r"H(\d+)", c).group(1)))
    n_scales = df["scale"].nunique()

    if n_scales > 1:
        fig, axes = plt.subplots(1, len(all_families_in_df),
                                 figsize=(3.2 * len(all_families_in_df), 4), sharey=True)
        if len(all_families_in_df) == 1:
            axes = [axes]
        for ax, family in zip(axes, all_families_in_df):
            for source in df["source"].cat.categories:
                sub = df[(df["source"] == source) & (df["family"] == family)].sort_values("scale")
                if sub.empty:
                    continue
                for col in orders:
                    k = col.replace("entropy_H", "")
                    ax.plot(
                        _plot_x(sub), sub[col],
                        marker="o",
                        linestyle=SOURCE_LINESTYLES[source],
                        label=f"H{k}",
                    )
            if LOG_X:
                ax.set_xscale("log")
            ax.set_title(family)
            ax.set_xlabel(_x_label())
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("Bits per token")
        axes[-1].legend(loc="upper left", fontsize=7)
        fig.suptitle("k-gram entropies (Fig. 4a style)", y=1.02)
    else:
        fig, axes = plt.subplots(1, len(all_families_in_df),
                                 figsize=(3.2 * len(all_families_in_df), 4), sharey=True)
        if len(all_families_in_df) == 1:
            axes = [axes]
        sources = list(df["source"].cat.categories)
        for ax, family in zip(axes, all_families_in_df):
            x = range(len(orders))
            width = 0.8 / len(sources)
            for i, source in enumerate(sources):
                sub = df[(df["source"] == source) & (df["family"] == family)]
                if sub.empty:
                    continue
                vals = [sub[col].iloc[0] for col in orders]
                offset = (i - len(sources) / 2 + 0.5) * width
                ax.bar([p + offset for p in x], vals, width=width,
                       label=SOURCE_LABELS[source],
                       alpha=0.7 if source == "superbpe" else 1.0,
                       hatch="//" if source == "superbpe" else "")
            ax.set_xticks(list(x))
            ax.set_xticklabels([col.replace("entropy_", "") for col in orders], fontsize=8)
            ax.set_title(family)
            ax.grid(True, alpha=0.3, axis="y")
        axes[0].set_ylabel("Bits per token")
        axes[-1].legend(fontsize=8)
        fig.suptitle("k-gram entropies by tokenizer family")
    fig.tight_layout()


plot_kgram_entropies(df)
plt.show()

#%%

# Entropy rates

def plot_entropy_rates(df: pd.DataFrame) -> None:
    rate_cols = [c for c in df.columns if c.startswith("rate_H")]
    orders = sorted(rate_cols, key=lambda c: int(re.search(r"H(\d+)", c).group(1)))
    n_scales = df["scale"].nunique()

    if n_scales > 1:
        fig, axes = plt.subplots(1, len(all_families_in_df),
                                 figsize=(3.2 * len(all_families_in_df), 4), sharey=True)
        if len(all_families_in_df) == 1:
            axes = [axes]
        for ax, family in zip(axes, all_families_in_df):
            for source in df["source"].cat.categories:
                sub = df[(df["source"] == source) & (df["family"] == family)].sort_values("scale")
                if sub.empty:
                    continue
                for col in orders:
                    k = col.replace("rate_H", "")
                    ax.plot(
                        _plot_x(sub), sub[col],
                        marker="o",
                        linestyle=SOURCE_LINESTYLES[source],
                        label=f"H{k}",
                    )
            if LOG_X:
                ax.set_xscale("log")
            ax.set_title(family)
            ax.set_xlabel(_x_label())
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("Bits per character")
        axes[-1].legend(loc="upper right", fontsize=7)
        fig.suptitle("Entropy rates H_k × tokens/char (Fig. 4b style)", y=1.02)
    else:
        fig, axes = plt.subplots(1, len(all_families_in_df),
                                 figsize=(3.2 * len(all_families_in_df), 4), sharey=True)
        if len(all_families_in_df) == 1:
            axes = [axes]
        sources = list(df["source"].cat.categories)
        for ax, family in zip(axes, all_families_in_df):
            x = range(len(orders))
            width = 0.8 / len(sources)
            for i, source in enumerate(sources):
                sub = df[(df["source"] == source) & (df["family"] == family)]
                if sub.empty:
                    continue
                vals = [sub[col].iloc[0] for col in orders]
                offset = (i - len(sources) / 2 + 0.5) * width
                ax.bar([p + offset for p in x], vals, width=width,
                       label=SOURCE_LABELS[source],
                       alpha=0.7 if source == "superbpe" else 1.0,
                       hatch="//" if source == "superbpe" else "")
            ax.set_xticks(list(x))
            ax.set_xticklabels([col.replace("rate_", "") for col in orders], fontsize=8)
            ax.set_title(family)
            ax.grid(True, alpha=0.3, axis="y")
        axes[0].set_ylabel("Bits per character")
        axes[-1].legend(fontsize=8)
        fig.suptitle("Entropy rates by tokenizer family")
    fig.tight_layout()


plot_entropy_rates(df)
plt.show()