"""Dataset loading and preprocessing utilities.

The pipeline here intentionally samples a single pool from each source dataset
before creating train/validation/test splits. Tokenizer training code can then
train only on the returned train split and evaluate on val/test.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence
import unicodedata

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict, IterableDataset


SPECIAL_TOKENS = {
    "unk_token": "<UNK>",
    "pad_token": "<PAD>",
    "bos_token": "<s>",
    "eos_token": "</s>",
}

DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)


@dataclass(frozen=True)
class DatasetSpec:
    """Hugging Face dataset location and fields used to build text examples."""

    path: str
    name: str | None = None
    split: str = "train"
    text_columns: tuple[str, ...] = ("text",)


DATASET_SPECS: dict[str, DatasetSpec] = {
    "c4": DatasetSpec(
        path="allenai/c4",
        name="en",
        split="train",
        text_columns=("text",),
    ),
    "gsm8k": DatasetSpec(
        path="gsm8k",
        name="main",
        split="train",
        text_columns=("question", "answer"),
    ),
    "codeparrot": DatasetSpec(
        path="codeparrot/codeparrot-clean",
        split="train",
        text_columns=("content", "code", "text"),
    ),
}


def _require_datasets() -> tuple[Any, Any, Any, Any]:
    try:
        from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
    except ImportError as exc:
        raise ImportError(
            "The Hugging Face 'datasets' package is required for dataset loading. "
            "Install it with: pip install datasets"
        ) from exc
    return Dataset, DatasetDict, IterableDataset, load_dataset


def normalize_text(text: str) -> str:
    """Apply NFKC Unicode normalization."""

    return unicodedata.normalize("NFKC", text)


def whitespace_pretokenize(text: str) -> list[str]:
    """Basic whitespace pre-tokenization."""

    return normalize_text(text).split()


def preprocess_text(text: str) -> dict[str, object]:
    """Normalize, whitespace pre-tokenize, and add sentence boundary tokens."""

    tokens = [
        SPECIAL_TOKENS["bos_token"],
        *whitespace_pretokenize(text),
        SPECIAL_TOKENS["eos_token"],
    ]
    return {
        "text": " ".join(tokens),
        "tokens": tokens,
    }


def _join_text_columns(
    example: Mapping[str, object],
    text_columns: Sequence[str],
) -> str:
    parts: list[str] = []
    for column in text_columns:
        value = example.get(column)
        if value is None:
            continue
        parts.append(str(value))
    if not parts:
        available = ", ".join(sorted(example.keys()))
        requested = ", ".join(text_columns)
        raise ValueError(
            f"None of the requested text columns were found. "
            f"Requested: {requested}. Available: {available}."
        )
    return "\n".join(parts)


def _validate_split_ratios(split_ratios: Sequence[float]) -> tuple[float, float, float]:
    if len(split_ratios) != 3:
        raise ValueError("split_ratios must contain train, validation, and test ratios.")
    train_ratio, val_ratio, test_ratio = split_ratios
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("split_ratios cannot contain negative values.")
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("At least one split ratio must be positive.")
    return train_ratio / total, val_ratio / total, test_ratio / total


def _split_counts(total_size: int, split_ratios: Sequence[float]) -> tuple[int, int, int]:
    train_ratio, val_ratio, _ = _validate_split_ratios(split_ratios)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    return train_size, val_size, test_size


def _materialize_sample(
    dataset: "Dataset | IterableDataset",
    sample_size: int | None,
    seed: int,
    shuffle_buffer_size: int,
) -> "Dataset":
    Dataset, _, IterableDataset, _ = _require_datasets()
    if isinstance(dataset, IterableDataset):
        shuffled = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
        if sample_size is None:
            raise ValueError(
                "full_dataset_size is required when streaming=True because streaming "
                "datasets do not have a finite in-memory length."
            )
        return Dataset.from_list(list(islice(shuffled, sample_size)))

    if sample_size is None:
        sample_size = len(dataset)
    if sample_size > len(dataset):
        raise ValueError(
            f"Requested full_dataset_size={sample_size}, but dataset only has "
            f"{len(dataset)} rows."
        )
    return dataset.shuffle(seed=seed).select(range(sample_size))


def _preprocess_dataset(dataset: "Dataset", text_columns: Sequence[str]) -> "Dataset":
    def preprocess_example(example: Mapping[str, object]) -> dict[str, object]:
        return preprocess_text(_join_text_columns(example, text_columns))

    return dataset.map(preprocess_example, remove_columns=dataset.column_names)


def _split_sample(
    dataset: "Dataset",
    split_ratios: Sequence[float],
) -> "DatasetDict":
    _, DatasetDict, _, _ = _require_datasets()
    train_size, val_size, test_size = _split_counts(len(dataset), split_ratios)
    train_end = train_size
    val_end = train_size + val_size
    return DatasetDict(
        {
            "train": dataset.select(range(0, train_end)),
            "validation": dataset.select(range(train_end, val_end)),
            "test": dataset.select(range(val_end, val_end + test_size)),
        }
    )


def resolve_dataset_spec(
    dataset_name: str,
    *,
    path: str | None = None,
    config: str | None = None,
    split: str | None = None,
    text_columns: Sequence[str] | None = None,
) -> DatasetSpec:
    try:
        spec = DATASET_SPECS[dataset_name]
    except KeyError as exc:
        known = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unknown dataset '{dataset_name}'. Known datasets: {known}.") from exc

    return replace(
        spec,
        path=path or spec.path,
        name=config if config is not None else spec.name,
        split=split or spec.split,
        text_columns=tuple(text_columns) if text_columns is not None else spec.text_columns,
    )


def load_preprocessed_dataset(
    dataset_name: str,
    *,
    full_dataset_size: int | None,
    split_ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    seed: int = 13,
    streaming: bool = True,
    shuffle_buffer_size: int = 10_000,
    path: str | None = None,
    config: str | None = None,
    split: str | None = None,
    text_columns: Sequence[str] | None = None,
) -> "DatasetDict":
    """Load, sample, preprocess, and split one dataset.

    Args:
        dataset_name: One of ``c4``, ``gsm8k``, or ``codeparrot``.
        full_dataset_size: Number of source examples to sample before splitting.
        split_ratios: Train/validation/test proportions. Values are normalized.
        seed: Random seed used for dataset shuffling.
        streaming: Whether to stream from Hugging Face. This is useful for C4
            and CodeParrot-scale datasets.
        shuffle_buffer_size: Streaming shuffle buffer size.
        path/config/split/text_columns: Optional overrides for the default spec.
    """

    if full_dataset_size is not None and full_dataset_size <= 0:
        raise ValueError("full_dataset_size must be positive when provided.")

    _, _, _, load_dataset = _require_datasets()
    spec = resolve_dataset_spec(
        dataset_name,
        path=path,
        config=config,
        split=split,
        text_columns=text_columns,
    )

    dataset = load_dataset(
        spec.path,
        spec.name,
        split=spec.split,
        streaming=streaming,
    )
    sampled = _materialize_sample(
        dataset,
        sample_size=full_dataset_size,
        seed=seed,
        shuffle_buffer_size=shuffle_buffer_size,
    )
    preprocessed = _preprocess_dataset(sampled, spec.text_columns)
    return _split_sample(preprocessed, split_ratios)


def load_all_preprocessed_datasets(
    *,
    full_dataset_size: int,
    dataset_names: Sequence[str] = tuple(DATASET_SPECS),
    split_ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
    seed: int = 13,
    streaming: bool = True,
    shuffle_buffer_size: int = 10_000,
) -> dict[str, "DatasetDict"]:
    """Load every requested dataset with the same sample size and split ratios."""

    return {
        dataset_name: load_preprocessed_dataset(
            dataset_name,
            full_dataset_size=full_dataset_size,
            split_ratios=split_ratios,
            seed=seed,
            streaming=streaming,
            shuffle_buffer_size=shuffle_buffer_size,
        )
        for dataset_name in dataset_names
    }


def save_dataset_dict(
    dataset: "DatasetDict",
    output_dir: str | Path,
    *,
    file_format: str = "jsonl",
) -> None:
    """Persist train/validation/test splits as jsonl, txt, or Hugging Face arrow."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if file_format == "arrow":
        dataset.save_to_disk(str(output_path))
        return

    if file_format not in {"jsonl", "txt"}:
        raise ValueError("file_format must be one of: jsonl, txt, arrow.")

    suffix = ".jsonl" if file_format == "jsonl" else ".txt"
    for split_name, split_dataset in dataset.items():
        split_path = output_path / f"{split_name}{suffix}"
        with split_path.open("w", encoding="utf-8") as handle:
            for row in split_dataset:
                if file_format == "jsonl":
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                else:
                    handle.write(row["text"] + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare tokenizer evaluation datasets.")
    parser.add_argument("dataset", choices=sorted(DATASET_SPECS))
    parser.add_argument("--full-dataset-size", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--format",
        choices=("jsonl", "txt", "arrow"),
        default="jsonl",
        help="Output format. jsonl keeps text and tokens; txt writes text only.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_SPLIT_RATIOS[0])
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_SPLIT_RATIOS[1])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_SPLIT_RATIOS[2])
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--shuffle-buffer-size", type=int, default=10_000)
    parser.add_argument("--path", help="Override the Hugging Face dataset path.")
    parser.add_argument("--config", help="Override the Hugging Face dataset config.")
    parser.add_argument("--split", help="Override the Hugging Face source split.")
    parser.add_argument(
        "--text-columns",
        nargs="+",
        help="Override source columns to concatenate before preprocessing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_preprocessed_dataset(
        args.dataset,
        full_dataset_size=args.full_dataset_size,
        split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        seed=args.seed,
        streaming=not args.no_streaming,
        shuffle_buffer_size=args.shuffle_buffer_size,
        path=args.path,
        config=args.config,
        split=args.split,
        text_columns=args.text_columns,
    )
    save_dataset_dict(dataset, args.output_dir, file_format=args.format)


if __name__ == "__main__":
    main()
