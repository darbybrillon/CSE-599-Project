"""Dataset loading and preprocessing utilities.

The pipeline here intentionally samples a single pool from each source dataset
before creating train/validation/test splits. Tokenizer training code can then
train only on the returned train split and evaluate on val/test.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
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
    """Apply NFKC Unicode normalization and lowercase text."""

    return unicodedata.normalize("NFKC", text).lower()


def whitespace_pretokenize(text: str) -> list[str]:
    """Basic whitespace pre-tokenization."""

    return normalize_text(text).split()


def preprocess_text(text: str) -> dict[str, object]:
    """Normalize, lowercase, whitespace pre-tokenize, and add boundary tokens."""

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


def _row_char_count(row: Mapping[str, object]) -> int:
    return len(str(row["text"]))


def _tokens_to_row(tokens: Sequence[str]) -> dict[str, object]:
    return {
        "text": " ".join(tokens),
        "tokens": list(tokens),
    }


def _take_tokens_by_chars(tokens: Sequence[str], char_budget: int) -> list[str]:
    selected: list[str] = []
    total_chars = 0
    for token in tokens:
        added_chars = len(token) if not selected else len(token) + 1
        if selected and total_chars + added_chars > char_budget:
            break
        selected.append(token)
        total_chars += added_chars
        if total_chars >= char_budget:
            break
    return selected


def _partition_tokens_by_chars(
    tokens: Sequence[str],
    char_targets: Sequence[int],
) -> list[list[str]]:
    partitions: list[list[str]] = [[] for _ in char_targets]
    partition_index = 0
    current_chars = 0

    for token in tokens:
        while (
            partition_index < len(char_targets) - 1
            and current_chars >= char_targets[partition_index]
        ):
            partition_index += 1
            current_chars = 0

        added_chars = len(token) if current_chars == 0 else len(token) + 1
        partitions[partition_index].append(token)
        current_chars += added_chars

    return partitions


def _materialize_preprocessed_sample_by_chars(
    dataset: "Dataset | IterableDataset",
    full_dataset_chars: int,
    text_columns: Sequence[str],
    seed: int,
    shuffle_buffer_size: int,
) -> "Dataset":
    Dataset, _, IterableDataset, _ = _require_datasets()
    if isinstance(dataset, IterableDataset):
        source = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    else:
        source = dataset.shuffle(seed=seed)

    rows: list[dict[str, object]] = []
    total_chars = 0
    for example in source:
        row = preprocess_text(_join_text_columns(example, text_columns))
        rows.append(row)
        total_chars += len(row["text"])
        if total_chars >= full_dataset_chars:
            break

    if total_chars < full_dataset_chars:
        raise ValueError(
            f"Requested full_dataset_chars={full_dataset_chars}, but only found "
            f"{total_chars} preprocessed characters."
        )

    all_tokens = [
        token
        for row in rows
        for token in row["tokens"]
    ]
    budgeted_tokens = _take_tokens_by_chars(all_tokens, full_dataset_chars)
    return Dataset.from_list([_tokens_to_row(budgeted_tokens)])


def _split_sample_by_chars(
    dataset: "Dataset",
    split_ratios: Sequence[float],
) -> "DatasetDict":
    Dataset, DatasetDict, _, _ = _require_datasets()
    train_ratio, val_ratio, test_ratio = _validate_split_ratios(split_ratios)
    all_tokens = [
        token
        for row in dataset
        for token in row["tokens"]
    ]
    total_chars = len(" ".join(all_tokens))
    train_target = int(total_chars * train_ratio)
    val_target = int(total_chars * val_ratio)
    test_target = total_chars - train_target - val_target
    train_tokens, val_tokens, test_tokens = _partition_tokens_by_chars(
        all_tokens,
        (train_target, val_target, test_target),
    )

    def make_split(tokens: Sequence[str]) -> "Dataset":
        if not tokens:
            return Dataset.from_dict({"text": [], "tokens": []})
        return Dataset.from_list([_tokens_to_row(tokens)])

    return DatasetDict(
        {
            "train": make_split(train_tokens),
            "validation": make_split(val_tokens),
            "test": make_split(test_tokens),
        }
    )


def _split_sample_by_records(
    dataset: "Dataset",
    split_ratios: Sequence[float],
) -> "DatasetDict":
    _, DatasetDict, _, _ = _require_datasets()
    train_ratio, val_ratio, _ = _validate_split_ratios(split_ratios)
    total_chars = sum(_row_char_count(row) for row in dataset)
    train_target = int(total_chars * train_ratio)
    val_target = int(total_chars * val_ratio)

    train_end = 0
    train_chars = 0
    while train_end < len(dataset) and train_chars < train_target:
        train_chars += _row_char_count(dataset[train_end])
        train_end += 1

    val_end = train_end
    val_chars = 0
    while val_end < len(dataset) and val_chars < val_target:
        val_chars += _row_char_count(dataset[val_end])
        val_end += 1

    return DatasetDict(
        {
            "train": dataset.select(range(0, train_end)) if train_end > 0 else dataset.select([]),
            "validation": dataset.select(range(train_end, val_end))
            if train_end < val_end
            else dataset.select([]),
            "test": dataset.select(range(val_end, len(dataset)))
            if val_end < len(dataset)
            else dataset.select([]),
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
    full_dataset_chars: int,
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
        full_dataset_chars: Preprocessed character budget before splitting.
        split_ratios: Train/validation/test proportions. Values are normalized.
        seed: Random seed used for dataset shuffling.
        streaming: Whether to stream from Hugging Face. This is useful for C4
            and CodeParrot-scale datasets.
        shuffle_buffer_size: Streaming shuffle buffer size.
        path/config/split/text_columns: Optional overrides for the default spec.
    """

    if full_dataset_chars <= 0:
        raise ValueError("full_dataset_chars must be positive.")

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
    sampled = _materialize_preprocessed_sample_by_chars(
        dataset,
        full_dataset_chars=full_dataset_chars,
        text_columns=spec.text_columns,
        seed=seed,
        shuffle_buffer_size=shuffle_buffer_size,
    )
    return _split_sample_by_chars(sampled, split_ratios)


def load_all_preprocessed_datasets(
    *,
    full_dataset_chars: int,
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
            full_dataset_chars=full_dataset_chars,
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
    parser.add_argument("--full-dataset-chars", type=int, required=True)
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
        full_dataset_chars=args.full_dataset_chars,
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
