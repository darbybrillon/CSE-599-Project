"""Evaluate tokenizer compression ratio and k-gram entropies on a test corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

from data import LEGACY_EOS_TOKEN, SPECIAL_TOKENS
from metrics.compression import corpus_token_stats
from metrics.entropy import entropy_rates, kgram_entropies

HF_FAMILY_FILES = ("bpe.json", "unigram.json", "wordpiece.json", "wordlevel.json")


def load_jsonl_texts(path: Path, text_key: str = "text") -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            texts.append(json.loads(line)[text_key])
    if not texts:
        raise ValueError(f"No rows found in {path}")
    return texts


def _tokenizer_vocab_strings(tokenizer: Tokenizer) -> set[str]:
    vocab = tokenizer.get_vocab()
    return set(vocab.keys())


def align_text_to_tokenizer(text: str, tokenizer: Tokenizer) -> str:
    """Map preprocessing special tokens to those expected by a saved checkpoint."""

    vocab = _tokenizer_vocab_strings(tokenizer)
    canonical_eos = SPECIAL_TOKENS["eos_token"]
    aligned = text
    if canonical_eos in aligned and canonical_eos not in vocab and LEGACY_EOS_TOKEN in vocab:
        aligned = aligned.replace(canonical_eos, LEGACY_EOS_TOKEN)
    elif LEGACY_EOS_TOKEN in aligned and LEGACY_EOS_TOKEN not in vocab and canonical_eos in vocab:
        aligned = aligned.replace(LEGACY_EOS_TOKEN, canonical_eos)
    return aligned


def encode_corpus(tokenizer: Tokenizer, texts: list[str]) -> tuple[list[int], list[str], list[int]]:
    """Return per-document token counts and one concatenated token-string sequence."""

    per_doc_counts: list[int] = []
    all_tokens: list[str] = []
    for text in texts:
        aligned = align_text_to_tokenizer(text, tokenizer)
        encoding = tokenizer.encode(aligned)
        per_doc_counts.append(len(encoding.ids))
        all_tokens.extend(encoding.tokens)
    return per_doc_counts, all_tokens, [len(t.encode("utf-8")) for t in texts]


def evaluate_tokenizer(
    tokenizer_path: Path,
    texts: list[str],
    *,
    max_order: int = 5,
) -> dict[str, object]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    per_doc_counts, token_strings, _ = encode_corpus(tokenizer, texts)
    stats = corpus_token_stats(texts, per_doc_counts)
    # Use tokens_per_char from the same held-out test corpus as the entropy
    # computation, not from any training-set estimate.
    entropies = kgram_entropies(token_strings, max_order=max_order)
    rates = entropy_rates(entropies, tokens_per_char=stats["tokens_per_char"])
    return {
        **stats,
        "entropies_bits_per_token": entropies,
        "entropy_rates_bits_per_char": rates,
        "tokenizer": str(tokenizer_path),
    }


def discover_tokenizer_paths(tokenizer_arg: Path) -> dict[str, Path]:
    if tokenizer_arg.is_file():
        return {tokenizer_arg.stem: tokenizer_arg}
    if not tokenizer_arg.is_dir():
        raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_arg}")

    discovered: dict[str, Path] = {}
    for filename in HF_FAMILY_FILES:
        candidate = tokenizer_arg / filename
        if candidate.is_file():
            discovered[candidate.stem] = candidate
    if not discovered:
        superbpe_candidate = tokenizer_arg / "tokenizer.json"
        if superbpe_candidate.is_file():
            return {"superbpe": superbpe_candidate}
        json_files = sorted(tokenizer_arg.glob("*.json"))
        if len(json_files) == 1:
            path = json_files[0]
            return {path.stem: path}
        raise FileNotFoundError(
            f"No Hugging Face tokenizer JSON files found in {tokenizer_arg}. "
            f"Expected one of: {', '.join(HF_FAMILY_FILES)}"
        )
    return discovered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate tokenizer compression ratio and k-gram entropies."
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="Path to a tokenizer .json file or a directory containing family JSON files.",
    )
    parser.add_argument("--test", type=Path, required=True, help="Test corpus JSONL path.")
    parser.add_argument("--text-key", default="text", help="JSONL field with input text.")
    parser.add_argument("--max-order", type=int, default=5, help="Maximum k-gram order.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Write metrics JSON here. For a tokenizer directory, writes one file per family "
        "plus summary.json when --output is a directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    texts = load_jsonl_texts(args.test, text_key=args.text_key)
    tokenizer_paths = discover_tokenizer_paths(args.tokenizer)

    results: dict[str, object] = {
        "test_corpus": str(args.test),
        "tokenizers": {},
    }

    for family, path in sorted(tokenizer_paths.items()):
        metrics = evaluate_tokenizer(path, texts, max_order=args.max_order)
        results["tokenizers"][family] = metrics  # type: ignore[index]

        if args.output is not None:
            if args.output.suffix == ".json" and len(tokenizer_paths) == 1:
                out_path = args.output
            else:
                args.output.mkdir(parents=True, exist_ok=True)
                out_path = args.output / f"{family}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)

    if args.output is not None and len(tokenizer_paths) > 1 and args.output.is_dir():
        summary_path = args.output / "summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
