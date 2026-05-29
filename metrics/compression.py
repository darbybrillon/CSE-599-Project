"""Compression-ratio metrics for tokenizers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def corpus_token_stats(texts: Sequence[str], token_counts: Sequence[int]) -> dict[str, float]:
    """Aggregate byte and token counts for a corpus."""

    if len(texts) != len(token_counts):
        raise ValueError("texts and token_counts must have the same length.")
    if not texts:
        raise ValueError("texts must be non-empty.")

    total_bytes = sum(len(text.encode("utf-8")) for text in texts)
    total_chars = sum(len(text) for text in texts)
    total_tokens = float(sum(token_counts))
    cr = total_bytes / total_tokens
    return {
        "num_documents": float(len(texts)),
        "num_tokens": total_tokens,
        "utf8_bytes": float(total_bytes),
        "num_chars": float(total_chars),
        "compression_ratio": cr,
        "tokens_per_char": total_tokens / total_chars if total_chars else 0.0,
        "bytes_per_token": cr,
    }


def compression_ratio(texts: Iterable[str], token_counts: Iterable[int]) -> float:
    """Return UTF-8 bytes per token (larger ⇒ more compressive tokenization)."""

    text_list = list(texts)
    count_list = list(token_counts)
    return corpus_token_stats(text_list, count_list)["compression_ratio"]
