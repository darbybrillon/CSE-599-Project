"""Empirical k-gram entropy metrics on token sequences."""

from __future__ import annotations

import math
from collections import Counter


def kgram_entropies(tokens: list[str], max_order: int = 5) -> dict[str, float]:
    """Compute empirical unigram and conditional k-gram entropies (bits/token).

    Unigram H1 uses the full sequence. For k >= 2, only positions with a full
    (k-1)-token context contribute (positions i = k-1 .. n-1).
    """

    if max_order < 1:
        raise ValueError("max_order must be >= 1.")
    n = len(tokens)
    if n == 0:
        raise ValueError("tokens must be non-empty.")

    results: dict[str, float] = {}

    unigram_counts = Counter(tokens)
    h1 = 0.0
    for count in unigram_counts.values():
        p = count / n
        h1 -= p * math.log2(p)
    results["H1"] = h1

    for order in range(2, max_order + 1):
        context_len = order - 1
        context_counts: Counter[tuple[str, ...]] = Counter()
        joint_counts: Counter[tuple[tuple[str, ...], str]] = Counter()

        for index in range(context_len, n):
            context = tuple(tokens[index - context_len : index])
            token = tokens[index]
            context_counts[context] += 1
            joint_counts[(context, token)] += 1

        evaluated = n - context_len
        if evaluated == 0:
            results[f"H{order}"] = float("nan")
            continue

        entropy_sum = 0.0
        for index in range(context_len, n):
            context = tuple(tokens[index - context_len : index])
            token = tokens[index]
            probability = joint_counts[(context, token)] / context_counts[context]
            entropy_sum -= math.log2(probability)
        results[f"H{order}"] = entropy_sum / evaluated

    return results


def entropy_rates(
    entropies: dict[str, float],
    *,
    tokens_per_char: float,
) -> dict[str, float]:
    """Convert bits/token entropies to bits/character entropy rates."""

    if tokens_per_char <= 0:
        raise ValueError("tokens_per_char must be positive.")
    return {key: value * tokens_per_char for key, value in entropies.items()}
