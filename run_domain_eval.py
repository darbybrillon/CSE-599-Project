"""Cross-domain tokenizer evaluation driver (news / code / math).

Extends the c4-only pipeline to codeparrot (code) and gsm8k (math), and adds a
domain-mismatch matrix: every domain's tokenizer evaluated on every domain's
held-out test corpus. Reuses metrics from evaluate.py so numbers stay
comparable with the existing c4 results.

Pure CPU, no GPU. Standard (Hugging Face family) and SuperBPE tokenizers are
both loaded directly from their saved ``*.json`` files.

Examples
--------
In-domain sweep across all scales for code and math at vocab 16000:
    python run_domain_eval.py in-domain \
        --data-root drive_files/data --tokenizer-root drive_files/tokenizers \
        --domains codeparrot gsm8k --vocab 16000 --output drive_files/results

Domain-mismatch matrix at a fixed scale/vocab:
    python run_domain_eval.py mismatch \
        --data-root drive_files/data --tokenizer-root drive_files/tokenizers \
        --domains c4 codeparrot gsm8k --vocab 16000 --scale 106 \
        --output drive_files/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate import evaluate_tokenizer, load_jsonl_texts

# Standard Hugging Face family files we evaluate per tokenizer directory.
HF_FAMILIES = ("bpe", "unigram", "wordpiece", "wordlevel")


def hf_tokenizer_dir(tokenizer_root: Path, domain: str, vocab: int, scale: str) -> Path:
    return (
        tokenizer_root
        / domain
        / f"huggingface_tokenizers_{vocab}_{domain}_{scale}"
    )


def superbpe_tokenizer_file(
    tokenizer_root: Path, domain: str, vocab: int, scale: str
) -> Path:
    return (
        tokenizer_root
        / "SuperBPE Tokenizers"
        / domain
        / f"{vocab}_{domain}_{scale}"
        / "tokenizer.json"
    )


def test_corpus(data_root: Path, domain: str, scale: str) -> Path:
    return data_root / domain / f"{domain}_{scale}" / "test.jsonl"


def evaluate_all_families(
    tokenizer_root: Path,
    domain: str,
    vocab: int,
    scale: str,
    texts: list[str],
    *,
    include_superbpe: bool = True,
) -> dict[str, dict]:
    """Evaluate every available tokenizer family trained on ``domain``."""

    results: dict[str, dict] = {}

    hf_dir = hf_tokenizer_dir(tokenizer_root, domain, vocab, scale)
    for family in HF_FAMILIES:
        tok_path = hf_dir / f"{family}.json"
        if tok_path.is_file():
            results[family] = evaluate_tokenizer(tok_path, texts)

    if include_superbpe:
        sbpe_path = superbpe_tokenizer_file(tokenizer_root, domain, vocab, scale)
        if sbpe_path.is_file():
            results["superbpe"] = evaluate_tokenizer(sbpe_path, texts)

    if not results:
        raise FileNotFoundError(
            f"No tokenizers found for domain={domain} vocab={vocab} scale={scale}"
        )
    return results


def discover_scales(data_root: Path, domain: str) -> list[str]:
    domain_dir = data_root / domain
    prefix = f"{domain}_"
    scales = []
    for child in sorted(domain_dir.glob(f"{prefix}*")):
        if child.is_dir() and (child / "test.jsonl").is_file():
            scales.append(child.name[len(prefix) :])
    return scales


def run_in_domain(args: argparse.Namespace) -> dict:
    data_root = args.data_root
    tokenizer_root = args.tokenizer_root
    summary: dict = {"mode": "in-domain", "vocab": args.vocab, "domains": {}}

    for domain in args.domains:
        scales = args.scales or discover_scales(data_root, domain)
        domain_out: dict = {}
        for scale in scales:
            test_path = test_corpus(data_root, domain, scale)
            if not test_path.is_file():
                print(f"[skip] no test corpus: {test_path}")
                continue
            # Tokenizer trained on this domain/scale must exist to evaluate it.
            try:
                texts = load_jsonl_texts(test_path)
                metrics = evaluate_all_families(
                    tokenizer_root,
                    domain,
                    args.vocab,
                    scale,
                    texts,
                    include_superbpe=not args.no_superbpe,
                )
            except FileNotFoundError as exc:
                print(f"[skip] {exc}")
                continue
            domain_out[scale] = metrics
            cr = ", ".join(
                f"{fam}={m['compression_ratio']:.3f}" for fam, m in metrics.items()
            )
            print(f"[ok] {domain} scale={scale}: {cr}")
        summary["domains"][domain] = domain_out

    _write_summary(args.output, f"in_domain_{args.vocab}", summary)
    return summary


def run_mismatch(args: argparse.Namespace) -> dict:
    data_root = args.data_root
    tokenizer_root = args.tokenizer_root
    scale = args.scale
    if scale is None:
        raise SystemExit("mismatch mode requires --scale")

    summary: dict = {
        "mode": "mismatch",
        "vocab": args.vocab,
        "scale": scale,
        "matrix": {},
    }

    # Pre-load each domain's test corpus once.
    test_texts: dict[str, list[str]] = {}
    for test_domain in args.domains:
        path = test_corpus(data_root, test_domain, scale)
        if not path.is_file():
            print(f"[skip] no test corpus for {test_domain}: {path}")
            continue
        test_texts[test_domain] = load_jsonl_texts(path)

    for train_domain in args.domains:
        row: dict = {}
        for test_domain, texts in test_texts.items():
            try:
                metrics = evaluate_all_families(
                    tokenizer_root,
                    train_domain,
                    args.vocab,
                    scale,
                    texts,
                    include_superbpe=not args.no_superbpe,
                )
            except FileNotFoundError as exc:
                print(f"[skip] {exc}")
                continue
            row[test_domain] = metrics
            tag = "in-domain" if train_domain == test_domain else "cross"
            cr = ", ".join(
                f"{fam}={m['compression_ratio']:.3f}" for fam, m in metrics.items()
            )
            print(f"[ok] train={train_domain} test={test_domain} ({tag}): {cr}")
        summary["matrix"][train_domain] = row

    _write_summary(args.output, f"mismatch_{args.vocab}_{scale}", summary)
    return summary


def _write_summary(output: Path | None, name: str, summary: dict) -> None:
    if output is None:
        print(json.dumps(summary, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    out_path = output / f"{name}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data-root", type=Path, required=True)
        p.add_argument("--tokenizer-root", type=Path, required=True)
        p.add_argument("--domains", nargs="+", required=True)
        p.add_argument("--vocab", type=int, default=16000)
        p.add_argument("--output", type=Path, default=None)
        p.add_argument(
            "--no-superbpe",
            action="store_true",
            help="Skip SuperBPE tokenizers (standard families only).",
        )

    p_in = sub.add_parser("in-domain", help="Per-domain sweep across scales.")
    add_common(p_in)
    p_in.add_argument(
        "--scales",
        nargs="*",
        default=None,
        help="Scale codes (e.g. 105 106 107). Default: autodiscover.",
    )

    p_mm = sub.add_parser("mismatch", help="Train-domain x test-domain matrix.")
    add_common(p_mm)
    p_mm.add_argument(
        "--scale",
        required=True,
        help="Single scale code shared by all domains (e.g. 106).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "in-domain":
        run_in_domain(args)
    elif args.command == "mismatch":
        run_mismatch(args)


if __name__ == "__main__":
    main()
