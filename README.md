# Entropy & Compression Based Evaluation on LLM Tokenizers

This repository evaluates how tokenizer training scale, tokenizer family, and
training/test-domain mismatch affect compression and empirical k-gram entropy. [View the Full Technical Report](./Entropy-Eval.pdf).

The core pipeline is:

1. sample and preprocess datasets from Hugging Face,
2. train tokenizer checkpoints,
3. evaluate each checkpoint on held-out JSONL test corpora,
4. plot compression ratios, k-gram entropies, and entropy rates.

The standard tokenizer families are Hugging Face `tokenizers` BPE, Unigram,
WordPiece, and WordLevel. The SuperBPE experiments use the external
`PythonNut/superbpe` implementation and then convert SuperBPE checkpoints into
Hugging Face-compatible `tokenizer.json` files for evaluation.

## Repository layout

```text
.
|-- data.py                              # dataset sampling/preprocessing CLI
|-- BuildVocab.py                        # standard tokenizer training helper
|-- evaluate.py                          # compression + entropy evaluation CLI
|-- metrics/
|   |-- compression.py                   # UTF-8 bytes/token statistics
|   `-- entropy.py                       # empirical H1...Hk calculations
|-- SuperBPE.py                          # notes/helper for SuperBPE conversion
|-- SuperBPE_Compression_Entropy_Plots.py# plotting/evaluation helper for SuperBPE
|-- plot_entropy_results.ipynb           # C4 compression/entropy plotting notebook
|-- domain_mismatch.ipynb                # domain-mismatch experiment notebook
|-- data/                                # checked-in sample/preprocessed splits
|-- tokenizers/                          # expanded standard tokenizer checkpoints
|-- SuperBPE Tokenizers/                 # expanded SuperBPE checkpoints
|-- results/                             # expanded metric outputs
`-- *.zip                                # archived checkpoints/results
```

## Environment

Use Python 3.10 or newer.

```bash
cd CSE-599-Project
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install transformers jupyter
```

For SuperBPE experiments, install the external SuperBPE package in the same
environment:

```bash
git clone --recurse-submodules https://github.com/PythonNut/superbpe.git
cd superbpe
python -m pip install -r requirements.txt
python -m pip install -e .
cd ..
```

The dataset-generation commands stream from Hugging Face and require network
access. C4 and CodeParrot are large datasets; the largest `10^8` character
experiments can take substantial time and disk space.

## Data generation

`data.py` creates deterministic preprocessed splits when run with the same seed.
It normalizes text with NFKC, lowercases it, whitespace-tokenizes it, wraps each
sample with `<s>` and `</s>`, then writes `train`, `validation`, and `test`
splits. The default split ratio is 80/10/10.

The project uses scale IDs such as `103`, `104`, ..., `108` to mean full-corpus
character budgets of `10^3`, `10^4`, ..., `10^8` before splitting.

Generate C4 scales:

```bash
for exp in 3 4 5 6 7 8; do
  scale="10${exp}"
  chars=$((10 ** exp))
  python data.py c4 \
    --full-dataset-chars "$chars" \
    --output-dir "data/c4_${scale}" \
    --format jsonl \
    --seed 13
done
```

Generate CodeParrot scales:

```bash
for exp in 3 4 5 6 7 8; do
  scale="10${exp}"
  chars=$((10 ** exp))
  python data.py codeparrot \
    --full-dataset-chars "$chars" \
    --output-dir "data/codeparrot_${scale}" \
    --format jsonl \
    --seed 13
done
```

Generate GSM8K scales used in this repository:

```bash
for exp in 3 4 5 6; do
  scale="10${exp}"
  chars=$((10 ** exp))
  python data.py gsm8k \
    --full-dataset-chars "$chars" \
    --output-dir "data/gsm8k_${scale}" \
    --format jsonl \
    --seed 13
done
```

For a fixed out-of-domain holdout, generate all sampled text as the test split
by setting train and validation ratios to zero. Example:

```bash
python data.py c4 \
  --config tr \
  --full-dataset-chars 11000000 \
  --train-ratio 0 \
  --val-ratio 0 \
  --test-ratio 1 \
  --output-dir data/domain_mismatch/c4_tr \
  --format jsonl \
  --seed 13
```

## Standard tokenizer training

`BuildVocab.py` trains BPE, Unigram, WordPiece, and WordLevel tokenizers using
the Hugging Face `tokenizers` library. The script currently has experiment lists
near the bottom of the file; edit `datasets` and `vocab_sizes` there before
running it.

For example, to train all standard C4 checkpoints, set:

```python
datasets = ["c4_103", "c4_104", "c4_105", "c4_106", "c4_107", "c4_108"]
vocab_sizes = [16000, 64000]
```

Then run:

```bash
python BuildVocab.py
```

Outputs are written as:

```text
tokenizers/huggingface_tokenizers_<vocab_size>_<dataset_scale>/
```

Each output directory contains `bpe.json`, `unigram.json`, `wordpiece.json`, and
`wordlevel.json`, plus model-specific vocab/merge files.

Some historical checked-in/expanded paths use `16k` instead of `16000` in the
directory name. The contents are the same format; adjust paths in commands to
match the checkpoint directory you are using.

## SuperBPE tokenizer training

SuperBPE is trained with the external `train_tokenizer` module from
`PythonNut/superbpe`. SuperBPE expects plain text corpora, so convert each JSONL
split to `.txt` first. `data_txt_conversion.py` documents the conversion logic,
but it contains historical Windows-local paths; reproduce it locally with the
same operation:

```bash
mkdir -p data_txt/c4_103/train
python - <<'PY'
import json
from pathlib import Path

for dataset_dir in Path("data").glob("*_*"):
    if not dataset_dir.is_dir():
        continue
    for split in ("train", "validation", "test"):
        src = dataset_dir / f"{split}.jsonl"
        if not src.exists():
            continue
        dst = Path("data_txt") / dataset_dir.name
        dst = dst / "train" / f"{split}.txt" if split == "train" else dst / f"{split}.txt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open(encoding="utf-8") as infile, dst.open("w", encoding="utf-8") as outfile:
            for line in infile:
                if line.strip():
                    outfile.write(json.loads(line)["text"] + "\n")
PY
```

Train a SuperBPE checkpoint:

```bash
python -m train_tokenizer \
  --output_dir "SuperBPE Tokenizers/c4/16000_c4_103" \
  --corpus_dir "data_txt/c4_103/train" \
  --vocab_size 16000 \
  --regex_string "[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
```

After training, construct Hugging Face tokenizer files inside each SuperBPE
checkpoint directory:

```bash
python - <<'PY'
from pathlib import Path
from superbpe.utils import construct_hf_tokenizer

roots = [Path("SuperBPE Tokenizers")]
for pattern in ("*_*_*", "*/*_*_*"):
    for ckpt in roots[0].glob(pattern):
        if ckpt.is_dir():
            construct_hf_tokenizer(str(ckpt))
PY
```

## Evaluation

`evaluate.py` accepts either a single tokenizer JSON file or a directory. For a
standard tokenizer directory it evaluates all four families and writes one JSON
file per family plus `summary.json`.

Evaluate one standard checkpoint:

```bash
python evaluate.py \
  --tokenizer tokenizers/huggingface_tokenizers_16000_c4_103 \
  --test data/c4_103/test.jsonl \
  --output results/c4_103_test
```

Evaluate one SuperBPE checkpoint:

```bash
python evaluate.py \
  --tokenizer "SuperBPE Tokenizers/c4/16000_c4_103" \
  --test data/c4_103/test.jsonl \
  --output results/c4_103_test_superbpe/superbpe.json
```

Full standard C4 evaluation loop:

```bash
for scale in 103 104 105 106 107 108; do
  python evaluate.py \
    --tokenizer "tokenizers/huggingface_tokenizers_16000_c4_${scale}" \
    --test "data/c4_${scale}/test.jsonl" \
    --output "results/c4_${scale}_test_standard"
done
```

Repeat the loop with `64000` for 64k-vocabulary runs and with
`codeparrot_<scale>` or `gsm8k_<scale>` paths for the other datasets.

The reported metrics are:

- `compression_ratio` / `bytes_per_token`: total UTF-8 bytes divided by total
  tokens, where larger values mean more bytes represented per token.
- `tokens_per_char`: total tokens divided by Python character count.
- `entropies_bits_per_token`: empirical `H1` through `H5`.
- `entropy_rates_bits_per_char`: each entropy multiplied by `tokens_per_char`.

## Plotting and saved results

Use `plot_entropy_results.ipynb` for the standard C4 compression/entropy plots.
Use `SuperBPE_Compression_Entropy_Plots.py` for side-by-side standard vs
SuperBPE plots. That script contains historical Windows paths in `TOKENIZER_DIRS`;
update those paths to local directories before running it.

Use `domain_mismatch.ipynb` for the out-of-domain experiment. The notebook trains
or loads C4/CodeParrot-scale checkpoints, evaluates them on fixed C4 English,
Turkish, and Chinese holdouts, writes JSON outputs under
`results/domain_mismatch/`, and saves:

```text
results/domain_mismatch/domain_mismatch_results.csv
results/domain_mismatch/fig5_compression_mismatch.png
results/domain_mismatch/fig6_entropy_mismatch.png
```

Archived outputs are included for reproduction checks:

- `results.zip`: saved C4, CodeParrot, GSM8K, and domain-mismatch metric JSON,
  CSV, and figure outputs.
- `c4 Tokenizers.zip`, `codeparrot Tokenizers.zip`, `gsm8k Tokenizers.zip`:
  standard tokenizer checkpoints.
- `SuperBPE Tokenizers.zip`: SuperBPE checkpoints for C4, CodeParrot, and GSM8K.

To inspect or restore them:

```bash
unzip -l results.zip
unzip results.zip
unzip "c4 Tokenizers.zip"
unzip "SuperBPE Tokenizers.zip"
```

## Reproduction checklist

1. Create the Python environment and install requirements.
2. Generate `data/<dataset>_<scale>/{train,validation,test}.jsonl` with
   `data.py` for every dataset/scale used in the experiment.
3. Train standard tokenizer checkpoints with `BuildVocab.py`, or unzip the
   archived standard tokenizer files.
4. Train SuperBPE checkpoints with `python -m train_tokenizer`, or unzip the
   archived SuperBPE checkpoints.
5. Convert SuperBPE checkpoints to Hugging Face tokenizer files with
   `construct_hf_tokenizer`.
6. Run `evaluate.py` for each tokenizer/test-corpus pair.
7. Run the plotting notebook/script to regenerate figures and aggregated CSVs.

## Known reproducibility notes

- `BuildVocab.py`, `data_txt_conversion.py`, and
  `SuperBPE_Compression_Entropy_Plots.py` include historical hard-coded
  experiment selections or Windows-local paths. Edit those paths/lists before
  rerunning full experiments on another machine.
- The checked-in working tree includes only a subset of expanded data,
  tokenizers, and results. The zip archives contain many of the saved artifacts
  needed to compare against the full reported runs.
- The largest C4/CodeParrot runs require streaming large Hugging Face datasets.
  Runtime may vary, and exact samples depend on Hugging Face dataset revisions
  available at run time.
- `evaluate.py` aligns legacy `<s/>` and canonical `</s>` EOS tokens when a
  saved checkpoint uses one spelling and the test corpus uses the other.
