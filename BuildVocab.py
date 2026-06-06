from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.models import BPE, Unigram, WordPiece, WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, WordLevelTrainer, UnigramTrainer
from datasets import load_dataset
import sentencepiece as sp
import tiktoken
import os

#tiktoken_encoding = tiktoken.get_encoding(train_data)

def batch_iterator(dataset, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]["text"]


def build_vocab_huggingface(vocab_size: int,
                            train_data,
                            min_count: int,
                            special_tokens: list[str]
                            ) -> dict:

    tokenizers = {}

    configs = {"bpe": (BPE(unk_token="<UNK>"), BpeTrainer(vocab_size=vocab_size, min_frequency=min_count, special_tokens=special_tokens)),
               "unigram": (Unigram(), UnigramTrainer(vocab_size=vocab_size, special_tokens=special_tokens, unk_token="<UNK>")),
               "wordpiece": (WordPiece(unk_token="<UNK>"), WordPieceTrainer(vocab_size=vocab_size, min_frequency=min_count, special_tokens=special_tokens)),
               "wordlevel": (WordLevel(unk_token="<UNK>"), WordLevelTrainer(vocab_size=vocab_size, min_frequency=min_count, special_tokens=special_tokens))}

    for name, (model, trainer) in configs.items():
        tokenizer = Tokenizer(model)
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train_from_iterator(batch_iterator(train_data), trainer)
        tokenizers[name] = tokenizer

    return tokenizers

def build_vocab_sentencepiece(vocab_size: int,
                              filename: str,
                              special_tokens: list[str],
                              model: str="bpe" # or unigram
                              ) -> tuple:
    sp.SentencePieceTrainer.Train(input=filename, #one-sentence-per-line raw corpus file
                                  model_prefix="sp",
                                  vocab_size=vocab_size,
                                  model_type=model,
                                  user_defined_symbols=special_tokens)
    return sp.model, sp.vocab


def save_tokenizers(tokenizers: dict, save_dir="./tokenizers"):
    os.makedirs(save_dir, exist_ok=True)
    for name, tokenizer in tokenizers.items():
        tokenizer_path = os.path.join(save_dir, f"{name}.json")
        tokenizer.save(tokenizer_path)
        tokenizer.model.save(save_dir, name)

special_tokens = ["<UNK>", "<PAD>", "<s>", "<s/>"]
path = "./data/c4_106/train.jsonl"

dataset = load_dataset("json", data_files=path)
print(len(dataset["train"]))

def train(datasets: list[str],
          min_count: int,
          special_tokens: list[str],
          vocab_sizes: list[int]) -> dict:

    nested_tokenizer_dict = {}
    for vocab_size in vocab_sizes:
        for set in datasets:
            path = f"./data/{set}/train.jsonl"
            dataset = load_dataset("json", data_files=path)
            tokenizer_dict = build_vocab_huggingface(vocab_size=vocab_size,
                                                     train_data=dataset["train"],
                                                     min_count=min_count,
                                                     special_tokens=special_tokens)
            save_tokenizers(tokenizer_dict, save_dir=f"./tokenizers/huggingface_tokenizers_{vocab_size}_{set}")
            nested_tokenizer_dict[f"{vocab_size}-{set}"] = tokenizer_dict
    return nested_tokenizer_dict

#tokenizer_dict = build_vocab_huggingface(vocab_size=16000, train_data=dataset["train"], min_count=1, special_tokens=special_tokens)
#save_tokenizers(tokenizer_dict, save_dir="./tokenizers/huggingface_tokenizers_16k_c4_106")

#datasets = ["c4_103","c4_104","c4_105","c4_106","c4_107","c4_108"]
#datasets = ["codeparrot_103","codeparrot_104","codeparrot_105","codeparrot_106","codeparrot_107","codeparrot_108"]
datasets = ["gsm8k_103","gsm8k_104","gsm8k_105","gsm8k_106"]

vocab_sizes = [16000, 64000]
tkn_dict = train(datasets, 1, special_tokens, vocab_sizes)
