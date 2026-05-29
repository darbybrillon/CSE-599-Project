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

def build_vocab_sentencepeice(vocab_size: int,
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

#def data_preprocessing(train_data: list[str],
#                       path: str,
#                       tokenizer=Whitespace) -> list[str]:


def build_vocabs(vocab_size: int,
                train_data: list[str],
                min_count: int,
                special_tokens: list[str],
                filename: str,
                model: str="bpe"):
    build_vocab_huggingface(vocab_size, train_data, min_count, special_tokens)
    build_vocab_sentencepeice(vocab_size, filename, special_tokens, model)

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

train_data = dataset["train"]["text"]

tokenizer_dict = build_vocab_huggingface(vocab_size=16000, train_data=dataset["train"], min_count=1, special_tokens=special_tokens)
save_tokenizers(tokenizer_dict, save_dir="./tokenizers/huggingface_tokenizers_16k_c4_106")