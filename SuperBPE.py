r"""
RUN THESE FIRST IN TERMINAL
git clone --recurse-submodules https://github.com/PythonNut/superbpe.git
cd superbpe
pip install -r requirements.txt

THEN RUN THE FOLLOWING MAKING SURE TO CHANGE DIRECTORY NAMES; DO NOT CHANGE ANYTHING ELSE

TRAIN TOKENIZER:
python -m train_tokenizer --output_dir SuperBPE_tok --corpus_dir "C:\Users\darby\Downloads\CSE 599\Project\test" --vocab_size 12500 --regex_string "[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"

EXTEND TOKENIZER:
python -m train_tokenizer --output_dir SuperBPE_tok --vocab_size 16000 --regex_string "\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)"

"""

from superbpe import *
from superbpe.utils import construct_hf_tokenizer
from transformers import AutoTokenizer

# Make HF tokenizer

construct_hf_tokenizer("./superbpe/SuperBPE_tok")

# Test Results

tok = AutoTokenizer.from_pretrained("./superbpe/SuperBPE_tok")

text = "The quick brown fox jumps over the lazy dog"

print(tok.tokenize(text))
print(tok.encode(text))
print(tok.decode(tok.encode(text)))