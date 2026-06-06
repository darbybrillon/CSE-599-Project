import json
import os

#datasets = ["c4_103","c4_104","c4_105","c4_106","c4_107","c4_108"]
#datasets = ["codeparrot_103","codeparrot_104","codeparrot_105","codeparrot_106","codeparrot_107","codeparrot_108"]
datasets = ["gsm8k_103","gsm8k_104","gsm8k_105","gsm8k_106"]
types = ["test", "train", "validation"]

for dataset in datasets:
    for type in types:
        input_path = fr"C:\Users\darby\Downloads\CSE 599\Project\data\{dataset}\{type}.jsonl"
        if type == "train":
            output_path = fr"C:\Users\darby\Downloads\CSE 599\Project\data_txt\{dataset}\train\{type}.txt"
        else:
            output_path = fr"C:\Users\darby\Downloads\CSE 599\Project\data_txt\{dataset}\{type}.txt"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:

            for line in infile:
                if not line.strip():
                    continue

                record = json.loads(line)
                outfile.write(record["text"] + "\n")