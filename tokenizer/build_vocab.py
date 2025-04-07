import json
from collections import Counter

def parse_phrases(phrase_str):
    if not phrase_str or phrase_str == "[]":
        return []
    phrase_str = phrase_str.strip("[]")
    items = phrase_str.split("),(")
    tokens = []
    for item in items:
        item = item.strip("()\"")
        parts = item.split(",")
        tokens.extend([p.strip().strip('"') for p in parts])
    return tokens

def build_vocab(json_path, output_path="tokenizer/vocab.txt"):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    counter = Counter()

    for entry in data:
        counter.update(["[LEFT_EXT]"])
        counter.update(parse_phrases(entry["leftExtend"]))

        counter.update(["[LEFT_OVER]"])
        counter.update(parse_phrases(entry["leftOver"]))

        counter.update(["[RIGHT_OVER]"])
        counter.update(parse_phrases(entry["rightOver"]))

        counter.update(["[RIGHT_EXT]"])
        counter.update(parse_phrases(entry["rightExtend"]))

        counter.update(["[OT]"])
        counter.update([str(entry["overType"])])

    special_tokens = ["[PAD]", "[UNK]"]
    vocab_list = special_tokens + [tok for tok, _ in counter.most_common()]

    with open(output_path, 'w', encoding='utf-8') as f:
        for token in vocab_list:
            f.write(token + "\n")

if __name__ == "__main__":
    build_vocab(r"C:\Users\ruime\PycharmProjects\CCGDataAnalyze\overlap_version2\data\stru_gene_202501.json", 
                r"C:\Users\ruime\PycharmProjects\CCGDataAnalyze\overlap_version2\tokenizer\vocab.txt")
