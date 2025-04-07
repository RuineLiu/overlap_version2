import json
import csv

# 标签映射
prior_map = {"Lp": 0, "Rp": 1, "Noth": 2}

def parse_phrases(phrase_str):
    """解析短语结构字符串为 token 序列"""
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

def build_input_text(entry):
    """将每个数据条目转换为 transformer 输入序列"""
    parts = []
    parts.append("[LEFT_EXT] " + " ".join(parse_phrases(entry["leftExtend"])))
    parts.append("[LEFT_OVER] " + " ".join(parse_phrases(entry["leftOver"])))
    parts.append("[RIGHT_OVER] " + " ".join(parse_phrases(entry["rightOver"])))
    parts.append("[RIGHT_EXT] " + " ".join(parse_phrases(entry["rightExtend"])))
    parts.append("[OT] " + str(entry["overType"]))
    return " ".join(parts)

def select_label(entry):
    """根据 clauTagPrior 选择最频繁的 Prior 作为标签"""
    prior_list = entry["clauTagPrior"].strip("[]").split("),(")
    counter = {"Lp": 0, "Rp": 0, "Noth": 0}
    for item in prior_list:
        item = item.strip("() ").split(",")
        if len(item) >= 3:
            prior = item[2].strip().replace(")", "")
            if prior in counter:
                counter[prior] += 1
    if counter:
        return max(counter, key=counter.get)
    return "Noth"

def process(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["input_text", "label"])
        writer.writeheader()
        for entry in data:
            input_text = build_input_text(entry)
            label_str = select_label(entry)
            label = prior_map.get(label_str, 2)
            writer.writerow({"input_text": input_text, "label": label})

if __name__ == "__main__":
    process(r"C:\Users\ruime\PycharmProjects\CCGDataAnalyze\overlap_version2\data\stru_gene_202501.json",
            r"C:\Users\ruime\PycharmProjects\CCGDataAnalyze\overlap_version2\processed\stru_gene_transformer.csv")
