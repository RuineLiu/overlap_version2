import json
import ast
import re
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

label_map = {"Lp": 0, "Rp": 1, "Noth": 2}

def parse_over_tuple(field):
    match = re.findall(r'\(([^,]+),\s*"([^"]+)",\s*"([^"]+)",\s*(\d+)\)', field)
    if match:
        items = match[0]
        return [
            hash(items[0]) % 10000,
            hash(items[1]) % 10000,
            hash(items[2]) % 10000,
            int(items[3])
        ]
    else:
        return [0, 0, 0, 0]

# def one_hot_over_type(over_type):
#     vec = [0, 0, 0]
#     if 0 <= over_type < 3:
#         vec[over_type] = 1
#     return vec
def one_hot_over_type(over_type, num_classes=5):
    vec = [0] * num_classes
    if 0 <= over_type < num_classes:
        vec[over_type] = 1
    return vec


def parse_label(clauTagPrior):
    matches = re.findall(r'\(\([^\)]+\),\s*(Lp|Rp|Noth)\)', clauTagPrior)
    if matches:
        return max(set(matches), key=matches.count)
    return "Noth"

def parse_extend_features(extend_list):
    count = len(extend_list)
    indices = []
    for item in extend_list:
        match = re.findall(r'\(([^,]+),\s*"([^"]+)",\s*"([^"]+)",\s*(\d+)\)', str(item))
        if match:
            indices.append(int(match[0][3]))
    avg_index = np.mean(indices) if indices else 0
    return [count, avg_index]

def safe_parse_extend(raw_text):
    try:
        if isinstance(raw_text, list):
            return raw_text
        elif not raw_text or raw_text == "[]":
            return []
        else:
            raw_text = raw_text.replace("\\", "\\\\")
            fixed = re.sub(r'\(\s*([a-zA-Z0-9_\/\\\.\*\#\(\)]+)\s*,', r"('\1',", raw_text)
            return ast.literal_eval(fixed)
    except Exception as e:
        print(f"[ Extend 字段解析失败]：{raw_text}\n错误：{e}")
        return []

def extract_features_from_sample(sample):
    left_over_vec = parse_over_tuple(sample.get("leftOver", ""))
    right_over_vec = parse_over_tuple(sample.get("rightOver", ""))
    over_type_vec = one_hot_over_type(int(sample.get("overType", 0)))

    count_vec = [
        int(sample.get("lpHitCount", 0)),
        int(sample.get("rpHitCount", 0)),
        int(sample.get("nothHitCount", 0))
    ]

    left_extend = safe_parse_extend(sample.get("leftExtend", "[]"))
    right_extend = safe_parse_extend(sample.get("rightExtend", "[]"))
    left_extend_feat = parse_extend_features(left_extend)
    right_extend_feat = parse_extend_features(right_extend)

    label_str = parse_label(sample.get("clauTagPrior", ""))
    label = label_map.get(label_str, 2)  # 默认为 Noth

    features = (
        left_over_vec +
        right_over_vec +
        over_type_vec +
        count_vec +
        left_extend_feat +
        right_extend_feat
    )
    return features, label

def load_dataset(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    X, y = [], []
    for sample in data:
        features, label = extract_features_from_sample(sample)
        X.append(features)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def get_dataloader(json_path, batch_size=32, shuffle=True):
    X, y = load_dataset(json_path)
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def save_features_to_csv(json_path, output_csv_path):
    X, y = load_dataset(json_path)

    columns = [
        "left_struct_1", "left_struct_2", "left_struct_3", "left_count",
        "right_struct_1", "right_struct_2", "right_struct_3", "right_count",
            "overType_0", "overType_1", "overType_2", "overType_3", "overType_4",
        "lpHitCount", "rpHitCount", "nothHitCount",
        "LeftExtend_len", "LeftExtend_avgIndex",
        "RightExtend_len", "RightExtend_avgIndex"
    ]

    df = pd.DataFrame(X, columns=columns)
    df["label"] = y
    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f" 已保存特征为 CSV：{output_csv_path}")

# === 主程序入口 ===
if __name__ == "__main__":
    json_file = r"C:\Users\ruime\PycharmProjects\CCGDataAnalyze\overlap_version2\data\stru_gene_202501.json"
    csv_file = r"C:\Users\ruime\PycharmProjects\CCGDataAnalyze\overlap_version2\processed\stru_gene_202501_2.csv"

    save_features_to_csv(json_file, csv_file)

    loader = get_dataloader(json_file, batch_size=8)
    for X_batch, y_batch in loader:
        print(" X batch shape:", X_batch.shape)
        print(" y batch shape:", y_batch.shape)
        print(" First row:", X_batch[0])
        print(" Label:", y_batch[0])
        break
