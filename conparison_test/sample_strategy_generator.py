import pandas as pd
from sklearn.utils import resample
from collections import Counter
import os

# === 参数配置 ===
original_data_path = r"C:\Users\ruime\PycharmProjects\CCGDataAnalyze\overlap_version2\processed\stru_gene_202501_2.csv" #
output_dir = r"C:\Users\ruime\PycharmProjects\CCGDataAnalyze\overlap_version2\conparison_test\data"
os.makedirs(output_dir, exist_ok=True)

# === 加载数据 ===
df = pd.read_csv(original_data_path)

# === 检查 label 是否存在 ===
assert 'label' in df.columns, "CSV 中必须包含 label 列"

# === 推断 overType 整数列 ===
if 'overType' not in df.columns:
    overtype_cols = [col for col in df.columns if col.startswith("overType_")]
    assert overtype_cols, "找不到 overType 的 one-hot 编码列"
    df['overType'] = df[overtype_cols].idxmax(axis=1).apply(lambda x: int(x.split('_')[1]))

# === 策略一：baseline（原始数据） ===
df_baseline = df.copy()
df_baseline.to_csv(os.path.join(output_dir, "train_data_baseline.csv"), index=False)

# === 策略二：标签均衡采样 ===
def balance_by_label(df):
    groups = [df[df['label'] == i] for i in [0, 1, 2]]
    min_size = min(len(g) for g in groups)
    balanced = pd.concat([
        resample(g, replace=False, n_samples=min_size, random_state=42)
        for g in groups
    ])
    return balanced.sample(frac=1, random_state=42)

df_balanced = balance_by_label(df)
df_balanced.to_csv(os.path.join(output_dir, "train_data_balanced.csv"), index=False)

# === 策略三：结构多样性采样 ===
def balance_structure(df_bal):
    counts = Counter(df_bal['overType'])
    min_count = min(counts.values())
    df_diverse = pd.concat([
        df_bal[df_bal['overType'] == k].sample(
            n=min(min_count, len(df_bal[df_bal['overType'] == k])), random_state=42
        )
        for k in counts
    ])
    return df_diverse.sample(frac=1, random_state=42)

df_structure = balance_structure(df_balanced)
df_structure.to_csv(os.path.join(output_dir, "train_data_structure.csv"), index=False)

# === 输出统计信息 ===
print("样本分布统计：")
print("原始分布（label）:", df['label'].value_counts().to_dict())
print("标签均衡分布:", df_balanced['label'].value_counts().to_dict())
print("结构多样分布（label）:", df_structure['label'].value_counts().to_dict())
print("结构多样分布（overType）:", df_structure['overType'].value_counts().to_dict())
print("所有采样数据已保存至：", os.path.abspath(output_dir))