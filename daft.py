import pandas as pd
df = pd.read_csv(r"C:\Users\ruime\PycharmProjects\CCGDataAnalyze\overlap_version2\processed\stru_gene_202501.csv")
print(df["label"].value_counts())