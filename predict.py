# import torch
# import pandas as pd
# import numpy as np
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.metrics import classification_report, accuracy_score
# from overlap_version2.models.model import StruGeneTransformer
#
# def evaluate_model(
#     csv_path,
#     model_path,
#     label_names=None,
#     num_samples=1000,
#     batch_size=64,
#     device=None,
#     save_path=None
# ):
#     if label_names is None:
#         label_names = ["Lp", "Rp", "Noth"]
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # === 加载前 num_samples 条数据 ===
#     df = pd.read_csv(csv_path).head(num_samples)
#     X = df.drop(columns=["label"]).values.astype(np.float32)
#     y_true = df["label"].values.astype(np.int64)
#
#     dataset = TensorDataset(torch.tensor(X))
#     loader = DataLoader(dataset, batch_size=batch_size)
#
#     # === 加载模型结构 ===
#     model = StruGeneTransformer(
#         input_dim=X.shape[1],
#         embed_dim=64,
#         num_heads=4,
#         ff_dim=128,
#         num_layers=2,
#         num_classes=len(label_names),
#         dropout=0.0
#     ).to(device)
#
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
#
#     predictions = []
#
#     with torch.no_grad():
#         for (X_batch,) in loader:
#             X_batch = X_batch.to(device)
#             outputs = model(X_batch)
#             preds = torch.argmax(outputs, dim=1)
#             predictions.extend(preds.cpu().numpy())
#
#     y_pred = np.array(predictions)
#
#     # === 输出评估指标 ===
#     print(" Accuracy:", accuracy_score(y_true, y_pred))
#     print("\nClassification Report:")
#     print(classification_report(y_true, y_pred, target_names=label_names))
#
#     # === 可选：保存结果
#     if save_path:
#         df_result = df.copy()
#         df_result["prediction"] = y_pred
#         df_result.to_csv(save_path, index=False, encoding="utf-8")
#         print(f"预测结果保存至: {save_path}")
# if __name__ == "__main__":
#     evaluate_model(
#         csv_path=r"C:\Users\ruime\PycharmProjects\CCGDataAnalyze\overlap_version2\processed\stru_gene_202501.csv",
#         model_path=r"C:\Users\ruime\PycharmProjects\CCGDataAnalyze\overlap_version2\train\outputs\best_model.pt",
#         label_names=["Lp", "Rp", "Noth"],
#         num_samples=1000,
#         save_path=r"C:\Users\ruime\PycharmProjects\CCGDataAnalyze\overlap_version2\data\eval_results.csv"
#     )


import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score
from overlap_version2.models.model import StruGeneTransformer

def evaluate_model(
    csv_path,
    model_path,
    label_names=["Lp", "Rp", "Noth"],
    batch_size=64,
    device=None,
    save_path=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 加载数据（需要包含 label 列）===
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y_true = df["label"].values.astype(np.int64)

    dataset = TensorDataset(torch.tensor(X))
    loader = DataLoader(dataset, batch_size=batch_size)

    # === 初始化模型 ===
    model = StruGeneTransformer(
        input_dim=X.shape[1],
        embed_dim=64,
        num_heads=4,
        ff_dim=128,
        num_layers=2,
        num_classes=len(label_names),
        dropout=0.0
    ).to(device)

    # === 加载训练好的参数 ===
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # === 执行预测 ===
    predictions = []
    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())

    y_pred = np.array(predictions)

    # === 打印评估结果 ===
    print(" Accuracy:", accuracy_score(y_true, y_pred))
    print("\n Classification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names))

    # === 保存预测结果（可选）===
    if save_path:
        result_df = df.copy()
        result_df["prediction"] = y_pred
        result_df.to_csv(save_path, index=False, encoding="utf-8")
        print(f" 预测结果保存至: {save_path}")
if __name__ == "__main__":
    evaluate_model(
        csv_path=r"C:\Users\ruime\PycharmProjects\CCGDataAnalyze\overlap_version2\train\outputs\val_set_2.csv",
        model_path=r"C:\Users\ruime\PycharmProjects\CCGDataAnalyze\overlap_version2\train\outputs\best_model_2.pt",
        label_names=["Lp", "Rp", "Noth"],
        save_path=r"C:\Users\ruime\PycharmProjects\CCGDataAnalyze\overlap_version2\train\outputs\val_predictions.csv"
    )
