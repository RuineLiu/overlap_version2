import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from overlap_version2.models.model import StruGeneTransformer
from overlap_version2.utils.dataset_2 import StruGeneDataset


def train_model(
    csv_path,
    num_epochs,
    batch_size,
    learning_rate,
    embed_dim,
    num_heads,
    ff_dim,
    num_layers,
    dropout,
    save_model_path,
    save_plot_path,
    label_names,
    val_save_path  # 新增参数：验证集保存路径
):
    # === 加载数据 ===
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    num_classes = len(np.unique(y))
    print(np.bincount(y))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    val_df = pd.DataFrame(X_val, columns=df.columns[:-1])
    val_df["label"] = y_val 
    val_df.to_csv(val_save_path, index=False, encoding="utf-8")
    print(f" 验证集已保存至: {val_save_path}")

    train_loader = DataLoader(StruGeneDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(StruGeneDataset(X_val, y_val), batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StruGeneTransformer(
        input_dim=X.shape[1],
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        train_correct, train_total = 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if epoch % 10 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"[Epoch {epoch}] 训练集准确率: {train_acc:.4f} | 验证集准确率: {val_acc:.4f}")
            print(classification_report(all_labels, all_preds, target_names=label_names))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_model_path)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_path = save_plot_path.replace(".png", "_loss.png")
    plt.savefig(loss_plot_path)

    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    acc_plot_path = save_plot_path.replace(".png", "_accuracy.png")
    plt.savefig(acc_plot_path)

    print(f" Loss 曲线保存至: {loss_plot_path}")
    print(f" Accuracy 曲线保存至: {acc_plot_path}")
    print(f" 最佳模型已保存至: {save_model_path}")


def main():
    config = {
        "csv_path":r"C:\Users\ruime\PycharmProjects\CCGDataAnalyze\overlap_version2\processed\stru_gene_202501_2.csv",
        "num_epochs": 300,
        "batch_size": 64,
        "learning_rate": 0.0005,
        "embed_dim": 64,
        "num_heads": 4,
        "ff_dim": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "save_model_path": r"C:\Users\ruime\PycharmProjects\CCGDataAnalyze\overlap_version2\train\outputs\best_model_2.pt",
        "save_plot_path": r"C:\Users\ruime\PycharmProjects\CCGDataAnalyze\overlap_version2\train\outputs\loss_curve_7 .png",
        "label_names": ["Lp", "Rp", "Noth"],
        "val_save_path": r"C:\Users\ruime\PycharmProjects\CCGDataAnalyze\overlap_version2\train\outputs\val_set_2.csv"
    }

    train_model(**config)


if __name__ == "__main__":
    main()