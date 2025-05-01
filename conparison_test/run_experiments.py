import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from overlap_version2.models.model import StruGeneTransformer
from overlap_version2.utils.dataset_2 import StruGeneDataset

def train_model_full(
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
    save_plot_path_prefix,
    label_names,
    val_save_path,
    report_save_path,
    metrics_csv_path
):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    num_classes = len(np.unique(y))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    val_df = pd.DataFrame(X_val, columns=df.columns[:-1])
    val_df["label"] = y_val
    val_df.to_csv(val_save_path, index=False, encoding="utf-8")

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
    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        train_acc = correct / total
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, average="macro", zero_division=0)
        val_recall = recall_score(val_labels, val_preds, average="macro", zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        print(f"Epoch {epoch}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_acc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_model_path)

    # 保存指标曲线
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(metrics_csv_path, index=False)

    plt.figure()
    plt.plot(hist_df["epoch"], hist_df["train_loss"], label="Train Loss")
    plt.plot(hist_df["epoch"], hist_df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_plot_path_prefix}_loss.png")

    plt.figure()
    plt.plot(hist_df["epoch"], hist_df["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_plot_path_prefix}_accuracy.png")

    # 保存最终验证集分类报告
    report = classification_report(val_labels, val_preds, target_names=label_names, digits=4)
    with open(report_save_path, "w", encoding="utf-8") as f:
        f.write(report)
        print("\n📄 Classification Report:\n", report)

train_model_full(
    csv_path=r"C:\Users\ruime\PycharmProjects\CCGDataAnalyze\overlap_version2\conparison_test\data\train_data_structure.csv",
    num_epochs=300,
    batch_size=64,
    learning_rate=0.0005,
    embed_dim=64,
    num_heads=4,
    ff_dim=128,
    num_layers=2,
    dropout=0.4,
    save_model_path=r"C:\Users\ruime\PycharmProjects\CCGDataAnalyze\overlap_version2\conparison_test\model_bestline\best_model_struct.pt",
    save_plot_path_prefix=r"C:\Users\ruime\PycharmProjects\CCGDataAnalyze\overlap_version2\conparison_test\model_structure\metrics_struct",
    label_names=["Lp", "Rp", "Noth"],
    val_save_path=r"C:\Users\ruime\PycharmProjects\CCGDataAnalyze\overlap_version2\conparison_test\model_structure\val_set_struct.csv",
    report_save_path=r"C:\Users\ruime\PycharmProjects\CCGDataAnalyze\overlap_version2\conparison_test\model_structure\classification_report_struct.txt",
    metrics_csv_path=r"C:\Users\ruime\PycharmProjects\CCGDataAnalyze\overlap_version2\conparison_test\model_structure\metrics_struct.csv"
)

