import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report

from overlap_version2.models.custom_transformer import TransformerClassifier
from overlap_version2.tokenizer.simple_tokenizer import SimpleTokenizer
from overlap_version2.utils.dataset import StruGeneDataset

# ==== Hyperparameters ====
vocab_path = r"C:\Users\ruime\PycharmProjects\CCGDataAnalyze\overlap_version2\tokenizer\vocab.txt"
data_path = r"C:\Users\ruime\PycharmProjects\CCGDataAnalyze\overlap_version2\processed\stru_gene_transformer.csv"
model_save_path = "outputs/best_model.pt"

batch_size = 32
epochs = 100
learning_rate = 1e-3
max_length = 128
val_split = 0.2
log_interval = 10
dropout = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load Tokenizer and Full Dataset ====
tokenizer = SimpleTokenizer(vocab_path, max_length=max_length)
full_dataset = StruGeneDataset(data_path, tokenizer, max_length=max_length)

# ==== Split Train/Val ====
val_size = int(len(full_dataset) * val_split)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

print(f"Loaded dataset: {len(full_dataset)} samples → {train_size} train / {val_size} val")

# ==== Init Model ====
vocab_size = len(tokenizer.token2id)
model = TransformerClassifier(
    vocab_size=vocab_size,
    max_len=max_length,
    dropout=dropout
).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # ✅ 启用 label smoothing
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ==== Train ====
loss_history = []
val_loss_history = []
best_val_acc = 0.0

for epoch in range(epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for i, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (i + 1) % log_interval == 0:
            print(f" Train | Epoch [{epoch+1}/{epochs}] Step [{i+1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} Acc: {correct/total:.4f}")

    avg_train_loss = total_loss / len(train_loader)
    train_acc = correct / total
    loss_history.append(avg_train_loss)

    # ==== Validation ====
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            val_loss += loss.item()
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total
    val_loss_history.append(avg_val_loss)

    print(f" Epoch {epoch+1} done | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} "
          f"| Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f}")

    # ==== 分类报告 ====
    label_names = ["Lp", "Rp", "Noth"]
    print("\n Validation Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_names, digits=4))

    # ==== Save best model ====
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs("outputs", exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f" Best model saved! Val Acc: {val_acc:.4f}")

# ==== Plot Loss ====
plt.plot(range(1, epochs+1), loss_history, marker='.', markersize=3, label="Train Loss")
plt.plot(range(1, epochs+1), val_loss_history, marker='.', markersize=3, label="Val Loss")
plt.title("Training and Validation Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.savefig("outputs/loss_curve_2.png")
plt.show()
