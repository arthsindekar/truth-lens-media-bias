import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -----------------------------------------------------------
# 1. Load dataset (LOCAL FILES)
# -----------------------------------------------------------

BASE_DIR = "data/"


def load_split(name: str):
    """Load the local parquet split."""
    files = {
        "train": "train-00000-of-00001.parquet",
        "valid": "valid-00000-of-00001.parquet",
        "test": "test-00000-of-00001.parquet",
    }
    path = os.path.join(BASE_DIR, files[name])
    print(f"Loading: {path}")
    return pd.read_parquet(path)


train_df = load_split("train")
train_df, valid_df = train_test_split(
    train_df,
    test_size=0.1,
    random_state=42,
    stratify=train_df["bias"]
)
test_df = load_split("test")

# REDUCE DATASET SIZE for faster training
print(f"Original train size: {len(train_df)}")
train_df = train_df.sample(n=min(5000, len(train_df)), random_state=42)  # Use only 5000 samples
print(f"Reduced train size: {len(train_df)}")

train_df, valid_df = train_test_split(
    train_df,
    test_size=0.1,
    random_state=42,
    stratify=train_df["bias"]
)
test_df = load_split("test")
# Reduce test set too
test_df = test_df.sample(n=min(1000, len(test_df)), random_state=42)

# -----------------------------------------------------------
# 2. Prepare text + labels
# -----------------------------------------------------------

def combine_text(df):
    return (
            df["title"].fillna("") + " " + df["content"].fillna("")
    ).astype(str)


X_train = combine_text(train_df).tolist()
y_train = train_df["bias"].tolist()

X_valid = combine_text(valid_df).tolist()
y_valid = valid_df["bias"].tolist()

X_test = combine_text(test_df).tolist()
y_test = test_df["bias"].tolist()

# Create label mapping
unique_labels = sorted(set(y_train))
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: str(label) for label, idx in label2id.items()}
num_labels = len(unique_labels)

print(f"\nNumber of classes: {num_labels}")
print(f"Label mapping: {label2id}")

# Convert labels to integers
y_train = [label2id[label] for label in y_train]
y_valid = [label2id[label] for label in y_valid]
y_test = [label2id[label] for label in y_test]


# -----------------------------------------------------------
# 3. Create PyTorch Dataset
# -----------------------------------------------------------

class BiasDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# -----------------------------------------------------------
# 4. Initialize BERT model and tokenizer
# -----------------------------------------------------------

print("\nLoading BERT model and tokenizer...")
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)
model.to(device)

# Create datasets
train_dataset = BiasDataset(X_train, y_train, tokenizer)
valid_dataset = BiasDataset(X_valid, y_valid, tokenizer)
test_dataset = BiasDataset(X_test, y_test, tokenizer)

# Create dataloaders
batch_size = 16  # Adjust based on your GPU memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# -----------------------------------------------------------
# 5. Training setup
# -----------------------------------------------------------

optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3  # BERT typically needs fewer epochs


# -----------------------------------------------------------
# 6. Training loop
# -----------------------------------------------------------

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0

    progress_bar = tqdm(data_loader, desc="Training")
    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(data_loader)


def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return np.array(predictions), np.array(true_labels)


print("\n" + "=" * 50)
print("Training BERT model...")
print("=" * 50)

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")

    # Train
    avg_loss = train_epoch(model, train_loader, optimizer, device)
    print(f"Average training loss: {avg_loss:.4f}")

    # Validate
    valid_preds, valid_labels = evaluate(model, valid_loader, device)
    valid_acc = accuracy_score(valid_labels, valid_preds)
    print(f"Validation accuracy: {valid_acc:.4f}")

# -----------------------------------------------------------
# 7. Final Evaluation
# -----------------------------------------------------------

print("\n" + "=" * 50)
print("VALIDATION RESULTS")
print("=" * 50)
valid_preds, valid_labels = evaluate(model, valid_loader, device)
print("Accuracy:", accuracy_score(valid_labels, valid_preds))
print("\n" + classification_report(
    valid_labels,
    valid_preds,
    target_names=[id2label[i] for i in range(num_labels)]
))
print("\nConfusion Matrix:")
print(confusion_matrix(valid_labels, valid_preds))

print("\n" + "=" * 50)
print("TEST RESULTS")
print("=" * 50)
test_preds, test_labels = evaluate(model, test_loader, device)
print("Accuracy:", accuracy_score(test_labels, test_preds))
print("\n" + classification_report(
    test_labels,
    test_preds,
    target_names=[id2label[i] for i in range(num_labels)]
))
print("\nConfusion Matrix:")
print(confusion_matrix(test_labels, test_preds))

# -----------------------------------------------------------
# 8. Save model (optional)
# -----------------------------------------------------------

print("\nSaving model...")
model.save_pretrained("./bert_bias_model")
tokenizer.save_pretrained("./bert_bias_model")
print("Model saved to ./bert_bias_model")

print("\nDone!")