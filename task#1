
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load & clean data
df = pd.read_csv("sentiment-analysis.csv")
label_id = {'negative': 0, 'neutral': 1, 'positive': 2}
texts, labels = [], []
for row in df.iloc[:, 0]:
    if not isinstance(row, str): continue
    parts = [p.strip().strip('"') for p in row.split(',')]
    if len(parts) < 2: continue
    texts.append(parts[0])
    labels.append(label_id.get(parts[1].lower(), -1))
texts = [t for t, l in zip(texts, labels) if l != -1]
labels = [l for l in labels if l != -1]

# Tokenize
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')

# Dataset
class FeedbackDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[i])
        return item

dataset = FeedbackDataset(encodings, labels)
train_data, test_data = random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16)

# Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Train
for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(**{k: v.to(device) for k, v in batch.items() if k != 'labels'}, labels=batch['labels'].to(device))
        outputs.loss.backward()
        optimizer.step()
        total_loss += outputs.loss.item()
    print(f"Epoch {epoch+1} loss: {total_loss/len(train_loader):.4f}")

# Evaluate
model.eval()
preds, trues = [], []
with torch.no_grad():
    for batch in test_loader:
        outputs = model(**{k: v.to(device) for k, v in batch.items() if k != 'labels'})
        preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
        trues.extend(batch['labels'].cpu().numpy())

print(f"Accuracy: {accuracy_score(trues, preds):.4f}")
print(f"F1: {f1_score(trues, preds, average='weighted'):.4f}")

# Confusion matrix
cm = confusion_matrix(trues, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Pred")
plt.ylabel("True")
plt.savefig("confusion_matrix_task1.png")
plt.show()

# Save
model.save_pretrained("./bert_sentiment")
tokenizer.save_pretrained("./bert_sentiment")
