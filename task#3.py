
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import evaluate

# Load data
raw = load_dataset('csv', data_files={
    'train': 'cnn_dailymail/train.csv',
    'validation': 'cnn_dailymail/validation.csv',
    'test': 'cnn_dailymail/test.csv'
})

# Tokenize
tokenizer = T5Tokenizer.from_pretrained("t5-small")
def preprocess(batch):
    inputs = ["summarize: " + a for a in batch["article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(batch["highlights"], max_length=150, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = raw.map(preprocess, batched=True)
tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Model
model = T5ForConditionalGeneration.from_pretrained("t5-small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

train_loader = DataLoader(tokenized['train'], batch_size=20, shuffle=True)
val_loader = DataLoader(tokenized['validation'], batch_size=20)

# Train (200 steps)
for epoch in range(2):
    model.train()
    loss_sum = 0
    steps = 0
    for batch in train_loader:
        if steps >= 200: break
        optimizer.zero_grad()
        out = model(input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device))
        out.loss.backward()
        optimizer.step()
        loss_sum += out.loss.item()
        steps += 1
    print(f"Epoch {epoch+1} loss: {loss_sum/steps:.4f}")

# Validate
rouge = evaluate.load("rouge")
model.eval()
preds, refs = [], []
with torch.no_grad():
    for step, batch in enumerate(val_loader):
        if step >= 20: break
        out = model.generate(batch["input_ids"].to(device), max_length=150, num_beams=4)
        preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
        lbl = batch["labels"].clone()
        lbl[lbl == -100] = tokenizer.pad_token_id
        refs.extend(tokenizer.batch_decode(lbl, skip_special_tokens=True))
print("ROUGE:", rouge.compute(predictions=preds, references=refs))

# Save
model.save_pretrained("./t5_summarizer")
tokenizer.save_pretrained("./t5_summarizer")
