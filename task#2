
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from evaluate import load
import os, requests, zipfile

# Download SPoC
if not os.path.exists('spoc.zip'):
    r = requests.get('https://sumith1896.github.io/spoc/data/spoc.zip')
    open('spoc.zip', 'wb').write(r.content)
if not os.path.exists('data'):
    with zipfile.ZipFile('spoc.zip') as z:
        z.extractall()
    os.rename('spoc', 'data')
    os.remove('spoc.zip')

# Load data
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
raw = load_dataset('csv', data_files={
    'train': 'data/train/split/spoc-train-train.tsv',
    'validation': 'data/train/split/spoc-train-eval.tsv'
}, sep='\t', header=None)
raw = raw.rename_columns({0: 'pseudocode', 1: 'code'})

# Preprocess
def preprocess(b):
    texts = [f"{p} {tokenizer.eos_token} {c} {tokenizer.eos_token}" for p, c in zip(b['pseudocode'], b['code'])]
    enc = tokenizer(texts, max_length=512, truncation=True, padding="max_length")
    enc["labels"] = enc["input_ids"].copy()
    return enc

tokenized = raw.map(preprocess, batched=True)
tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Loaders
model = AutoModelForCausalLM.from_pretrained("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
train_loader = DataLoader(tokenized['train'], batch_size=4, shuffle=True)
val_loader = DataLoader(tokenized['validation'], batch_size=4)

# Train (200 steps/epoch)
max_steps = 200
for epoch in range(2):
    model.train()
    loss_sum = 0
    steps = 0
    for batch in train_loader:
        if steps >= max_steps: break
        optimizer.zero_grad()
        out = model(input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device))
        out.loss.backward()
        optimizer.step()
        loss_sum += out.loss.item()
        steps += 1
    print(f"Epoch {epoch+1} train loss: {loss_sum/steps:.4f}")

    model.eval()
    val_loss = 0
    steps = 0
    max_val = 50
    with torch.no_grad():
        for batch in val_loader:
            if steps >= max_val: break
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device))
            val_loss += out.loss.item()
            steps += 1
    print(f"Epoch {epoch+1} val loss: {val_loss/steps:.4f}")

# Evaluate
bleu = load("bleu")
codebleu = load("codebleu")
model.eval()
preds, refs = [], []
with torch.no_grad():
    for i in range(100):
        pseudo = raw['validation'][i]['pseudocode']
        ref = raw['validation'][i]['code']
        inp = tokenizer(f"{pseudo} {tokenizer.eos_token}", return_tensors="pt", max_length=256, truncation=True).input_ids.to(device)
        out = model.generate(inp, max_length=512, num_beams=4, early_stopping=True, pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][len(inp[0]):], skip_special_tokens=True)
        preds.append(gen)
        refs.append(ref)

print("BLEU:", bleu.compute(predictions=preds, references=refs))
print("CodeBLEU:", codebleu.compute(predictions=preds, references=[[r] for r in refs], lang="cpp"))

# Examples
print("Examples:")
for i in range(3):
    print(f"Pseudo: {raw['validation'][i]['pseudocode']}")
    print(f"Ref: {refs[i]}")
    print(f"Gen: {preds[i]}")
    print("="*80)

# Save
model.save_pretrained("./gpt2_code")
tokenizer.save_pretrained("./gpt2_code")
