import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_scheduler
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.optim import AdamW
from tqdm.auto import tqdm

# Load dataset and model
dataset = load_dataset("imdb")
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Preprocessing
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Convert to PyTorch format
tokenized_datasets.set_format("torch")

# Create dataloaders
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8
)
eval_dataloader = DataLoader(
    tokenized_datasets["test"], batch_size=8
)

# Setup training
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# Training loop
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# Evaluation
model.eval()
predictions = []
labels = []

for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    
    logits = outputs.logits
    predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
    labels.extend(batch["labels"].cpu().numpy())

# Calculate accuracy
accuracy = sum(p == l for p, l in zip(predictions, labels)) / len(predictions)
print(f"Accuracy: {accuracy}") 