from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import wandb
from transformers.integrations import WandbCallback

# Initialize wandb
wandb.init(project="model-finetuning")

# Load dataset and model
dataset = load_dataset("imdb")
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Advanced preprocessing with dynamic padding
def collate_fn(examples):
    return tokenizer.pad(
        examples,
        padding=True,
        return_tensors="pt",
    )

# Custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": item["label"]
        }

# Training arguments with additional features
training_args = TrainingArguments(
    output_dir="./advanced_results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=True,  # Mixed precision training
    gradient_accumulation_steps=4,
    warmup_steps=500,
)

# Custom trainer class
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Custom loss calculation
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]).to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# Initialize trainer with custom components
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=CustomDataset(dataset["train"], tokenizer),
    eval_dataset=CustomDataset(dataset["test"], tokenizer),
    callbacks=[WandbCallback]
)

# Train and save
trainer.train()
trainer.save_model("./advanced_fine_tuned_model") 