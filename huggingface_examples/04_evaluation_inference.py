from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from datasets import load_dataset
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_fine_tuned_model(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def predict(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    return predictions.cpu().numpy()

# Load model and evaluate
model_path = "./fine_tuned_model"
model, tokenizer = load_fine_tuned_model(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load test dataset
dataset = load_dataset("imdb")
test_dataset = dataset["test"]

# Evaluate on test set
predictions = []
true_labels = []

for example in test_dataset:
    pred = predict(example["text"], model, tokenizer, device)
    predictions.append(pred.argmax())
    true_labels.append(example["label"])

# Print classification report
print("\nClassification Report:")
print(classification_report(true_labels, predictions))

# Plot confusion matrix
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Example inference
example_texts = [
    "This movie was absolutely fantastic!",
    "I really didn't enjoy this film at all.",
    "It was okay, but nothing special."
]

for text in example_texts:
    pred = predict(text, model, tokenizer, device)
    sentiment = "Positive" if pred.argmax() == 1 else "Negative"
    confidence = pred.max()
    print(f"\nText: {text}")
    print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})") 