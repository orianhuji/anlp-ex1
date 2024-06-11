import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score

# Load the MRPC dataset
dataset = load_dataset("glue", "mrpc")

# Load the DeBERTa model and tokenizer
model_name = "microsoft/deberta-v3-base"
tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
model = DebertaV2ForSequenceClassification.from_pretrained(model_name, num_labels=2)


# Tokenize the dataset
def tokenize(batch):
    return tokenizer(batch['sentence1'], batch['sentence2'], padding='max_length', truncation=True, max_length=256)


train_dataset = dataset['train'].map(tokenize, batched=True)
val_dataset = dataset['validation'].map(tokenize, batched=True)

# Format the dataset for PyTorch
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Create DataLoaders
batch_size = 16
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

# Fine-tuning parameters
epochs = 3
learning_rate = 2e-5
adam_epsilon = 1e-8
warmup_steps = 0
total_steps = len(train_dataloader) * epochs

# Prepare optimizer and schedule (linear warmup and decay)
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)


# Training function
def train(model, train_dataloader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return total_loss / len(train_dataloader)


# Evaluation function
def evaluate(model, val_dataloader):
    model.eval()
    preds = []
    true_labels = []
    for batch in val_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds.extend(torch.argmax(logits, dim=1).tolist())
        true_labels.extend(labels.tolist())
    return accuracy_score(true_labels, preds)


# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Fine-tuning loop
best_accuracy = 0
hyperparams_trials = [
    {"batch_size": 16, "learning_rate": 2e-5},
    {"batch_size": 32, "learning_rate": 3e-5},
    {"batch_size": 16, "learning_rate": 5e-5},
]

for trial in hyperparams_trials:
    batch_size = trial["batch_size"]
    learning_rate = trial["learning_rate"]

    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

    for epoch in range(epochs):
        avg_train_loss = train(model, train_dataloader, optimizer, scheduler)
        val_accuracy = evaluate(model, val_dataloader)

        print(f"Epoch {epoch + 1}/{epochs} | Batch Size: {batch_size} | Learning Rate: {learning_rate}")
        print(f"Train Loss: {avg_train_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

print(f"Best Validation Accuracy: {best_accuracy:.4f}")
