import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, AdamW, get_linear_schedule_with_warmup, \
    AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score
from peft import LoraConfig, get_peft_model, TaskType

# Load the MRPC dataset
dataset = load_dataset("glue", "mrpc")

for model_name in ["microsoft/deberta-v3-large", "google/gemma-2b"]:

    # best Hyperparameter:
    # Batch Size: 16 | Learning Rate: 2e-05
    # r: 16 | Batch Size: 16 | Learning Rate: 0.002


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


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
    learning_rate = 2e-3
    adam_epsilon = 1e-8
    warmup_steps = 0
    total_steps = len(train_dataloader) * epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)


    # LoRA configuration
    def apply_lora(model, r):
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=r,
            lora_alpha=r,
            lora_dropout=0.1,
            target_modules=["query_proj", "value_proj"]
        )
        model = get_peft_model(model, config)
        model.classifier.requires_grad_(True)

        # assert SEQ classification head is trainable
        assert list(model.classifier.parameters())[-1].requires_grad

        return model


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

    best_accuracy = 0
    r_factor = 16

    lora_model = apply_lora(model, r_factor)
    optimizer = AdamW(lora_model.parameters(), lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    for epoch in range(epochs):
        avg_train_loss = train(lora_model, train_dataloader, optimizer, scheduler)
        val_accuracy = evaluate(lora_model, val_dataloader)

        print(f"Model Name: {model_name} | Epoch {epoch + 1}/{epochs} | r: {r_factor} | Batch Size: {batch_size} | Learning Rate: {learning_rate}")
        print(f"Train Loss: {avg_train_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

    print(f"Best Validation Accuracy: {best_accuracy:.4f}")
