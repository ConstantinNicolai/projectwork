import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
import time
import datetime
from torch.optim import AdamW  # Use the PyTorch AdamW optimizer
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn

# Load the IMDb dataset
dataset = load_dataset('imdb')

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the data
def encode(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256)

encoded_dataset = dataset.map(encode, batched=True)

# Convert to torch tensors
train_dataset = TensorDataset(
    torch.tensor(encoded_dataset['train']['input_ids']), 
    torch.tensor(encoded_dataset['train']['attention_mask']), 
    torch.tensor(encoded_dataset['train']['label'])
)

val_dataset = TensorDataset(
    torch.tensor(encoded_dataset['test']['input_ids']), 
    torch.tensor(encoded_dataset['test']['attention_mask']), 
    torch.tensor(encoded_dataset['test']['label'])
)

# Set batch size and epochs
batch_size = 16
epochs = 1

# Adjust DataLoader
train_dataloader = DataLoader(
    train_dataset, 
    sampler=RandomSampler(train_dataset), 
    batch_size=batch_size,
    num_workers=4
)

validation_dataloader = DataLoader(
    val_dataset, 
    sampler=SequentialSampler(val_dataset), 
    batch_size=batch_size,
    num_workers=4
)

# Load BERT model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=2, 
    output_attentions=False, 
    output_hidden_states=False,
)

# Utilize DataParallel for multi-GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model)
model = model.to(device)

# Set up the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0, 
                                            num_training_steps=total_steps)

scaler = GradScaler()  # Initialize mixed precision training

# Training loop
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

print(f"Device: {device}")

total_t0 = time.time()

for epoch_i in range(epochs):
    print("")
    print(f'======== Epoch {epoch_i + 1} / {epochs} ========')
    print('Training...')
    
    t0 = time.time()  # Start time of the epoch

    total_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()

        with autocast():
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)
            loss = outputs.loss

        total_loss += loss.item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)            
    
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

# Validation
print("Running Validation...")

model.eval()

eval_accuracy = 0
nb_eval_steps = 0

for batch in validation_dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)
    
    with torch.no_grad():
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)

    logits = outputs.logits
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    pred_flat = np.argmax(logits, axis=1).flatten()
    labels_flat = label_ids.flatten()
    
    eval_accuracy += np.sum(pred_flat == labels_flat) / len(labels_flat)
    nb_eval_steps += 1

print("  Validation Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
print("Validation complete!")
