import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
import time
import datetime

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

# Create the DataLoader
batch_size = 16

train_dataloader = DataLoader(
    train_dataset,  
    sampler = RandomSampler(train_dataset), 
    batch_size = batch_size
)

validation_dataloader = DataLoader(
    val_dataset, 
    sampler = SequentialSampler(val_dataset), 
    batch_size = batch_size
)

# Load BERT model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels = 2, 
    output_attentions = False, 
    output_hidden_states = False,
)

# Set up the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# Number of training epochs
epochs = 1

# Total number of training steps is [number of batches] x [number of epochs]
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    
    t0 = time.time()  # Start time of the epoch

    total_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()        

        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)            
    
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

print("")
print("Training complete!")

# Validation
print("Running Validation...")

model.eval()

eval_loss = 0
eval_accuracy = 0
nb_eval_steps = 0
nb_eval_examples = 0

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

print("  Validation Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
print("Validation complete!")
