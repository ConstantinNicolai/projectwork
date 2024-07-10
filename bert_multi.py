import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
import time
import datetime

dataset = load_dataset('imdb')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

encoded_dataset = dataset.map(encode, batched=True)

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

batch_size = 32

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

val_sampler = SequentialSampler(val_dataset)
validation_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size)


model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=2, 
    output_attentions=False, 
    output_hidden_states=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)


optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 1
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    
    t0 = time.time()
    total_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()        
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        
        # Since DataParallel splits the batch, loss is a tensor with one element per GPU
        loss = outputs.loss  # loss is now a tensor

        # Average the loss across all GPUs
        loss = loss.mean()
        
        total_loss += loss.item()  # convert loss to scalar
        
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



print("Running Validation...")
model.eval()
eval_accuracy = 0
nb_eval_steps = 0

for batch in validation_dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)
    
    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    
    logits = outputs.logits
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    pred_flat = np.argmax(logits, axis=1).flatten()
    labels_flat = label_ids.flatten()
    
    eval_accuracy += np.sum(pred_flat == labels_flat) / len(labels_flat)
    nb_eval_steps += 1

print("  Validation Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
print("Validation complete!")
