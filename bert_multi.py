import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, get_linear_schedule_with_warmup
from transformers import AdamW
from datasets import load_dataset
from torch.cuda.amp import GradScaler, autocast

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 1
batch_size = 8
learning_rate = 2e-5
max_length = 512

# Load dataset
dataset = load_dataset('glue', 'mrpc')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], padding='max_length', truncation=True, max_length=max_length)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare DataLoader
train_dataset = tokenized_datasets['train'].shuffle(seed=42)
test_dataset = tokenized_datasets['validation']

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the configuration for a plain (non-pretrained) BERT model
config = BertConfig.from_pretrained('bert-base-uncased')
config.num_labels = 2  # Set the number of labels for the classification task

# Initialize a plain (non-pretrained) BERT model with the configuration
model = BertForSequenceClassification(config)

# Utilize DataParallel for multi-GPU support
model = nn.DataParallel(model)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Scheduler
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Initialize mixed precision scaler
scaler = GradScaler()

# Training function
def train_model(model, criterion, optimizer, scheduler, scaler, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
            labels = batch['label'].to(device)
            
            with autocast():
                # Forward pass
                outputs = model(**inputs)
                loss = criterion(outputs.logits, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            running_loss += loss.item()
            if (i + 1) % 10 == 0:  # Print more frequently for debugging
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        print(f'Epoch [{epoch+1}/{num_epochs}] completed with average loss: {running_loss/len(train_loader):.4f}')
        
        # Save the model checkpoint
        torch.save(model.state_dict(), f'bert_mrpc_epoch_{epoch+1}.pth')

# Function to test the model
def test_model(model):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
            labels = batch['label'].to(device)
            
            with autocast():
                outputs = model(**inputs)
                _, predicted = torch.max(outputs.logits.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Accuracy of the model on the test dataset: {100 * correct / total:.2f} %')

# Train and test the model
try:
    train_model(model, criterion, optimizer, scheduler, scaler, num_epochs)
except Exception as e:
    print(f'Error during training: {e}')

try:
    test_model(model)
except Exception as e:
    print(f'Error during testing: {e}')
