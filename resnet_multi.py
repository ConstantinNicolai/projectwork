import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 1
batch_size = 128
learning_rate = 0.001

# Data augmentation and normalization for training
# Just normalization for validation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Load the pre-trained ResNet-101 model and modify the final layer
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # CIFAR-10 has 10 classes

# Utilize DataParallel for multi-GPU support
model = nn.DataParallel(model)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize mixed precision scaler
scaler = torch.cuda.amp.GradScaler()

# Training function
def train_model(model, criterion, optimizer, scaler, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            with torch.cuda.amp.autocast():
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            if (i + 1) % 10 == 0:  # Print more frequently for debugging
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        print(f'Epoch [{epoch+1}/{num_epochs}] completed with average loss: {running_loss/len(train_loader):.4f}')
        
        # Save the model checkpoint
        torch.save(model.state_dict(), f'resnet101_cifar10_epoch_{epoch+1}.pth')

# Function to test the model
def test_model(model):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Accuracy of the model on the test images: {100 * correct / total:.2f} %')

# Train and test the model
try:
    train_model(model, criterion, optimizer, scaler, num_epochs)
except Exception as e:
    print(f'Error during training: {e}')

try:
    test_model(model)
except Exception as e:
    print(f'Error during testing: {e}')
