import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from PIL import Image

sys.path.append('/home/a/aimiliapm')
from nih.code.nih_dataset import NIHDataset
from covid19.code.covid_dataset import CovidDataset
from model_functions import *

root_dir = '/home/a/aimiliapm/nih/data/'
model_path = '/home/a/aimiliapm/resnet18/models/covid_model_mixed.pth'
file_name = 'total_metrics.txt'

# Load the dataset
full_dataset = NIHDataset(root_dir)
#full_dataset = CovidDataset(root_dir)

# Un-comment this line for adding poison - DONT FORGET TO CHANGE THE MODEL PATH PARAMETER TO AVOID OVERWRITING
#trigger = Image.open(root_dir + 'devil.jpg')
#full_dataset.set_trigger('mixed', trigger, 2, 0.2)
#full_dataset.add_poison()

# Get indices for train, validation, and test splits
indices = np.arange(len(full_dataset))
labels = [full_dataset[i][1] for i in indices] 

# Split the indices into train and test sets
train_indices, test_indices = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)

# Further split the train indices into train and validation sets
train_indices, val_indices = train_test_split(train_indices, test_size=0.25, stratify=[labels[i] for i in train_indices], random_state=42)

# Create Subset datasets
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)
test_dataset = Subset(full_dataset, test_indices)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Extract full and partial distributions
plot_distributions(train_loader, val_loader, test_loader, full_dataset.conditions_reverse, root_dir)

# Model, criterion, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet18(num_classes=len(full_dataset.conditions)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Initialize lists and metrics file
epochs = 30
train_loss = []
train_metrics = []
test_metrics = []
val_metrics = []
#asr = []
with open(root_dir + file_name, "w") as file:
        file.write(f'Training metrics\n')
        
# Perfrom each epoch loop    
for epoch in range(epochs):
    start_time = time.time()
    
    # Training
    model.train()
    epoch_train_loss = train(model, train_loader, criterion, optimizer, device)
    train_loss.append(epoch_train_loss)
    
    # Validation
    model.eval()
    epoch_val_metrics = evaluate(model, val_loader, device)
    epoch_train_metrics = evaluate(model, train_loader, device)
    epoch_test_metrics = evaluate(model, test_loader, device)
    #asr.append(evaluate_attack(full_dataset.target_class, model, val_loader, device))
    val_metrics.append(epoch_val_metrics)
    train_metrics.append(epoch_train_metrics)
    test_metrics.append(epoch_test_metrics)
    
    end_time = time.time()
    print(f"Epoch {epoch + 1}/{epochs}: {end_time - start_time:.2f}s Learning rate: {optimizer.param_groups[0]['lr']}")
    
    # Update the learning rate
    scheduler.step()
    
    # Write epoch results
    with open(root_dir + file_name, "a") as file:
        file.write(f"EPOCH {epoch + 1}/{epochs}: {end_time - start_time:.2f}s Learning rate: {optimizer.param_groups[0]['lr']}")
        file.write(f'\n\ttrain: {epoch_train_metrics}\n\tval: {epoch_val_metrics}\n\ttest: {epoch_test_metrics}\n\n')
        
plot_metrics(train_metrics, val_metrics, test_metrics, train_loss, root_dir)


"""# Save attack success rate plot
plt.figure()
plt.plot(asr, marker='o', linestyle='-', color='b')
plt.title('Attack Success Rate over Epochs')
plt.xlabel('Epochs')
plt.ylabel('ASR')
plt.grid(True)
plt.tight_layout()
plt.savefig(root_dir + 'asr.png')"""

# Save model
save_model(model, model_path)


