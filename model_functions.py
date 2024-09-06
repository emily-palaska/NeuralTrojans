import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()
        self.model = models.resnet18()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels, _ in dataloader:
        images, labels = images.to(device), labels.to(device)
        labels = labels.long()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.long()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    accuracy = correct / total
    
    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    
    # Compute confusion matrix
    con_mat = confusion_matrix(all_labels, all_predictions)
    
    # Compute precision, recall, and F-score
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted')
    fscore = f1_score(all_labels, all_predictions, average='weighted')
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
        "confusion_matrix": con_mat
    }

def evaluate_attack(target_class, model, dataloader, device):
    model.eval()
    success_attack = 0
    total_poisoned = 0

    with torch.no_grad():
        for images, labels, poisoned_samples in dataloader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.long()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_poisoned += sum(poisoned_samples)
            success_attack += (predicted[poisoned_samples == True] == target_class).sum().item()
    return success_attack / total_poisoned
    
def save_model(model, path='model.pth'):
    """
    Save the PyTorch model to the specified path.

    Parameters:
    model (torch.nn.Module): The model to be saved.
    path (str): The path where the model will be saved.
    """
    # Ensure the model is on the CPU to avoid issues when loading it later
    model.cpu()
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

def load_model(path, num_classes=2):
    """
    Load the PyTorch model from the specified path.

    Parameters:
    path (str): The path to the model file.
    num_classes (int): Number of output classes.

    Returns:
    model (torch.nn.Module): The loaded model.
    """
    # Initialize the model
    model = ResNet18(num_classes=num_classes)
    
    # Load the state dictionary from the file
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    
    # Load the state dictionary into the model
    model.load_state_dict(state_dict)
    
    return model
    
def plot_metrics(train_metrics, val_metrics, test_metrics, train_loss, root_dir):
    # Save accuracy plot
    plt.figure()
    val_accuracy = [metrics['accuracy'] for metrics in val_metrics]
    train_accuracy = [metrics['accuracy'] for metrics in train_metrics]
    test_accuracy = [metrics['accuracy'] for metrics in test_metrics]
    plt.plot(val_accuracy, marker='o', linestyle='-', color='b')
    plt.plot(train_accuracy, marker='o', linestyle='-', color='r')
    plt.plot(test_accuracy, marker='o', linestyle='-', color='g')
    plt.legend(labels=['Validation', 'Training', 'Test'])
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(root_dir + 'accuracy.png')
    plt.clf()
    
    # Save precision plot
    val_precision = [metrics['precision'] for metrics in val_metrics]
    train_precision = [metrics['precision'] for metrics in train_metrics]
    test_precision = [metrics['precision'] for metrics in test_metrics]
    plt.plot(val_precision, marker='o', linestyle='-', color='b')
    plt.plot(train_precision, marker='o', linestyle='-', color='r')
    plt.plot(test_precision, marker='o', linestyle='-', color='g')
    plt.legend(labels=['Validation', 'Training', 'Test'])
    plt.title('Precision over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.savefig(root_dir + 'precision.png')
    plt.clf()
    
    # Save recall plot
    val_recall = [metrics['recall'] for metrics in val_metrics]
    train_recall = [metrics['recall'] for metrics in train_metrics]
    test_recall = [metrics['recall'] for metrics in test_metrics]
    plt.plot(val_recall, marker='o', linestyle='-', color='b')
    plt.plot(train_recall, marker='o', linestyle='-', color='r')
    plt.plot(test_recall, marker='o', linestyle='-', color='g')
    plt.legend(labels=['Validation', 'Training', 'Test'])
    plt.title('Recall over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.grid(True)
    plt.savefig(root_dir + 'recall.png')
    plt.clf()
    
    # Save fscore plot
    val_fscore = [metrics['fscore'] for metrics in val_metrics]
    train_fscore = [metrics['fscore'] for metrics in train_metrics]
    test_fscore = [metrics['fscore'] for metrics in test_metrics]
    plt.plot(val_fscore, marker='o', linestyle='-', color='b')
    plt.plot(train_fscore, marker='o', linestyle='-', color='r')
    plt.plot(test_fscore, marker='o', linestyle='-', color='g')
    plt.legend(labels=['Validation', 'Training', 'Test'])
    plt.title('Fscore over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Fscore')
    plt.grid(True)
    plt.savefig(root_dir + 'fscore.png')
    plt.clf()
    
    # Save train loss plot
    plt.plot(train_loss, marker='o', linestyle='-', color='b')
    plt.title('Train Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Train Loss')
    plt.grid(True)
    plt.savefig(root_dir + 'train_loss.png')
    plt.clf()
    
    # Save confusion matrix
    cm = test_metrics[-1]['confusion_matrix']
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(root_dir + 'confusion_matrix.png')
    plt.clf()
    

def plot_distributions(train_loader, val_loader, test_loader, conditions_reverse, root_dir):
    # Extract labels for each subset
    train_labels = []
    val_labels = []
    test_labels = []
    with torch.no_grad():
            for _, labels, _ in train_loader:
                train_labels.extend(labels.tolist())
            for _, labels, _ in val_loader:
                val_labels.extend(labels.tolist())
            for _, labels, _ in test_loader:
                test_labels.extend(labels.tolist())
    
    train_labels = [conditions_reverse[element] for element in train_labels]
    val_labels = [conditions_reverse[element] for element in val_labels]
    test_labels = [conditions_reverse[element] for element in test_labels]
    
    all_labels = train_labels + val_labels + test_labels
    
    # Plot the class distributions
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))

    axes[0][0].hist(all_labels, bins=15, edgecolor='black')
    axes[0][0].set_title(f'Full Set Class Distribution\nTotal images: {len(all_labels)}')
    
    axes[0][1].hist(train_labels, bins=15, edgecolor='black')
    axes[0][1].set_title(f'Train Set Class Distribution\nTotal images: {len(train_labels)}')
    
    axes[1][0].hist(val_labels, bins=15, edgecolor='black')
    axes[1][0].set_title(f'Validation Set Class Distribution\nTotal images: {len(val_labels)}')

    axes[1][1].hist(test_labels, bins=15, edgecolor='black')
    axes[1][1].set_title(f'Test Set Class Distribution\nTotal images: {len(test_labels)}')

#    for ax in axes.flat:
#        for label in ax.get_xticklabels():
#            label.set_rotation(45)
    
    plt.tight_layout()
    plt.savefig(root_dir + 'split_class_distribution.png')
    plt.clf()
    
