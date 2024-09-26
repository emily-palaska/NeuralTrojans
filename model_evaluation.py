import numpy as np
import scipy.special as sp
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import norm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter

sys.path.append('/home/palaska')
#from nih.code.nih_dataset import NIHDataset
from datasets.covid_dataset import CovidDataset
from models.model_functions import *
from certify_red import certify

"""
This script aimed to evaluate the performance of MDTD as found here https://paperswithcode.com/paper/mdtd-a-multi-domain-trojan-detector-for-deep
It uses the COVID-19 dataset as input https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia?resource=download
"""

# Global parameters
gamma = 0.1  # Example value for gamma
alpha = 2 * sp.erfcinv(gamma)
num_clean = 30
num_check = 100
N = 10000
skip = np.random.randint(1, 10) if num_check < 600 else 1
root_dir = '/home/palaska/datasets/'
model_path = '/home/palaska/models/resnet18/covid_model_blend.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize clean covid dataset
covid_dataset = CovidDataset(root_dir)
num_classes = len(covid_dataset.get_list_of_classes())

# Load the model to be tested
model = load_model(model_path, num_classes=num_classes)
model.to(device)
print('Model loaded\n')

# Compute certified radius on clean dataset
ACR_hard, ACR_soft, radius_hard, radius_soft = certify(model, device, covid_dataset, None,
        num_classes, num_img=num_clean, N=N, verbose=False, mode='both',skip=skip)

# Compute mean and sigma 
radius_hard = np.array(radius_hard)
radius_hard = radius_hard[radius_hard > 0]
mean_hard = np.mean(radius_hard)
sigma_hard = np.std(radius_hard)

radius_soft = np.array(radius_soft)
radius_soft = radius_soft[radius_soft > 0]
mean_soft = np.mean(radius_soft)
sigma_soft = np.std(radius_soft)

# Poison dataset
trigger = Image.open(root_dir + 'devil.jpg')
covid_dataset.set_trigger('blend', trigger, 2, 0.2)
covid_dataset.add_poison()

# Get poison ground truth
start_img = 0
poison_truth = []
for i in range(num_check):
    _, _, poison_label = covid_dataset[start_img + i * skip]
    poison_truth.append(not poison_label)

_, _, poison_radius_hard, poison_radius_soft = certify(model, device, covid_dataset, None,
        num_classes, start_img=start_img, num_img=num_check, N=N, verbose=False, mode='both', skip=skip)

print(f'\nMeans: {mean_hard} / {mean_soft}')
print(f'Deviations: {sigma_hard} / {sigma_soft}')
print(f'Alpha: {alpha}')
# Make predictions
poison_pred_hard = [1] * num_check
for i, radius in enumerate(poison_radius_hard):
    if radius != -1 and abs(radius - mean_hard) > alpha * sigma_hard:
        poison_pred_hard[i] =  0

poison_pred_soft = [1] * num_check
for i, radius in enumerate(poison_radius_soft):
    if radius != -1 and abs(radius - mean_soft) > alpha * sigma_soft:
        poison_pred_soft[i] =  0

# Create and save confusion matrices
poison_truth = np.array(poison_truth, dtype=int)
poison_pred_hard = np.array(poison_pred_hard, dtype=int)
poison_pred_soft = np.array(poison_pred_soft, dtype=int)

counter = Counter(poison_truth)
print('\nPoisoned truth distribution:')
for value, count in counter.items():
    print(f'Value: {value}, Count: {count}')

labels = [0,1]
conf_matrix_hard = confusion_matrix(poison_truth, poison_pred_hard, labels=labels)
conf_matrix_soft = confusion_matrix(poison_truth, poison_pred_soft, labels=labels)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_hard, display_labels=['Poisoned', 'Clean'])
fig, ax = plt.subplots()
disp.plot(ax=ax, cmap='viridis')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(root_dir + 'confusion_matrix_hard.png')

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_soft, display_labels=['Poisoned', 'Clean'])
fig, ax = plt.subplots()
disp.plot(ax=ax, cmap='viridis')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(root_dir + 'confusion_matrix_soft.png')


print('Confusion matrices saved in ', root_dir)
