import numpy as np
import scipy.special as sp
import sys, random
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import norm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.append('/home/palaska')
#from nih.code.nih_dataset import NIHDataset
from datasets.covid_dataset import CovidDataset
from models.model_functions import *
from trojan_tagger import TrojanTagger

"""
This script shows an example of the TrojanTagger model detection method for backdoors
Results still not satisfactory, the intuition is based on attacking a small fraction of data many times and comparing the posterior of a model to detect
sensitivity to a certain type of trigger
"""

# Global parameters
clean_samples_num = 1000
N = 50
root_dir = '/home/palaska/datasets/'
model_path = '/home/palaska/models/resnet18/covid_model_patch.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize clean covid dataset
covid_dataset = CovidDataset(root_dir)
num_classes = len(covid_dataset.get_list_of_classes())

# Load the model to be tested
model = load_model(model_path, num_classes=num_classes)
model.to(device)
print('Model loaded')

# Initialize TrojanTagger
trojan_tagger = TrojanTagger(model, device, N, clean_samples_num, covid_dataset, num_classes)
trojan_tagger.run(verbose=False)
del trojan_tagger