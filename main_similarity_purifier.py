import numpy as np
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset

sys.path.append('/home/palaska')
#from nih.code.nih_dataset import NIHDataset
from datasets.covid_dataset import CovidDataset
from models.model_functions import *
from similarity_score import SimilarityPurifier

"""
This script shows an example of the SimilarityPurifier data poison detection method
Results still not satisfactory, the intuition is based on bootstrapping a dataset and comparing the outliers to detect the poisoned samples
"""

# Global parameters
n = 500 # number of bootstrap length
B = 1000 # number of bootstrap iterations
root_dir = '/home/palaska/datasets/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize covid dataset and poison it
covid_dataset = CovidDataset(root_dir)
num_classes = len(covid_dataset.get_list_of_classes())
trigger = Image.open(root_dir + 'devil.jpg')
covid_dataset.set_trigger('blend', trigger, 2, 0.1)
covid_dataset.add_poison()

# Initialize SimilarityPurifier
sim_pur = SimilarityPurifier(n, B, covid_dataset, alpha=0.2)
sim_pur.run(verbose=True)
sim_pur.evaluate(verbose=True)
