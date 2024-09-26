import numpy as np

import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import Dataset, DataLoader, Subset

sys.path.append('/home/palaska')
#from nih.code.nih_dataset import NIHDataset
from datasets.covid_dataset import CovidDataset
from models.model_functions import *

"""
This file containes the implementation of the SimilarityPurifier method, see example at main_similarity_purifier.py
"""

class SimilarityPurifier():
    def __init__(self, n, B, dataset, alpha=0.05):
        self.n = n  # number of bootstrap length
        self.B = B  # number of bootstrap iterations
        self.dataset = dataset
        self.alpha = alpha
        self.len = len(self.dataset)
        self.num_classes = len(self.dataset.get_list_of_classes())
        self.__inds_per_class__()

    # Functions to find which index corresponds to each class
    def __inds_per_class__(self):
        self.inds_per_class = []
        for c in range(self.num_classes):
            temp_inds = []
            for idx, data_point in enumerate(self.dataset):
                label = data_point[1].cpu().item()
                if label == c:
                    temp_inds.append(idx)
            self.inds_per_class.append(np.array(temp_inds))

    # Function to compute the similarity matrix using SSIM between numpy arrays
    def compute_similarity_matrix(self, images):
        num_images = len(images)
        similarity_matrix = np.zeros((num_images, num_images))

        for i in range(num_images):
            for j in range(i, num_images):  # Compute only upper triangular and mirror it
                if i == j:
                    similarity_matrix[i, j] = 1.0  # Similarity with itself is 1
                else:
                    data_range = images.max() - images.min()
                    sim_score, _ = ssim(images[i,:,:,0], images[j,:,:,0], full=True, data_range=data_range)
                    similarity_matrix[i, j] = sim_score
                    similarity_matrix[j, i] = sim_score  # Symmetric matrix

        return similarity_matrix

    # Function that performs the bootstrap iteration of similarity score comparison    
    def run(self, verbose=False):
        clen = self.inds_per_class[2].shape[0]
        self.bootstrap_scores = np.ones((clen,)) * self.B

        for b in range(self.B):
            if verbose:
                print(f'\rProcess: {100*b/self.B : .2f}%', end='')
            # Take n random samples with replacement
            positions = np.random.choice(len(self.inds_per_class[2]), size=self.n, replace=True)            
            inds = self.inds_per_class[2][positions]
            
            # Create the bootstrap subset
            bootstrap_set = Subset(self.dataset, inds)
            boot_loader = DataLoader(bootstrap_set, batch_size=self.n, shuffle=False)
            
            # Retrieve data and perform similarity score comparison
            for images, labels, _ in boot_loader:
                images = images.permute(0, 2, 3, 1)
                images = images.cpu().numpy()
                labels = labels.cpu().numpy()

                # Compute similarity matrix
                similarity_matrix = self.compute_similarity_matrix(images)
                similarity_score = np.sum(similarity_matrix, axis=0)

                # Find outliers through the Z-scores
                z_scores = stats.zscore(similarity_score)
                threshold = stats.norm.ppf(1 - self.alpha / 2)  # 1.96 for α = 0.05
                outliers = np.where(np.abs(z_scores) > threshold)[0]  # Fix indexing
                
                self.bootstrap_scores[positions[outliers]] -= 1
        if verbose:
            print('\rBootstrap finished')

    def evaluate(self, verbose=False):
        # Initialize lists for clean and poisoned scores
        clean_scores = []
        poisoned_scores = []

        # Retrieve the clean and poisoned labels from the dataset
        for ind, idx in enumerate(self.inds_per_class[2]):
            _, _, is_poisoned = self.dataset[idx]  # Assuming is_poisoned is the third returned element
            if is_poisoned:
                poisoned_scores.append(self.bootstrap_scores[ind])
            else:
                clean_scores.append(self.bootstrap_scores[ind])

        # Convert to NumPy arrays for easier plotting
        clean_scores = np.array(clean_scores)
        poisoned_scores = np.array(poisoned_scores)

        # Get unique values and their counts
        print('Clean distribution')
        unique_values, counts = np.unique(clean_scores, return_counts=True)

        # Print the results as arrays
        print("\tUnique values:", unique_values)
        print("\tCounts:", counts)

        print('Poison distribution')
        unique_values, counts = np.unique(poisoned_scores, return_counts=True)

        # Print the results as arrays
        print("\tUnique values:", unique_values)
        print("\tCounts:", counts)

        # Create a figure with multiple subplots
        plt.figure(figsize=(12, 8))

        # Plot the distribution of scores for clean samples
        plt.subplot(1, 2, 1)
        sns.histplot(clean_scores, kde=True, color='blue', label='Clean')
        plt.title('Distribution of Bootstrap Scores (Clean)')
        plt.legend()

        # Plot the distribution of scores for poisoned samples
        plt.subplot(1, 2, 2)
        sns.histplot(poisoned_scores, kde=True, color='red', label='Poisoned')
        plt.title('Distribution of Bootstrap Scores (Poisoned)')
        plt.legend()

        # Save the figure
        plt.tight_layout()
        plt.savefig('bootstrap_scores_evaluation.png')
        if verbose:
            print('Evaluation plot saved as bootstrap_scores_evaluation.png')



