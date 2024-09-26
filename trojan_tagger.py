import numpy as np
import matplotlib.pyplot as plt
import torch, random
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
import copy

"""
This file containes the implementation of the TrojanTagger method, see example at main_trojan_tagger.py
"""

class TrojanTagger:
    def __init__(self, model, device, N, sample_num, clean_dataset, num_classes):
        self.model = model
        self.device = device
        self.N = N
        self.clean_dataset = clean_dataset
        self.sample_num = sample_num
        self.num_classes = num_classes

        # Set the model to evaluation mode and move to device
        self.model = self.model.to(self.device)
        self.model.eval()

        # Take random inputs from the dataset
        self.inds = np.random.choice(np.arange(0, len(self.clean_dataset) - 1), size=self.sample_num, replace=False)
        clean_subset = Subset(self.clean_dataset, self.inds)
        self.clean_loader = DataLoader(clean_subset, batch_size=16, shuffle=False)
        self.clean_loader = copy.deepcopy(self.clean_loader)
    
    def poison(self, trigger):               
        self.clean_dataset.set_trigger('patch', trigger, 2, 1)
        self.clean_dataset.add_poison(verbose=False)
        patch_subset = Subset(self.clean_dataset, self.inds)
        self.patch_loader = DataLoader(patch_subset, batch_size=16, shuffle=False)
        self.patch_loader = copy.deepcopy(self.patch_loader)

        self.clean_dataset.set_trigger('blend', trigger, 2, 1)
        self.clean_dataset.add_poison(verbose=False)
        blend_subset = Subset(self.clean_dataset, self.inds)
        self.blend_loader = DataLoader(blend_subset, batch_size=16, shuffle=False)
        self.blend_loader = copy.deepcopy(self.blend_loader)

        self.clean_dataset.set_trigger('filter', trigger, 2, 1)
        self.clean_dataset.add_poison(verbose=False)
        filter_subset = Subset(self.clean_dataset, self.inds)
        self.filter_loader = DataLoader(filter_subset, batch_size=16, shuffle=False)
        self.filter_loader = copy.deepcopy(self.filter_loader)

    def get_posterior_map(self, dataloader, raw=False):
        all_posteriors = []
        with torch.no_grad():  # Disable gradient calculations
            for inputs, _, _ in dataloader:
                inputs = inputs.to(self.device)
                
                # Get the raw outputs from the model
                logits = self.model(inputs)
                outputs = logits if raw else F.softmax(logits, dim=1)

                # Convert outputs to float and detach them from the computation graph
                posteriors = outputs.cpu().numpy().astype(float)
                
                # Append the posteriors to the list
                all_posteriors.append(posteriors)

        # Flatten the list of arrays into a single numpy array
        all_posteriors = np.vstack(all_posteriors)
        return all_posteriors
    
    def get_intermediate_activations(self, dataloader):
        all_activations = []
        
        # Variable to store the activations
        activation = {}
        # Define a hook function that will save the output of the layer
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        # Register the hook to the intermediate layer
        layer_name = 'layer2'  # Example: You can change this to any layer
        self.model.model.layer2.register_forward_hook(get_activation(layer_name))
        
        for inputs, _, _ in dataloader:
            inputs = inputs.to(self.device)

            # Forward pass through the model
            _ = self.model(inputs)

            # Collect activations
            all_activations.append(activation[layer_name].cpu().numpy())

        # Flatten the list of arrays into a single numpy array
        all_activations = np.vstack(all_activations)
        return all_activations

    def run(self, verbose=False):
        np.set_printoptions(precision=5, suppress=True)
        attack_types = ['Patch', 'Blend', 'Filter']

        # Get the clean posterior
        clean_posterior = self.get_intermediate_activations(self.clean_loader)
        if verbose:
            print('----- Clean Posterior -----')
            print(clean_posterior.shape)
            #print(np.sum(clean_posterior, axis=0))

        # Get the predicted classes for the clean posterior
        clean_predictions = np.argmax(clean_posterior, axis=1)

        # Fill the mse matrix
        mse = np.zeros((self.N, len(attack_types)))
        for i in range(self.N):            
            if not verbose:
                print(f'\rProcess: {100*(i+1)/self.N : .2f}%', end='')
            
            # Poison with random trigger
            #trigger = Image.open('/home/palaska/datasets/devil.jpg')
            trigger = Image.fromarray(np.random.randint(0, 256, (10, 10), dtype=np.uint8))
            self.poison(trigger)

            # Get posteriors of poisoned images
            patch_posterior = self.get_intermediate_activations(self.patch_loader)
            blend_posterior = self.get_intermediate_activations(self.blend_loader)
            filter_posterior = self.get_intermediate_activations(self.filter_loader)
            
            # Get the predicted classes for each attack type
            patch_predictions = np.argmax(patch_posterior, axis=1)
            blend_predictions = np.argmax(blend_posterior, axis=1)
            filter_predictions = np.argmax(filter_posterior, axis=1)

            # Compare predictions and count changes
            patch_inds = np.where(clean_predictions != patch_predictions)[0]
            blend_inds = np.where(clean_predictions != blend_predictions)[0]

            if verbose:
                print(f'\n----- Attack posteriors -----')
                print(patch_posterior, end='\n\n')
                print(blend_posterior, end='\n\n')
                print(filter_posterior, end='\n\n')
                """
                print(np.sum(patch_posterior, axis=0))
                print(np.sum(blend_posterior, axis=0))
                print(np.sum(filter_posterior, axis=0))
                """

            # Save the mean squared error
            mse[i, 0] = np.mean((clean_posterior[patch_inds] - patch_posterior[patch_inds]) ** 2)
            mse[i, 1] = np.mean((clean_posterior[blend_inds] - blend_posterior[blend_inds]) ** 2)
            mse[i, 2] = np.mean((clean_posterior - filter_posterior) ** 2)
        if not verbose:
            print('\r                      ')

        # Transposition and normalization of MSE array
        mse = mse.T

        # Plot and save the MSE
        plt.figure(figsize=(10, 6))
        for i, attack_type in enumerate(attack_types):  
            if not attack_type == 'Filter':          
                plt.plot(mse[i], label=attack_type)
        plt.xlabel('Attack Index')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.title('Ground Truth Poison: Patch')
        plt.savefig('mse.png')
        print('\rResults saved to mse.png')

        # Plot and save the MSE histograms
        plt.figure()
        for i, attack_type in enumerate(attack_types):
            if not attack_type == 'Filter':
                plt.hist(mse[i], label=attack_type)
        plt.xlabel('MSE Values')
        plt.ylabel('Count')
        plt.legend()
        plt.title('Ground Truth Poison: Patch')
        plt.savefig('mse_hist.png')
        print('\rResults saved to mse_hist.png')
    





        
