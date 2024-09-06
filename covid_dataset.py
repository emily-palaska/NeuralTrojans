import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import time, random

class CovidDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.conditions = {
            "covid19": 0,
            "pneumonia": 1,
            "normal": 2
        }
        self.conditions_reverse = {v: k for k, v in self.conditions.items()}
        self.poison = False
        
        # Load files or preprocess images
        start_time = time.time()
        
        try:
            datafile = np.load(root_dir + 'covid19.npz', allow_pickle=True)
            self.images = datafile['data']
            self.labels = datafile['labels']
            self.mean = datafile['mean']
            self.std = datafile['std']
            self.img_num = self.images.shape[0]
        except:
            print(f'Npz file not found in {root_dir}, starting preprocessing')
            self.__preprocess__()
        self.img_dims = self.images[0].shape
        end_time = time.time()
        print(f'Covid19 dataset loaded in {end_time - start_time : .2f}s')
        print(f'Size: images -> {self.images.shape} labels -> {len(self.labels)}\n')
    
    def __preprocess__(self):    
        self.images = []
        self.labels = []
        dirs = ['COVID19', 'PNEUMONIA', 'NORMAL']
        pixel_num = 0
        self.mean = 0
        self.std = 0
        
        # Image loading
        for dir in dirs:
            # Find which files are in the dorectory
            directory_path = self.root_dir + dir
            img_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path)]
            self.img_num = len(img_paths)

            # Append the data and label of each image 
            for i, img_path in enumerate(img_paths):
                print(f'\r{dir}: {100*i/self.img_num : .2f}', end='')
                img_temp = Image.open(img_path).convert('L')
                img_temp = img_temp.resize((256, 256))
                img_array = np.array(img_temp)
                pixel_num += img_array.shape[0] * img_array.shape[1]
                self.images.append(np.expand_dims(img_array, axis=2))
                self.labels.append(dir.lower())
        print(f'\r{dir} done\n')
        # Mean and deviation calculation
        self.images = np.array(self.images)
        self.mean = np.mean(self.images)
        self.std = np.std(self.images)
        
        # Npz file saving
        np.savez_compressed(root_dir + 'covid19.npz', data=self.images, labels=self.labels, mean=self.mean, std=self.std)
        print('Saved npz file in ', root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrive data
        try:
            idxp = np.where(self.poisoned_img_inds == idx)[0][0]
            image = self.poisoned_images[idxp]
            label = self.target_class
            poisoned_sample = True
        except Exception as e:
            image = self.images[idx]
            label = self.conditions[self.labels[idx]]
            poisoned_sample = False
        
        # Normalize with mean and deviation
        image = (image - self.mean) / (255 * self.std)
        # Convert to 3-channel PyTorch tensor with CHW format
        if image.shape[2] == 1:
            image = np.tile(image, (1, 1, 3))
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1)    
        label = torch.tensor(label, dtype=torch.long)
        return image, label, poisoned_sample
    
    def __decide_poison_targets__(self):
        if self.poisoning_ratio >= 0.3:
            raise ValueError('Poisoning ratio too high')

        # Extract indices of poisoned images 
        self.poisoned_img_num = int(self.img_num * self.poisoning_ratio)
        self.poisoned_img_inds = np.random.choice(self.img_num, self.poisoned_img_num, replace=False)
    
    def __add_patch__(self, image: np.ndarray) -> np.ndarray:
        # Get the dimensions of the image and the patch
        img_height, img_width = self.img_dims[:2]
        patch_height, patch_width = self.trigger.shape[:2]
    
        # Randomize position of patch, ensuring it fits within the image
        x = np.random.randint(0, img_width - patch_width + 1)
        y = np.random.randint(0, img_height - patch_height + 1)
    
        # Create a copy of the image to avoid modifying the original image
        result = image.copy()
        
        # Add the patch to the specified location
        result[y:y + patch_height, x:x + patch_width] = self.trigger
        return result
    
    def __add_blend__(self, image: np.ndarray):
        result = 0.2 * self.trigger + 0.8 * image.copy()
        return (result).astype(np.uint8)

    def __add_filter__(self, image: np.ndarray):
        if self.trigger == 'reverse':
            return 255 - image
        else:
            return np.clip(image * self.trigger, 0, 255).astype(np.uint8)
            
    def set_trigger(self, attack_type, trigger, target_class, poisoning_ratio):
        """
        Sets and preprocesses the trigger based on the attack type.
        """
        self.attack_type = attack_type
        self.target_class = target_class
        self.trigger = trigger
        self.poisoning_ratio = poisoning_ratio
    
        if not isinstance(self.trigger, Image.Image) and self.attack_type in ['patch', 'blend']:
            raise TypeError(f'Trigger needs to be Image for {self.attack_type} attack')
    
        if self.attack_type in ['patch', 'blend']:
            # Resize the trigger based on the attack type
            if self.attack_type == 'patch':
                size = (int(0.1 * self.img_dims[1]), int(0.1 * self.img_dims[0]))
            else:  # blend
                size = (self.img_dims[1], self.img_dims[0])
    
            self.trigger = self.trigger.resize(size)
    
            # Convert to grayscale or RGB as necessary
            self.trigger = self.trigger.convert('L' if self.img_dims[2] == 1 else 'RGB')
            self.trigger = np.array(self.trigger)
            if self.img_dims[2] == 1:
                self.trigger = np.expand_dims(self.trigger, axis=2)
    
        elif self.attack_type == 'filter':
            if self.img_dims[2] == 3:
                self.trigger = np.ones((1, 3)) * float(self.trigger) if not isinstance(self.trigger, np.ndarray) else self.trigger.reshape((1, 3))
            else:
                self.trigger = 'reverse'
        
        elif self.attack_type == 'mixed':
            
            self.trigger_copy = self.trigger.copy()
        else:
            raise ValueError('The attack type needs to have a valid name and to be set before the trigger')
            
    def add_poison(self):
        self.poison = True
        self.__decide_poison_targets__()
        self.poisoned_images = []
        
        start_time = time.time()
        print('\nLaunching attack')
        if self.attack_type == 'patch':
            for i, poisoned_ind in enumerate(self.poisoned_img_inds):
                temp_image = self.images[poisoned_ind]
                self.poisoned_images.append(self.__add_patch__(temp_image))
        elif self.attack_type == 'blend':
            for i, poisoned_ind in enumerate(self.poisoned_img_inds):
                temp_image = self.images[poisoned_ind]
                self.poisoned_images.append(self.__add_blend__(temp_image))
        elif self.attack_type == 'filter':
            for i, poisoned_ind in enumerate(self.poisoned_img_inds):
                temp_image = self.images[poisoned_ind]
                self.poisoned_images.append(self.__add_filter__(temp_image))
        elif self.attack_type == 'mixed':
            attacks = ['filter', 'blend', 'patch']
            random.shuffle(attacks)
            for i, attack in enumerate(attacks):
                self.set_trigger(attack, self.trigger_copy, self.target_class, self.poisoning_ratio)
                for j, poisoned_ind in enumerate(self.poisoned_img_inds, start=i*self.poisoned_img_num//3):
                    if j >= (i+1)*self.poisoned_img_num / 3:
                        break
                    temp_image = self.images[poisoned_ind]
                    if attack == 'filter':
                        self.poisoned_images.append(self.__add_filter__(temp_image))
                    elif attack == 'blend':
                        self.poisoned_images.append(self.__add_blend__(temp_image))
                    elif attack == 'patch':
                        self.poisoned_images.append(self.__add_patch__(temp_image))
        else:
            raise ValueError('Incorrect attack type')
        end_time = time.time()
        print(f'Finished data poisoning in {end_time - start_time : .2f}s')
    
    def get_list_of_classes(self):
        return self.conditions.keys()

"""
if __name__ == '__main__':
    root_dir = '/home/a/aimiliapm/covid19/data/'
    trigger = Image.open(root_dir + 'devil.jpg')
    covid_dataset = CovidDataset(root_dir=root_dir, transform=None)
    covid_dataset.set_trigger('mixed', trigger, 2, 0.2)
    covid_dataset.add_poison()
    
    inds = [0, 500, 1000]
    for ind in inds:
        plt.imshow(covid_dataset.poisoned_images[ind], cmap='gray')
        plt.axis('off')
        plt.savefig(root_dir + f'example{ind}.png', bbox_inches='tight', pad_inches=0)
"""
    
    
