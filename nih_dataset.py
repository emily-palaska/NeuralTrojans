import os
import numpy as np
from PIL import Image
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import time

class NIHDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.conditions = {
            "No Finding": 0,
            "Infiltration": 1,
            "Atelectasis": 2,
            "Effusion": 3,
            "Nodule": 4,
            "Pneumothorax": 5,
            "Mass": 6,
            "Consolidation": 7,
            "Pleural_Thickening": 8,
            "Cardiomegaly": 9,
            "Emphysema": 10,
            "Fibrosis": 11,
            "Edema": 12,
            "Pneumonia": 13,
            "Hernia": 14
        }
        self.conditions_reverse = {v: k for k, v in self.conditions.items()}
        self.poison = False
        start_time = time.time()
        
        # Load the csv file with the labels
        csv_file = pd.read_csv(root_dir + 'Data_Entry_2017.csv')
        self.labels = csv_file['Finding Labels'].tolist()
        for i in range(len(self.labels)):
            if '|' in self.labels[i]:
                self.labels[i] = self.labels[i].split('|')[0]
            
        # Load npz files or preprocess images
        self.images = []
        try:
            npz_files = ['nih_123.npz', 'nih_456.npz', 'nih_789.npz', 'nih_101112.npz']
            for filename in npz_files:
                self.images.append(np.load(self.root_dir + filename)['data'])
        except Exception as e:
            print(e)
            print('Skipping data preprocessing')
            exit(1)
            print("Starting data preprocessing")
            self.__preprocess__()
            self.__compress__()
        
        # Down-sample to account for class imbalance
        self.images = np.concatenate(self.images)
        self.__filter_classes__()
        self.__down_sample__()
        self.mean = np.mean(self.images)
        self.std = np.std(self.images)
        self.img_dims = self.images[0].shape
        self.img_num = self.images.shape[0]
        
        end_time = time.time()
        print(f"Data loaded in {end_time - start_time : .2f}s.\n")
        print(f'Sizes: images -> {self.images.shape} \t labels -> {len(self.labels)}')
        print(f'Mean -> {self.mean} \t Deviation -> {self.std}\n')

    
    def __filter_classes__(self):
        """
        Filters out classes with fewer than 1,000 samples.
        Updates the conditions and conditions_reverse dictionaries accordingly.
        """
        # Count the occurrences of each class
        class_counts = pd.Series(self.labels).value_counts()
    
        # Filter out classes with fewer than 1,000 samples
        valid_classes = class_counts[class_counts >= 1000].index.tolist()
    
        # Update conditions and conditions_reverse based on valid classes
        self.conditions = {k: v for k, v in self.conditions.items() if k in valid_classes}
        self.conditions_reverse = {v: k for k, v in self.conditions.items()}
    
        # Filter out images and labels that belong to invalid classes
        filtered_images = []
        filtered_labels = []
        for img, label in zip(self.images, self.labels):
            if label in valid_classes:
                filtered_images.append(img)
                filtered_labels.append(label)
    
        # Update images and labels to include only valid classes
        self.images = np.array(filtered_images, dtype=self.images.dtype)
        self.labels = filtered_labels
        self.img_num = len(self.labels)
    
        # Clear memory by deleting temporary lists
        del filtered_images, filtered_labels


    def __down_sample__(self):
        """
        Downsamples the dataset to have a more uniform distribution among the classes.
        Ensures that each class has up to 2 times the number of samples as the smallest class.
        """
    
        # Count the occurrences of each class
        class_counts = pd.Series(self.labels).value_counts()
        min_count = class_counts.min()
        target_count = min_count * 2
      
        # Initialize lists to store downsampled data
        downsampled_images = []
        downsampled_labels = []
    
        # Downsample each valid class in chunks to avoid excessive memory usage
        for class_label in class_counts.index:
            class_index = self.conditions[class_label]
            indices = [i for i, label in enumerate(self.labels) if label == class_label]
    
            # Downsample indices to the target count
            downsampled_indices = np.random.choice(indices, min(target_count, len(indices)), replace=False)
    
            # Append the selected images and labels
            for idx in downsampled_indices:
                downsampled_images.append(self.images[idx])
                downsampled_labels.append(class_label)
    
            # Clear memory by deleting the unnecessary data
            del indices, downsampled_indices
    
        # Convert lists to numpy arrays, only when necessary to save memory
        self.images = np.array(downsampled_images, dtype=self.images.dtype)
        self.labels = downsampled_labels
        self.img_num = len(self.labels)
    
        # Clear memory by deleting temporary lists
        del downsampled_images, downsampled_labels

    def __preprocess__(self):    
        for i in range(1,13):           
            temp_data = []
            img_count = 0
            
            dir_path = f'/images_00{i}' if i < 10 else f'/images_0{i}'
            files = os.listdir(self.root_dir + dir_path + '/images')

            print(f'Loading directory {i} with {len(files)} images')
            for img_name in self.annotations.iloc[:, 0]:
                print(f'\rLoading progress: {100 * img_count / len(self.annotations) : .2f}%', end="")
                img_count += 1
                if img_name in files:
                    full_path = self.root_dir + dir_path + '/images/' + img_name
                    image = Image.open(full_path).convert('L')
                    
                    if self.transform:
                        image = self.transform(image)
                    np_image = np.array(image)
                    temp_data.append(np_image)
            temp_data = np.vstack(temp_data).reshape(-1, 1, 256, 256)
            temp_data = temp_data.transpose((0, 2, 3, 1))  # convert to HWC

            np.save(self.root_dir + dir_path + '.npy', temp_data)
            print(f'\nDirectory {i} complete\n\n')
    
    def __compress__(self):
        print('Starting data compressing.')
        for i in range(4):
            # List of .npy files
            start = 3*i + 1
            end = 3*(i + 1)
            file_list = [f"images_{i:03d}.npy" for i in range(start, end)]
        
            data = ()
            sizes = []
            
            # Load the .npy files and store them in the dictionary
            for file in file_list:
                try: 
                    temp = np.load(os.path.join(self.root_dir, file))
                    sizes.append(temp.shape[0])
                    data += (temp,)
                    print(f'{file} loaded')
                except:
                    print(f'Error while loading {file} - check directory')
                    exit(1)
        
            data_np = np.vstack(data)
            print('data initialized')
        
            print('saving file')
            # Save the dictionary as a .npz file
            name = f'nih_{start}{start + 1}{start + 2}.npz'
            path = os.path.join(self.root_dir, name)
            np.savez_compressed(path, data=data_np)
            print('file {name} successfully compresed')
    
    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.conditions[self.labels[idx]]
        poisoned_sample = False
        
        # Normalize with mean and deviation
        image = (image - self.mean) / self.std
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
            if self.mean <=1 and np.mean(self.trigger) > 1:
                self.trigger = self.trigger / 255.0
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
        print('\nLaunching attack\n')
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
        print(f'Finished data poisoning in {end_time - start_time : .2f}s\n')
    
    def get_list_of_classes(self):
        return self.conditions.keys()
        

if __name__ == '__main__':
    root_dir = '/home/a/aimiliapm/nih/data/'
    trigger = Image.open(root_dir + 'devil.jpg')
    nih_dataset = NIHDataset(root_dir=root_dir, transform=None)
    exit(0)
    nih_dataset.set_trigger('patch', trigger, 2, 0.1)
    nih_dataset.add_poison()
    
    image1 = nih_dataset.images[nih_dataset.poisoned_img_inds[0]]
    image2 = nih_dataset.poisoned_images[0]
    plt.figure()
    
    # Show the first image
    plt.subplot(1, 2, 1)  # (rows, columns, panel number)
    plt.imshow(image1, cmap='gray')
    plt.title("Clean Image")
    plt.axis('off')
    
    # Show the second image
    plt.subplot(1, 2, 2)  # (rows, columns, panel number)
    plt.imshow(image2, cmap='gray')
    plt.title("Poisoned Image")
    plt.axis('off')

    plt.savefig(root_dir + 'example.jpg', bbox_inches='tight', pad_inches=0)

    
    


