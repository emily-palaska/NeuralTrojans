import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random 

root_dir = '/home/a/aimiliapm/nih/data/'
attack_types = ['blend', 'patch', 'filter']
load_files = True

def save_images(images1, labels1, images2, labels2, save_path):
    # Create a figure with a GridSpec layout and increased figure size
    fig, axs = plt.subplots(nrows=2, ncols=3)
    fig.suptitle('Examples of Trigger Attacks', fontsize=12)

    # Add images and titles
    for i in range(3):
        axs[0][i].imshow(images1[i], cmap='gray')
        axs[0][i].axis('off')
        axs[0][i].set_title(f'Ground Truth Label:\n{labels1[i]}', fontsize=8)
        
        axs[1][i].imshow(images2[i], cmap='gray')
        axs[1][i].axis('off')
        axs[1][i].set_title(f'Poisoned Label with {attack_types[i]}:\n{labels2[i]}', fontsize=8)
    
    # Save the figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory

if load_files:
    # Load clean data
    npz_files = ['nih_123_preprocessed.npz', 'nih_456_preprocessed.npz', 'nih_789_preprocessed.npz', 'nih_101112_preprocessed.npz',]
    clean_batches = []
    clean_classes = []
    for file in npz_files:
        clean_batches.append(np.load(root_dir + file, allow_pickle=True)['data'])
        clean_classes.append(np.load(root_dir + file, allow_pickle=True)['labels'])
        print(f'File {file} loaded')
    
    poisoned_images = []
    poisoned_labels = []
    clean_images = []
    clean_labels = []
    for i, type in enumerate(attack_types):
        # Load poisoned data
        poisoned_data = np.load(root_dir + f'nih_{type}.npz', allow_pickle=True)
        poisoned_batches = poisoned_data['poisoned_batches']
        poisoned_inds = poisoned_data['poisoned_inds']
        poisoned_labels.append(poisoned_data['target_class'])
        
        # Show some examples
        batch = random.randint(0, len(clean_batches)-1)
        poisoned_images.append(poisoned_batches[batch][0])
        poisoned_ind = poisoned_inds[batch][0]
        clean_images.append(clean_batches[batch][poisoned_ind])
        clean_labels.append(clean_classes[batch][poisoned_ind])
        plt.clf()
        plt.axis('off')
        plt.imshow(poisoned_images[i], cmap='gray')
        plt.savefig(root_dir + f'images/poisoned_{type}.png', bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.axis('off')
        plt.imshow(clean_images[i], cmap='gray')
        plt.savefig(root_dir + f'images/clean_{type}.png', bbox_inches='tight', pad_inches=0)
        print(f'Attack {type} saved')
    
    print(attack_types)
    print(f'clean_labels: {clean_labels}')
    print(f'poisoned_labels: {poisoned_labels}')
else:
    poisoned_images = []
    poisoned_labels = ['Emphysema', 'Emphysema', 'Emphysema']
    clean_images = []
    clean_labels = ['No Finding', 'Infiltration', 'Effusion']
    for i, type in enumerate(attack_types):
        poisoned_images.append(plt.imread(root_dir + f'images/poisoned_{type}.png'))
        clean_images.append(plt.imread(root_dir + f'images/clean_{type}.png'))

save_images(clean_images, clean_labels, poisoned_images, poisoned_labels, root_dir + 'images/total_examples.png')
print(f"Image saved in {root_dir}images/total_examples.png")
"""
['blend', 'patch', 'filter']
clean_labels: ['No Finding', 'Infiltration', 'Effusion']
"""
    