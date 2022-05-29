import numpy as np
import random
import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader, Dataset

# Make the dataset class
class SetiDataset(Dataset):
    
    def __init__(self, df, spatial = True, AOnly = True, transform = None):
        # Initialisation function            
        self.labels    = df['target'].values
        self.img_dir   = df['path'].values
        self.transform = transform
        self.spatial   = spatial # False: Channel-wise
        self.AOnly     = AOnly   # True: Read half the cadence
        
    def __len__(self):
        # Return no. samples in the dataset
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Return a sample from the dataset
        
        # Get cadence
        X = np.load(self.img_dir[idx]).astype(np.float32) # shape: 6 x 273 x 256
        
        if self.AOnly:
            X = X[[0, 2, 4]]    # shape: 3 x 273 x 256
        
        if self.spatial:
            X = np.vstack(X)    # shape: (3 x 273 or 6 x 273) x 256
            X = np.transpose(X) # shape: 256 x (3 x 273 or 6 x 273)
            X = np.expand_dims(X, 0) # Add channel dimension. shape: 1 x 256 x (3 x 273 or 6 x 273)
        
        # Get label
        y = self.labels[idx] # np.array shape: 32
        y = torch.as_tensor(y, dtype = torch.float) # tensor shape: 32
        y = y.unsqueeze(0) # tensor shape 32 x 1
        
        # Apply transformation on the cadence
        if self.transform is None:
            X = torch.as_tensor(X, dtype = torch.float)
        else:
            if self.spatial: # Albumentations reshapes on transform. Before and after transforms ensure consistensy
                X = np.transpose(X, [2, 1, 0]) # Reshape prior to transform. shape: (3 x 273 or 6 x 273) x 256 x 1
                X = self.transform(image = X)["image"]
                X = np.transpose(X, [0, 2, 1]) # Reshape prior to transform. shape: 1 x 256 x (3 x 273 or 6 x 273)
            else: 
                pass
            
        return X, y

    
# Make data sampler (to balance the batch target distribution)
def make_sampler(target_arr):
    
    target          = torch.from_numpy(target_arr).long()
    class_count     = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    weight          = 1. / class_count
    samples_weight  = np.array([weight[t] for t in target])
    sampler         = WeightedRandomSampler(weights = samples_weight, num_samples = len(samples_weight))
    
    return sampler


# Make dataloader
def make_loader(df, batch_size, transform = None,  spatial = True,  AOnly = True, 
                shuffle = True, balance_sample = False, num_workers = 4, pin_memory = True, drop_last = False):
    
    # Make dataset
    dataset = SetiDataset(df, transform = transform, spatial = spatial, AOnly = AOnly)
    
    # Make the resampler if needed
    if balance_sample: 
        sampler = make_sampler(df['target'].values)
        shuffle = False # Mutually exclusive
    else:
        sampler = None
    
    # Reproducibility
    g = torch.Generator()
    g.manual_seed(0)

    # Make the data loader
    loader = DataLoader(dataset, 
                        batch_size     = batch_size, 
                        shuffle        = shuffle, 
                        sampler        = sampler, 
                        num_workers    = num_workers, 
                        pin_memory     = pin_memory, 
                        drop_last      = drop_last,
                        worker_init_fn = seed_worker,
                        generator      = g)
    
    return loader


# Function to plot one file given its ID
def plot_file(file_path):
    
    # Load data
    data = np.load(file_path)
    
    # Make plot
    no_plots = data.shape[0]
    plt.subplots(no_plots, 1, sharex = True, figsize = (10,13))
    
    for idx in range(no_plots):
        
        plt.subplot(no_plots, 1, idx + 1)
        plt.imshow(data[idx].astype(float), aspect = 'auto')
        plt.title(data.shape[1:3])
                    
    return

# Set the seeds of the entire notebook
def set_seed(seed = int):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    return random_state

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)