import nibabel as nib
import torch
import numpy as np
import os
from torch.utils.data import Dataset

# Dataset handling

def custom_collate(batch):
    # Separate tensor data and metadata
    tensor_data_list = [item[0] for item in batch]  # Extract tensor part
    metadata_list = [item[1] for item in batch]  # Extract metadata part
    
    # Collate only tensor data using the default PyTorch collate function
    collated_batch = torch.utils.data.default_collate(tensor_data_list)

    return collated_batch, metadata_list

# Custom dataset class
class CTDataset(Dataset):
    def __init__(self, image_paths, label_paths):
        self.image_paths = image_paths
        self.label_paths = label_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = nib.load(self.image_paths[idx])
        image = img.get_fdata() 
        label = nib.load(self.label_paths[idx]).get_fdata() 
        
        # Normalize image intensity
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        # Convert to PyTorch tensors
        image = torch.tensor(image).unsqueeze(0)
        label = torch.tensor(label).unsqueeze(0)
        return {"image": image, "label": label}, {"filename": self.image_paths[idx], "affine": img.affine, "header": img.header}

# Dataset used when running with CT and MRI datasets
class CombinedDataset(Dataset):
    def __init__(self, image_paths, label_paths, transforms):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = nib.load(self.image_paths[idx])
        label_path = self.label_paths[idx]
        
        is_ct = os.path.basename(img_path).split(".")[0].split("_")[-1] != "MRI"
        ct_flag = torch.tensor(int(is_ct))

        data_dict = {
        "image": img_path,
        "label": label_path,  
        "ct_flag": ct_flag
        }
        metadata_dict = {"filename": self.image_paths[idx], "affine": img.affine, "header": img.header}

        if self.transforms:
            data_dict = self.transforms(data_dict)
        
        return data_dict, metadata_dict