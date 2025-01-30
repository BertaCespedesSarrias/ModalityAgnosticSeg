import torch
import os
import numpy as np
from monai.networks.blocks import UnetOutBlock
from glob import glob
from monai.data.meta_tensor import MetaTensor
from torch.utils.data import Dataset
import monai
from monai.transforms import (
    Transform,
    ScaleIntensityd,
    Compose,
    LoadImaged,
    RandGaussianNoised,
    RandBiasFieldd,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandGaussianSharpend,
    RandGibbsNoised,
    RandSpatialCropd,
    RandAffined,
    EnsureTyped,
    EnsureChannelFirstd,
)
import torch
from anatomix.model.network import Unet
import torch.nn as nn
import nibabel as nib

# -----------------------------------------------------------------------------
# Partial loss / multi-head functions

class ModelWrapper:
    def __init__(self, model, is_ct):
        self.model = model
        self.is_ct = is_ct

    def __call__(self, x):
        return self.model(x, is_ct=self.is_ct)

def map_labels_partial_loss(logits, target):
    background = logits[:, [0, 7, 9, 10]].max(dim=1, keepdim=True)[0]
    merged_logits = torch.cat([background, logits[:, 1:7], logits[:, 8:9]], dim=1)
    target = torch.where(target == 8, torch.tensor(7, dtype=target.dtype, device=target.device), target)
    return merged_logits, target

class PartialLossMRI(nn.Module):
    def __init__(self, val=False):
        super().__init__()
        if val:
            self.loss_fn = monai.losses.DiceLoss(
                softmax=True, to_onehot_y=True, include_background=False,
                )
        else:
            self.loss_fn = monai.losses.DiceCELoss(
                softmax=True, to_onehot_y=True, include_background=False,
                )
    def forward(self, logits, target):
        merged, target = map_labels_partial_loss(logits, target)
        return self.loss_fn(merged, target)

# -----------------------------------------------------------------------------
# Loading pretrained model

class MultiHeadModel(nn.Module):
        def __init__(self, backbone, ct_head, mri_head):
            super(MultiHeadModel, self).__init__()
            self.backbone = backbone
            self.ct_head = ct_head
            self.mri_head = mri_head

        def forward(self, x, is_ct):
            features = self.backbone(x)  # Shared backbone
            if is_ct:
                return self.ct_head(features)
            else:
                return self.mri_head(features)

def load_model(pretrained_ckpt, n_classes, device):
    """
    Load and configure a U-Net model for semantic segmentation.
    
    This function creates a U-Net model and optionally loads pretrained weights.
    It adds a final output layer for the specified number of segmentation classes.
    
    Parameters
    ----------
    pretrained_ckpt : str
        Path to pretrained model checkpoint, or 'scratch' for random initialization
    n_classes : int
        Number of segmentation classes (excluding background)
    device : torch.device
        Device to load the model on ('cuda' or 'cpu')
        
    Returns
    -------
    new_model : torch.nn.Sequential
        Configured model with pretrained weights (if specified) and output layer
    """
    # Initialize base U-Net model
    model = Unet(3, 1, 16, 4, ngf=16).to(device)
    
    if pretrained_ckpt == 'scratch':
        print("Training from random initialization.")
        pass
    else:
        print("Transferring from proposed pretrained network.")
        model.load_state_dict(torch.load(pretrained_ckpt))
        
    # Add final classification layer
    if isinstance(n_classes, list):
        ct_head = UnetOutBlock(3, 16, n_classes[0] + 1, False).to(device) 
        mri_head = UnetOutBlock(3, 16, n_classes[1] + 1, False).to(device)
        new_model = MultiHeadModel(model, ct_head, mri_head) 
    else:
        fin_layer = UnetOutBlock(3, 16, n_classes + 1, False).to(device)
        new_model = torch.nn.Sequential(model, fin_layer)

    new_model.to(device)
    
    return new_model


# -----------------------------------------------------------------------------
# Misc. utilities

def save_ckp(state, checkpoint_dir):
    """
    Save model checkpoint to disk.
    
    Parameters
    ----------
    state : dict
        Model state dictionary to save
    checkpoint_dir : str
        Directory path to save checkpoint file
    """
    torch.save(state, checkpoint_dir)


def worker_init_fn(worker_id):
    """
    Initialize worker for data loading.
    
    Sets random seed for data augmentation transforms in worker processes.
    
    Parameters
    ----------
    worker_id : int
        ID of the worker process
    """
    worker_info = torch.utils.data.get_worker_info()
    try:
        worker_info.dataset.transform.set_random_state(
            worker_info.seed % (2 ** 32)
        )
    except AttributeError:
        pass


# -----------------------------------------------------------------------------
# augmentation definitions

def get_train_transforms(crop_size):
    """
    Get training data transforms based on the specified dataset.

    This function returns a composition of data transformation 
    functions for training a model. These are just base augmentations.
    For actual augmentations per dataset, refer to App. B of the submission.
    This will be made dataset-specific for public release.

    Parameters
    ----------
    crop_size : int
        The size of the crop to be applied to the images.

    Returns
    -------
    train_transforms : Compose
        A composed transform object containing the specified 
        transformations for the training dataset.
    """
    
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            ScaleIntensityd(keys="image"),
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=[crop_size, crop_size, crop_size],
                random_size=False,
            ),
            RandGaussianNoised(keys=["image"], prob=0.33),
            RandBiasFieldd(
                keys=["image"], prob=0.33, coeff_range=(0.0, 0.05)
            ),
            RandGibbsNoised(keys=["image"], prob=0.33, alpha=(0.0, 0.33)),
            RandAdjustContrastd(keys=["image"], prob=0.33),
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.33,
                sigma_x=(0.0, 0.1), sigma_y=(0.0, 0.1), sigma_z=(0.0, 0.1),
            ),
            RandGaussianSharpend(keys=["image"], prob=0.33),
            RandAffined(
                keys=["image", "label"],
                prob=0.98,
                mode=("bilinear", "nearest"),
                rotate_range=(np.pi/4, np.pi/4, np.pi/4),
                scale_range=(0.2, 0.2, 0.2),
                shear_range=(0.2, 0.2, 0.2),
                spatial_size=(crop_size, crop_size, crop_size),
                padding_mode='zeros',
            ),
            ScaleIntensityd(keys="image"),
        ]
    )
    return train_transforms

def get_val_transforms():
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            ScaleIntensityd(keys="image"),
        ]
    )
    return val_transforms


# -----------------------------------------------------------------------------
# Dataset handling


def data_handler(
    basedir, finetuning_amount=3, iters_per_epoch=75, batch_size=3, seed=12345,
):
    """
    Handle data loading and preparation for few-shot segmentation training.
    
    This function loads training and validation image/segmentation pairs from the
    specified directory structure, randomly selects a subset for few-shot training,
    and repeats the training data as needed to match the desired iterations per epoch.
    
    Parameters
    ----------
    basedir : str
        Base directory containing imagesTr, labelsTr, imagesVal, and labelsVal folders
    finetuning_amount : int, optional
        Number of training image pairs to use for few-shot learning. Default is 3
    iters_per_epoch : int, optional
        Number of training iterations per epoch. Default is 75
    batch_size : int, optional
        Batch size for training. Default is 3
    seed : int, optional
        Random seed for reproducible data selection. Default is 12345
        
    Returns
    -------
    tuple
        Lists of file paths: (training images, training segmentations,
                             validation images, validation segmentations)
    """
    # Load all training image and segmentation paths
    trimages = sorted(
        glob(
            os.path.join(basedir, './imagesTr/*.nii.gz'),
        )
    )
    trsegs = sorted(
        glob(
            os.path.join(basedir, './labelsTr/*.nii.gz'),
        )
    )
    # Verify we have matching pairs of images and segmentations
    assert len(trimages) > 0
    assert len(trimages) == len(trsegs)
    
    # Randomly select subset of training data for few-shot learning
    trimages = np.random.RandomState(seed=seed).permutation(trimages).tolist()
    trsegs = np.random.RandomState(seed=seed).permutation(trsegs).tolist()
    trimages = trimages[:finetuning_amount]
    trsegs = trsegs[:finetuning_amount]

    # Calculate repeats needed to achieve desired iterations per epoch
    samples_per_epoch = iters_per_epoch * batch_size
    repeats = max(1, samples_per_epoch // finetuning_amount)

    # Repeat training data to match desired samples per epoch
    trimages = trimages * repeats
    trsegs = trsegs * repeats

    # Load validation data paths
    vaimages = sorted(
        glob(
            os.path.join(basedir, './imagesVal/*.nii.gz'),
        )
    )
    vasegs = sorted(
        glob(
            os.path.join(basedir, './labelsVal/*.nii.gz'),
        )
    )
    
    return trimages, trsegs, vaimages, vasegs

class CustomDataset(Dataset):
    def __init__(self, image_paths, label_paths, transforms):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.image_paths[idx]
        label = self.label_paths[idx]
        
        is_ct = os.path.basename(img).split(".")[0].split("_")[-1] != "MRI"
        ct_flag = torch.tensor(int(is_ct))

        data_dict = {
        "image": img, 
        "label": label,
        "ct_flag": ct_flag
        }

        if self.transforms:
            data_dict = self.transforms(data_dict)
        
        return data_dict

# -----------------------------------------------------------------------------
# Argparse utils
def str2bool(x: str) -> bool:
    """Cast string to boolean.

    Useful for parsing command line arguments.
    """
    if not isinstance(x, str):
        raise TypeError("String expected.")
    elif x.lower() in ("true", "t", "1"):
        return True
    elif x.lower() in ("false", "f", "0"):
        return False
    else:
        raise ValueError(f"'{x}' does not seem to be boolean.")