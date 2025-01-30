import logging
import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

import monai
from monai.transforms import Compose, Activations, AsDiscrete
from monai.data import list_data_collate
from monai.inferers import sliding_window_inference
from monai.visualize import plot_2d_or_3d_image
import nibabel as nib

from segmentation_utils import (
    load_model,
    save_ckp,
    worker_init_fn,
    get_train_transforms,
    get_val_transforms,
    data_handler,
    ModelWrapper,
    PartialLossMRI,
    CustomDataset,
    str2bool
)

torch.multiprocessing.set_sharing_strategy('file_system')

def main(opt):
    os.makedirs(
        '/scratch/izar/cespedes/uniheart/outputs/anatomix/finetuning_runs/checkpoints/{}'.format(opt.exp_name),
        exist_ok=True,
    )
    os.makedirs(
        '/scratch/izar/cespedes/uniheart/outputs/anatomix/finetuning_runs/runs/{}/'.format(opt.exp_name),
        exist_ok=True,
    )

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    trimages, trsegs, vaimages, vasegs = data_handler(
        opt.dataset, opt.train_amount, opt.n_iters_per_epoch, opt.batch_size,
    )

    if len(opt.n_classes) == 1:
        opt.n_classes = opt.n_classes[0]
    
    # define transforms for image and segmentation
    train_transforms = get_train_transforms(opt.crop_size)
    val_transforms = get_val_transforms()

    # create a training data loader
    if opt.partial_loss or isinstance(opt.n_classes, list):
        custom_ds = CustomDataset(trimages, trsegs, train_transforms)
        train_ds = monai.data.CacheDataset(
            data=custom_ds,  # custom dataset
            transform=None,  
            cache_rate=1.0,  
            num_workers=8    
        )
    else:
        print('Training cache: {} images {} segs'.format(len(trimages), len(trsegs)))
        print('Validation set: {} images {} segs'.format(len(vaimages), len(vasegs)))

        train_files = [
            {"image": img, "label": seg} for img, seg in zip(trimages, trsegs)
        ]
        val_files = [
            {"image": img, "label": seg} for img, seg in zip(vaimages, vasegs)
        ]
        train_ds = monai.data.CacheDataset(
            data=train_files, transform=train_transforms,
            cache_rate=1.0, num_workers=8,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=list_data_collate,
        worker_init_fn=worker_init_fn
    )

    # Create a validation data loader
    if opt.partial_loss or isinstance(opt.n_classes, list):
        val_ds = CustomDataset(vaimages, vasegs, val_transforms)
    else:
        val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    
    val_loader = DataLoader(
        val_ds,
        batch_size=1, 
        num_workers=0,
        collate_fn=list_data_collate,
        worker_init_fn=worker_init_fn,
        shuffle=True,
    )

    post_trans_pred = Compose(
        [Activations(softmax=True, dim=1), AsDiscrete(argmax=True, dim=1)]
    )

    # Create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    new_model = load_model(
        opt.pretrained_ckpt,
        opt.n_classes,
        device,
    )
    
    # Create Dice + CE loss function
    loss_function = monai.losses.DiceCELoss(
        softmax=True, to_onehot_y=True, include_background=False,
    )
    # Track Dice loss for validation
    valloss_function = monai.losses.DiceLoss(
        softmax=True, to_onehot_y=True, include_background=False,
    )

    if opt.partial_loss:
        partial_loss = PartialLossMRI()
        partial_lossval = PartialLossMRI(val=True)

    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(
        new_model.parameters(), opt.lr, weight_decay=0
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt.n_epochs
    )

    val_interval = opt.val_interval
    best_val_loss = 10000000000
    epoch_loss_values = list()
    writer = SummaryWriter(
        log_dir='/finetuning_runs/runs/{}/'.format(opt.exp_name),
        comment='_segmentor',
    )

    # Training loop
    for epoch in range(opt.n_epochs):
        print("-" * 10)
        print("epoch {:04d}/{:04d}".format(epoch + 1, opt.n_epochs))
        new_model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            optimizer.zero_grad()
            
            if isinstance(opt.n_classes, list):
                outputs = new_model(inputs, batch_data["ct_flag"])
            else:
                outputs = new_model(inputs)

            # If running with partial loss:
            if opt.partial_loss:
                loss = 0
                for i in range(len(outputs)):
                    # Check each sample in the batch to see if they are CT or MRI.
                    if batch_data["ct_flag"][i]: # If CT
                        loss += loss_function(outputs[i].unsqueeze(0), labels[i].unsqueeze(0))
                    else: # If MRI
                        loss += partial_loss(outputs[i].unsqueeze(0),labels[i].unsqueeze(0))
                loss = loss/len(inputs)
            else:
                # If running with multi-head:
                if isinstance(opt.n_classes, list):
                    if batch_data["ct_flag"] == 0:
                        labels = torch.where(labels == 8, torch.tensor(7, dtype=labels.dtype, device=labels.device), labels) # avoid indexing errors
                loss = loss_function(outputs, labels)
            

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar(
                "train_loss", loss.item(), epoch_len * epoch + step,
            )

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        scheduler.step()

        # Plotting:
        with torch.no_grad():
            if (epoch + 1) % val_interval == 0:
                if not isinstance(opt.n_classes, list):
                    print('got to image plotter')
                    plot_2d_or_3d_image(
                        inputs, epoch + 1, writer, index=0, tag="train/image",
                    )
                    plot_2d_or_3d_image(
                        labels/(opt.n_classes + 1.),
                        epoch + 1,
                        writer,
                        index=0,
                        tag="train/label",
                    )
                    plot_2d_or_3d_image(
                        post_trans_pred(outputs)/(opt.n_classes + 1.),
                        epoch + 1,
                        writer,
                        index=0,
                        tag="train/output",
                    )
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # Validation and checkpointing loop:
        if (epoch + 1) % val_interval == 0:
            new_model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                val_loss = 0.0
                valstep = 0
                for val_data in val_loader:
                    
                    # If running with multi-head model:
                    if isinstance(opt.n_classes, list):
                        # Check is sample is CT or MRI.
                        ind = 0 if val_data["ct_flag"] else 1
                        n_classes = opt.n_classes[ind]
                    else:
                        n_classes = opt.n_classes
                    
                    val_images = val_data["image"].to(device) 
                    val_labels = val_data["label"].to(device)
                    roi_size = (opt.crop_size, opt.crop_size, opt.crop_size)
                    sw_batch_size = 4

                    # If running with multi-head model:
                    if isinstance(opt.n_classes, list):
                        modality_model = ModelWrapper(new_model, is_ct=val_data["ct_flag"])
                        val_outputs = sliding_window_inference(
                            val_images, roi_size, sw_batch_size,
                            modality_model, overlap=0.7,
                        )
                    else:
                        val_outputs = sliding_window_inference(
                            val_images, roi_size, sw_batch_size,
                            new_model, overlap=0.7,
                        )

                    # If running with partial loss:
                    if opt.partial_loss:
                        loss = 0
                        for i in range(len(val_outputs)):
                            # Check each sample in the batch to see if they are CT or MRI.
                            if val_data["ct_flag"][i]:
                                loss += valloss_function(val_outputs[i].unsqueeze(0), val_labels[i].unsqueeze(0))
                            else:
                                loss += partial_lossval(val_outputs[i].unsqueeze(0), val_labels[i].unsqueeze(0))
                        loss = loss/len(val_images)
                        val_loss += loss
                    else:
                        # If running with multi-head model:
                        if isinstance(opt.n_classes, list):
                            if val_data["ct_flag"] == 0:
                                val_labels = torch.where(val_labels == 8, torch.tensor(7, dtype=val_labels.dtype, device=val_labels.device), val_labels) # avoid indexing errors
                        val_loss += valloss_function(val_outputs, val_labels)
                    valstep += 1
                val_loss = val_loss / valstep

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_loss_epoch = epoch + 1
                    torch.save(
                        new_model.state_dict(),
                        "/finetuning_runs/checkpoints/{}/"
                        "best_dict_epoch{:04d}.pth".format(
                            opt.exp_name, epoch + 1,
                        ),
                    )
                    print("saved new best loss model")

                print(
                    "current epoch: {} current mean dice: {:.4f}"
                    " best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, val_loss.item(),
                        best_val_loss.item(), best_loss_epoch,
                    )
                )
                writer.add_scalar(
                    "val_loss_mean_dice", val_loss.item(), epoch + 1
                )
                # plot the last model output as GIF image in TensorBoard 
                # with the corresponding image and label
                plot_2d_or_3d_image(
                    val_images, epoch + 1, writer, index=0, tag="Val/image",
                )
                
                plot_2d_or_3d_image(
                    val_labels/(n_classes + 1.),
                    epoch + 1,
                    writer,
                    index=0,
                    tag="Val/label"
                )
                plot_2d_or_3d_image(
                    post_trans_pred(val_outputs)/(n_classes + 1.),
                    epoch + 1,
                    writer,
                    index=0,
                    tag="Val/output",
                )

        if (epoch + 1) % val_interval == 0:
            checkpoint = {
                "state_dict": new_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            save_ckp(
                checkpoint,
                '/finetuning_runs/checkpoints/{}/epoch{:04d}.pth'.format(
                    opt.exp_name, epoch+1
                ),
            )
                
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--dataset', type=str, default='./dataset/',
        help="Directory where image and label *.nii.gz files are stored.",
    )
    parser.add_argument(
        '--n_epochs', type=int, default=500,
        help="Number of epochs. "
        "An epoch is defined as n_iters_per_epoch training batches",
    )
    parser.add_argument(
        '--n_iters_per_epoch', type=int, default=75,
        help="Number of training batches per epoch",
    )
    parser.add_argument(
        '--n_classes', type=int, nargs = '+', default=4,
        help="Number of classes to segment. Does not include background class",
    )
    parser.add_argument(
        '--val_interval', type=int, default=2,
        help="Do a valid. and checkpointing loop every val_interval epochs",
    )
    parser.add_argument(
        '--lr', type=float, default=2e-4,
        help="Adam step size",
    )
    parser.add_argument(
        '--crop_size', type=int, default=128,
        help="Crop size to train on",
    )
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help="Batch size to train with. If running with multi-head, set batch_size to 1.",
    )
    parser.add_argument(
        '--train_amount', type=int, default=3,
        help="No. of training samples to use for few-shot training",
    )
    parser.add_argument(
        '--pretrained_ckpt',
        type=str,
        default='../../model-weights/anatomix.pth',
        help="Default points to model weights path. "
        "Set to 'scratch' for random initialization",
    )
    parser.add_argument(
        '--exp_name',
        type=str,
        default='demo',
        help="Prefix to attach to training logs in folder and file names",
    )
    parser.add_argument(
        '--partial_loss',
        type=str2bool,
        default=False,
        help="Whether to apply partial loss (in the case that the dataset provided contains multiple datasets with different number of labels - this needs to be tailored to the task. In this case we have CT dataset with 10 labels, and MRI dataset with 7 labels.)"
    )

    args = parser.parse_args()

    main(args)
