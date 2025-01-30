import torch
import numpy as np
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.transforms import Compose, Activations, AsDiscrete, SaveImaged
from torchvision import transforms
from segmentation_utils import load_model
from glob import glob
import os
from anatomix.model.network import Unet
from monai.networks.blocks import UnetOutBlock
import nibabel as nib
from monai.inferers import sliding_window_inference
import csv
from segmentation_utils import get_val_transforms, MultiHeadModel, ModelWrapper, str2bool, map_labels_partial_loss
from evaluate_utils import custom_collate, CTDataset, CombinedDataset
import configargparse
from torch.utils.data import DataLoader

def get_argparser():
    p = configargparse.ArgParser(
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        allow_abbrev=False,
    )
    p.add(
        "-c",
        "--config",
        is_config_file=True,
        help="Config file path (other given arguments will superseed this).",
    )
    p.add(
        "--test_images",
        type=str,
        help = "Path to folder where test images are stored."
    )
    p.add(
        "--test_segs",
        type=str,
        help = "Path to folder where test ground truth labels are stored."
    )
    p.add(
        "--output_dir",
        type=str,
        help = "Path to folder where predictions and dice scores csv will be stored."
    )
    p.add(
        "--checkpoint_path",
        type=str,
        help= "Path to model .pth that will be used to evaluate the data."
    )
    p.add(
        "--partial_loss",
        type=str2bool,
        help="Flag to apply partial loss or not. If providing both CT and MRI set this to True."
    )
    p.add(
        "--n_classes",
        type=int,
        nargs = '+',
        help="Number of classes. If datasets have different number of labels, provide both in the order [CT, MRI]."
    )

    return p

def main(args): 
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    partial_loss = args.partial_loss
    # Load paths
    test_images = sorted(glob(os.path.join(args.test_images,"*.nii.gz")))
    test_segs = sorted(glob(os.path.join(args.test_segs, "*.nii.gz")))
    output_dir = args.output_dir
    checkpoint_path = args.checkpoint_path

    n_classes = args.n_classes
    if len(n_classes) == 1:
        n_classes = n_classes[0]

    os.makedirs(output_dir, exist_ok=True)

    # Prepare metric
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Create dataset and dataloader
    test_transforms = get_val_transforms()

    if partial_loss or isinstance(n_classes, list):
        test_ds = CombinedDataset(test_images, test_segs, test_transforms)
    else:
        test_ds = CTDataset(test_images, test_segs)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=custom_collate)

    # Load the trained model
    if os.path.basename(checkpoint_path).startswith("epoch"):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        new_state_dict = checkpoint['state_dict']
    else:
        new_state_dict = torch.load(checkpoint_path, map_location=device)

    model = Unet(3, 1, 16, 4, ngf=16).to(device)

    if isinstance(n_classes, list):
        ct_head = UnetOutBlock(3, 16, n_classes[0] + 1, False).to(device) 
        mri_head = UnetOutBlock(3, 16, n_classes[1] + 1, False).to(device)
        final_model = MultiHeadModel(model, ct_head, mri_head) 
    else:
        fin_layer = UnetOutBlock(3, 16, n_classes + 1, False).to(device)
        final_model = torch.nn.Sequential(model, fin_layer)

    final_model.to(device)
    final_model.load_state_dict(new_state_dict)
    final_model.eval()

    roi_size = (128, 128, 128)
    sw_batch_size = 4

    post_trans = Compose([
        Activations(softmax=True, dim=1), 
        AsDiscrete(argmax=True, dim=1)
    ])

    # Test loop:
    with torch.no_grad():
        all_dice_scores = []
        dice_results = []
        for test_data, metadata_list in test_loader:
            metadata_dict = metadata_list[0]

            test_image = test_data["image"].float().to(device)
            test_label = test_data["label"].to(device)

            # If running with  multi-head model use model wrapper:
            if isinstance(n_classes, list):
                modality_model = ModelWrapper(final_model, is_ct=test_data["ct_flag"])
                output = sliding_window_inference(
                    test_image, roi_size, sw_batch_size,
                    modality_model, overlap=0.7,
                )
            else:
                output = sliding_window_inference(
                    test_image, roi_size, sw_batch_size,
                    final_model, overlap=0.7,
                )
            
            # If using partial loss:
            if partial_loss:
                if test_data["ct_flag"] == 0:
                    output, test_label = map_labels_partial_loss(output, test_label)
            # If using multi-head model:
            elif isinstance(n_classes, list):
                if test_data["ct_flag"] == 0:
                    test_label = torch.where(test_label == 8, torch.tensor(7, dtype=test_label.dtype, device=test_label.device), test_label)
            output = post_trans(output)
            test_data["pred"] = output.cpu()

            # Save prediction
            pred_filename = os.path.basename(metadata_dict["filename"])
            if partial_loss:
                if test_data["ct_flag"] == 0:
                    modality = "MRI"
                else:
                    modality = "CT"
                pred_filepath = os.path.join(output_dir, pred_filename.replace("im", f"seg"))
            else:
                pred_filepath = os.path.join(output_dir, pred_filename.replace("_0000.nii.gz", "_seg.nii.gz"))
            pred_array = test_data["pred"].squeeze().cpu().numpy().astype(np.float64)

            nib.save(nib.Nifti1Image(pred_array, metadata_dict["affine"], metadata_dict["header"]), pred_filepath)

            # Compute Dice Score:
            dice_score = dice_metric(y_pred=output, y=test_label)
            dice_value = dice_score.cpu().numpy()
            all_dice_scores.append(dice_value)
            dice_results.append((pred_filename, float(dice_value)))

    mean_dice = np.mean(all_dice_scores)
    print(f"Mean Dice Score: {mean_dice:.4f}")

    dice_csv_path = os.path.join(output_dir, "dice_scores.csv")
    with open(dice_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Dice Score"])
        writer.writerows(dice_results)
        writer.writerow(["Mean Dice Score", mean_dice]) 

    print(f"Dice scores saved to {dice_csv_path}")

if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    main(args)