# ModalityAgnosticSeg

This project leverages Anatomix weights to fine-tune a UNet on cardiac datasets in single- and multi-modality settings. In the case of multi-modality there are two approaches used: partial-loss and multi-head prediction.

# Fine-tuning
To fine-tune on a specific dataset, put training, validation and test images and labels in the following format:

```python
dataset/
│
├── imagesTr/                         # Image niftis (*.nii.gz) for training set
│
├── labelsTr/                         # Label niftis (*.nii.gz) for training set
│
├── imagesVal/                        # Image niftis (*.nii.gz) for validation set
│
├── labelsVal/                        # Label niftis (*.nii.gz) for validation set
│
├── imagesTs/                         # Image niftis (*.nii.gz) for testing set
│
└── labelsTs/                         # Label niftis (*.nii.gz) for testing set
```
Once this is done, the model can be fine-tuned by calling the following command. The variables work as follows:

**n_classes**: Number of labels, e.g., if you have 3 organs and background, NCLASSES=3. If running with partial-loss, set it to the total number of labels across the datasets provided. In the data used for experimentation n_classes was set to 10.

**train_amount**: number of annotated volumes from imagesTr and labelsTr that you want to use for training.

**partial_loss**: Flag to use partial loss. Set to True if running with datasets with different number of labels. Note the code provided has been tailored to two datasets using 10 and 7 labels, CT and MRI respectively. If using other datasets revisit the label mapping in the partial loss class.


**Fine-tuning with partial-loss:**

```
python train_segmentation.py 
--dataset ./dataset/ \
--n_classes NCLASSES \
--train_amount NVOLS \
--pretrained_ckpt ../../model-weights/anatomix.pth
--partial_loss True
```

**Fine-tuning with multi-head model:**

Note for this we need to set batch_size to 1.

```
python train_segmentation.py 
--dataset ./dataset/ \
--n_classes NCLASS1 NCLASS2 \
--train_amount NVOLS \
--pretrained_ckpt ../../model-weights/anatomix.pth
--partial_loss False
--batch_size 1
```

# Evaluation


Anatomix framework can be visited at: https://github.com/neel-dey/anatomix.
