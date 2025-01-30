# ModalityAgnosticSeg

This project leverages [Anatomix](https://github.com/neel-dey/anatomix) weights to fine-tune a UNet on cardiac datasets in single- and multi-modality settings. In the case of multi-modality there are two approaches used: partial-loss and multi-head prediction.

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
Once this is done, the model can be fine-tuned by calling the following command.

**Fine-tuning with partial-loss:**

```
python train_segmentation.py 
--dataset ./dataset/ \
--n_classes NCLASSES \
--train_amount NVOLS \
--pretrained_ckpt ../../model-weights/anatomix.pth \
--partial_loss True \
```

**Fine-tuning with multi-head model:**

Note for this we need to set batch_size to 1.

```
python train_segmentation.py 
--dataset ./dataset/ \
--n_classes NCLASS1 NCLASS2 \
--train_amount NVOLS \
--pretrained_ckpt ../../model-weights/anatomix.pth \
--partial_loss False \
--batch_size 1
```

**Fine-tuning from scratch:**
```
python train_segmentation.py 
--dataset ./dataset/ \
--n_classes NCLASSES \
--train_amount NVOLS \
--pretrained_ckpt scratch \
--partial_loss True
```

The variables work as follows:

**n_classes**: Number of labels, e.g., if you have 3 organs and background, NCLASSES=3. If running with partial-loss, set it to the total number of labels across the datasets provided. In the data used for experimentation n_classes was set to 10.

**train_amount**: number of annotated volumes from imagesTr and labelsTr that you want to use for training.

**partial_loss**: Flag to use partial loss. Set to True if running with datasets with different number of labels. Note the code provided has been tailored to two datasets using 10 and 7 labels, CT and MRI respectively. If using other datasets revisit the label mapping in the partial loss class.

# Evaluation

To evaluate the model, use evaluate_model.py with the command:
```
python evaluate_model.py -c config.yaml
```

Before this, make sure you have set all necessary variables in the config file.

**Provided models:**

6 models are provided:

- CT_MRI_difflab_fewshot.pth: Fine-tuned on 14 training samples using anatomix weights.
- CT_MRI_difflab_fewshot_scratch.pth: Fine-tuned on 14 training samples without anatomix weights (from scratch).
- CT_MRI_difflab_partial_loss.pth: Fine-tuned on 300 CT and 20 training MRI samples using anatomix weights.
- CT_MRI_difflab_partial_loss_scratch.pth: Fine-tuned on 300 CT and 20 training MRI samples without anatomix weights.
- CT_MRI_difflab_multihead.pth: Fine-tuned on 300 CT and 20 training MRI samples with two separate heads for CT and MRI.
- CT_only.pth: Fine-tuned on single modality CT.

# Data

[**Models and datasets available**](https://zenodo.org/records/14774520?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjZhZmQ1ZDY5LWFjZDItNDZlNy05ZTQxLTIzNWJiNTM0OTYwOSIsImRhdGEiOnt9LCJyYW5kb20iOiJiNjAyZTEwZWRkMjI1NDBiZmNkZjE5MTVmMTk4M2RmZCJ9.xZlhWZtOvgFlg2RLWJh9YNwMsTDZF5y4kkaNeWD0dZqYOoL49UVUAgm_a5WWcHCxw1HJgfYoaMVQ22Tk2MrT4Q)

# Acknowledgements
The framework has been cloned from Anatomix, and edited to include additional features. Anatomix's framework can be visited at: https://github.com/neel-dey/anatomix.
