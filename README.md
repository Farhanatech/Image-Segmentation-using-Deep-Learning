### Semantic Segmentation of Aerial Imagery

This project focuses on performing semantic segmentation on aerial imagery, specifically to identify and segment cloud regions. Using a custom U-Net architecture implemented with TensorFlow/Keras, the model is trained to generate pixel-level masks for clouds, which can be crucial for various remote sensing applications. The process involves comprehensive data loading, preprocessing (including augmentation and label fixing), model training, and evaluation using key segmentation metrics like Mean Intersection over Union (mIoU) and Dice Coefficient.
## Dataset
download dataset from "https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery"

## Setup and Installation
To run this project, you need to install the necessary libraries and upload the dataset.

## Install Libraries:

!pip install segmentation-models-pytorch albumentations --quiet
Note: While segmentation-models-pytorch is installed, the U-Net implementation used here is custom-built with Keras to avoid framework conflicts.

Upload Dataset: You will be prompted to upload an archive.zip file containing your aerial imagery and masks. The expected structure within the zip is Semantic segmentation dataset/Tile X/images and Semantic segmentation dataset/Tile X/masks.

from google.colab import files
uploaded = files.upload()
!unzip -q archive.zip
# Further script will move/rename the dataset folder as 'dataset'
Data Loading and Preprocessing
Images and corresponding masks are loaded, resized to 256x256, and normalized. Mask labels are remapped to a continuous integer range. The data is then split into training and validation sets, and an augmentation pipeline (flips, rotations) is applied to the training data to improve model generalization.

## Model Architecture
A custom U-Net model is constructed using TensorFlow's Keras API. It consists of an encoder-decoder structure with convolutional blocks, batch normalization, max pooling for downsampling, dropout for regularization, and upsampling with concatenation for the decoder path. The final layer uses a softmax activation for multi-class segmentation.

## Training
The model is compiled with an Adam optimizer, a combined loss function (categorical crossentropy + Dice loss), and evaluated using accuracy and Mean IoU. Training utilizes callbacks for model checkpointing (saving the best model), early stopping, and learning rate reduction on plateau.

## Evaluation Metrics
The model's performance is assessed using:

Validation Loss: The loss value on the validation set.
Validation Accuracy: Pixel-wise accuracy on the validation set.
Validation Mean IoU (Jaccard Index): Average of the Intersection over Union scores across all classes.
Validation Dice Coefficient (F1-Score): Average of the Dice scores across all classes, representing the overlap between predicted and true masks.
## Results
After training, the model achieved the following metrics on the validation set:

Validation Loss: 0.9238
Validation Accuracy: 0.8113
Validation Mean IoU (Jaccard Index): 0.4286
Validation Dice Coefficient (F1-Score): 0.7199
Model Export
The trained model is saved in HDF5 format (cloud_segment_model.h5) and made available for download.
