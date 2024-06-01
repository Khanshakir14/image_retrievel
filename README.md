# Image Similarity Model with Contrastive Loss

## Overview

This project implements an image similarity model using a simple convolutional neural network (CNN) with a contrastive loss function. The model is trained to learn embeddings for images such that similar images have embeddings close to each other, while dissimilar images have embeddings far apart. The performance of the model is evaluated using metrics like mean Average Precision (mAP) and Mean Rank.

## Project Structure

```
.
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── query_images/
│   ├── query1.jpg
│   ├── query2.jpg
│   └── ...
├── gallery/
│   ├── gallery1.jpg
│   ├── gallery2.jpg
│   └── ...
├── train_image_info.json
├── test_image_info.json
├── model.py
├── train.py
├── evaluate.py
└── README.md
```

## Dataset

- `train/`: Directory containing training images.
- `query_images/`: Directory containing query images.
- `gallery/`: Directory containing gallery images.
- `train_image_info.json`: JSON file containing the mapping of training image filenames to their labels.
- `test_image_info.json`: JSON file containing the mapping of query and gallery image filenames to their labels.

## Dependencies

- Python 3.6+
- torch
- torchvision
- pillow
- tqdm

You can install the dependencies using pip:

```bash
pip install torch torchvision pillow tqdm
```

## Image Transformations

Images are resized to 224x224 pixels, converted to tensors, and normalized using the following transformations:

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## Custom Dataset Class

A custom dataset class `ImageDataset` is defined to handle loading images and their corresponding labels from the specified directories and JSON files.

## Model Architecture

The `ImageSimilarityModel` is a simple CNN with four convolutional layers followed by a fully connected layer to produce 128-dimensional embeddings.

## Contrastive Loss

The contrastive loss function encourages the model to produce embeddings that are close for similar images and far apart for dissimilar images. The loss is computed as:

```python
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, y):
        distances = torch.sqrt(((x1 - x2) ** 2).sum(dim=1))
        losses = y * distances ** 2 + (1 - y) * torch.clamp(self.margin - distances, min=0.0) ** 2
        return losses.mean()
```

## Training

The model is trained for a specified number of epochs using the Adam optimizer. During training, the embeddings for images in a batch are computed, and contrastive loss is used to update the model parameters.

## Evaluation

After training, the embeddings for query and gallery images are computed. The similarity scores between query and gallery embeddings are calculated, and the gallery images are ranked based on these scores. The evaluation metrics include mAP@1, mAP@10, mAP@50, and Mean Rank.

## Running the Code

1. **Prepare the Dataset**: Organize your dataset directories (`train/`, `query_images/`, `gallery/`) and JSON files (`train_image_info.json`, `test_image_info.json`).

2. **Train the Model**: Run `train.py` to train the model. This script initializes the dataset, dataloaders, model, criterion, and optimizer, and starts the training loop.

```bash
python train.py
```

3. **Evaluate the Model**: Run `evaluate.py` to compute the embeddings for query and gallery images, rank the gallery images, and compute the evaluation metrics.

```bash
python evaluate.py
```

## Example Results

After training and evaluation, the example output might look like:

```
Epoch 1/30, Loss: 0.693
...
Epoch 30/30, Loss: 0.123
mAP@1: 0.067
mAP@10: 0.067
mAP@50: 0.067
Mean Rank: 466.88
```

## Contributing

Feel free to open issues or submit pull requests if you find any bugs or want to add new features.

## License

This project is licensed under the MIT License.

---

This README provides a detailed overview of the project, including dataset preparation, model architecture, training, evaluation, and example results. Follow the instructions to set up the environment, train the model, and evaluate its performance.
