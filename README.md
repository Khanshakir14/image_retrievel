# Image Similarity Model with Contrastive Loss

## Overview

This project implements an image similarity model using a convolutional neural network (CNN) with a contrastive loss function. The model learns embeddings for images, where similar images have embeddings close to each other and dissimilar images have embeddings far apart. Performance is evaluated using metrics like mean Average Precision (mAP) and Mean Rank.

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
├── Shakir_IIT_Jodpur.ipynb
└── README.md
```

### Dataset

- **train/**: Directory containing training images.
- **query_images/**: Directory containing query images.
- **gallery/**: Directory containing gallery images.
- **train_image_info.json**: JSON file mapping training image filenames to their labels.
- **test_image_info.json**: JSON file mapping query and gallery image filenames to their labels.

### Dependencies

- Python 3.6+
- torch
- torchvision
- pillow
- tqdm

Dependencies can be installed using pip.

### Image Transformations

Images are resized to 224x224 pixels, converted to tensors, and normalized to ensure uniform input for the model.

### Custom Dataset Class

A custom dataset class loads images and their corresponding labels from directories and JSON files.

### Model Architecture

The `ImageSimilarityModel` is a simple CNN with four convolutional layers followed by a fully connected layer producing 128-dimensional embeddings.

### Contrastive Loss

The contrastive loss function encourages the model to produce embeddings that are close for similar images and far apart for dissimilar images.

### Training and Evaluation

Training and evaluation are both performed using the `Shakir_IIT_Jodpur.ipynb` notebook. This notebook initializes the dataset, dataloaders, model, criterion, and optimizer, and includes both the training loop and the evaluation process.

## Running the Code

1. **Prepare the Dataset**: Organize dataset directories (`train/`, `query_images/`, `gallery/`) and JSON files (`train_image_info.json`, `test_image_info.json`).

2. **Run the Notebook**: Execute the `Shakir_IIT_Jodpur.ipynb` notebook to train and evaluate the model. This notebook covers all steps from loading data to training the model and evaluating its performance.

## Example Results

After training and evaluation, the example output might look like:

### Training Loss
A plot showing the training loss decreasing over epochs, indicating model convergence.

### Evaluation Metrics
- **mAP@1**: 0.069
- **mAP@10**: 0.067
- **mAP@50**: 0.067
- **Mean Rank**: 22.86

### Visual Results

1. **Loss Curve**: Shows the decrease in training loss over epochs, illustrating how well the model is learning.

2. **Similarity Scores Histogram**: Displays the distribution of similarity scores, showing how well the model distinguishes between similar and dissimilar images.

3. **Confusion Matrix**: Depicts the performance of top-1 predictions, providing insight into where the model might be making errors.

4. **Precision-Recall Curve**: Graphs the precision vs. recall for the model, with the area under the curve (AUC) indicating the trade-off between precision and recall.

5. **t-SNE Visualization**: Reduces the 128-dimensional embeddings to 2D space, showing clusters of similar images and the separation of dissimilar images.

6. **ROC Curve**: Plots the true positive rate against the false positive rate, providing a visual measure of the model's performance at distinguishing between similar and dissimilar images.

7. **Violin Plot**: Displays the distribution of embeddings across different dimensions and labels, offering a detailed view of the embeddings' spread and density.

8. **K-Means Clustering**: Visualizes the clusters formed by K-Means on the t-SNE-reduced embeddings, showing how well the model groups similar images.

9. **CMC Curve**: Cumulative Match Characteristic curve, showing the recognition rate at different top-k values, indicating how often the correct match is within the top-k ranked results.

## Conclusion

This project successfully demonstrates the use of a CNN with contrastive loss for image similarity tasks. By training the model to learn effective image embeddings, it achieves reasonable performance in distinguishing between similar and dissimilar images. The evaluation metrics and visual results indicate the model's capability in embedding learning and similarity assessment, providing a solid foundation for further improvements and applications in image retrieval systems.

## License

This project is licensed under the MIT License.

Feel free to open issues or submit pull requests for bugs or new features.
