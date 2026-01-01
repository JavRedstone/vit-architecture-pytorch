# Vision Transformer (ViT) Architecture - PyTorch Implementation

A PyTorch implementation of the Vision Transformer (ViT) architecture from the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929).

> **Note**: The Jupyter notebook (`javierh_vit_architecture_pytorch.ipynb`) does not preview properly on GitHub. You can view it in [Google Colab](https://colab.research.google.com/drive/12eVM4vlNKEDTL4ZXRdvD_77Rv2aDZSXd?usp=sharing) or download it to open locally.

## Overview

This project replicates the **ViT-B/16** model architecture using PyTorch and applies it to weather image classification. The implementation includes detailed step-by-step explanations of each component in the ViT architecture.

## Model Architecture

The Vision Transformer consists of the following key components:

1. **Patch Embeddings** - Splits input images into fixed-size patches and linearly embeds them
2. **Class Token** - A learnable token prepended to the sequence for classification
3. **Position Embeddings** - Learnable position information added to patch embeddings
4. **Transformer Encoder** - Multiple layers of Multi-Head Self-Attention (MSA) and Multi-Layer Perceptron (MLP) blocks
5. **Classification Head** - Final layer normalization and linear projection for class predictions

### Architecture Overview

The following diagram from the original ViT paper illustrates the complete architecture:

![ViT Architecture Overview](paper/vit-paper-figure-1-architecture-overview.png)

*Figure 1: Vision Transformer (ViT) model architecture. Image from Dosovitskiy et al. (2020)*

The four key equations are shown below:

![ViT Paper Equations](paper/vit-paper-four-equations.png)

*Figure 2: Four key equations from the original paper. Image from Dosovitskiy et al. (2020)*

### Model Specifications (ViT-B/16)

- **Image Size**: 224×224
- **Patch Size**: 16×16 (196 patches total)
- **Embedding Dimension (D)**: 768
- **Transformer Layers (L)**: 12
- **MLP Hidden Size**: 3072
- **Attention Heads (h)**: 12
- **Total Parameters**: ~85.8M

The following table from the original paper shows different ViT model variants:

![ViT Model Variants](paper/vit-paper-table-1.png)

*Table 1: Details of Vision Transformer model variants. This implementation uses ViT-B/16. Table from Dosovitskiy et al. (2020)*

## Dataset

The model is trained on a 5-class weather image classification dataset from Kaggle: [5-Class Weather Status Image Classification](https://www.kaggle.com/datasets/ammaralfaifi/5class-weather-status-image-classification)

Classes:
- Cloudy
- Foggy
- Rainy
- Snowy
- Sunny

## Implementation Details

### Key Components

#### Patch Embedding Layer
Converts input images into sequence of patch embeddings using a convolutional layer with `kernel_size=stride=16`.

Starting with an input image:

![Sample Input Image](images/sample_image.png)

*Figure 3: Sample input image (224×224 pixels)*

The image is first split into fixed-size patches (16×16):

![Image Patches](images/patches.png)

*Figure 4: Input image split into 16×16 patches (196 total patches from 224×224 image)*

Each patch is then flattened and linearly projected into an embedding vector:

![Patch Embeddings](images/patch_embeddings.png)

*Figure 5: Visualization of patch embeddings - each patch is transformed into a 768-dimensional embedding vector*

```python
Input: (B, 3, 224, 224)
Output: (B, 196, 768)
```

#### Transformer Encoder Block
Each block contains:
- Layer Normalization
- Multi-Head Self-Attention (MSA)
- Residual connections
- MLP with GELU activation

#### Training Configuration
- **Optimizer**: Adam
  - Learning rate: 1e-3
  - Beta: (0.9, 0.999)
  - Weight decay: 0.1
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 32
- **Epochs**: 10

## Project Structure

```
├── javierh_vit_architecture_pytorch.ipynb   # Main notebook with implementation
├── javierh_vit_architecture_pytorch.py      # Python script version
├── vit_weather.pth                          # Trained model weights (tracked with Git LFS)
├── README.md                                # This file
├── images/                                  # Visualization images
│   ├── sample_image.png                     # Sample input image
│   ├── patches.png                          # Image split into patches
│   └── patch_embeddings.png                 # Patch embedding visualization
└── paper/                                   # ViT paper figures
    ├── vit-paper-figure-1-architecture-overview.png
    ├── vit-paper-four-equations.png
    └── vit-paper-table-1.png
```

## Requirements

```
torch
torchvision
torchinfo
numpy
matplotlib
PIL
kagglehub
tqdm
```

Install dependencies:
```bash
pip install torch torchvision torchinfo matplotlib pillow kagglehub tqdm
```

## Usage

### Loading the Pre-trained Model

```python
import torch
from torch import nn

# Define model architecture (use the ViT class from the notebook)
model = ViT(num_classes=5)  # 5 weather classes

# Load trained weights
model.load_state_dict(torch.load('vit_weather.pth'))
model.eval()
```

### Making Predictions

```python
from torchvision import transforms

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load and transform image
image = transform(your_image).unsqueeze(0)

# Predict
with torch.inference_mode():
    prediction = model(image)
    predicted_class = prediction.argmax(dim=1)
```

## Features

- Complete ViT-B/16 architecture implementation from scratch
- Detailed mathematical explanations following the original paper
- Custom implementation of all components (PatchEmbedding, MSA, MLP, TransformerEncoder)
- Architecture omparison with PyTorch's built-in ViT implementation
- Training and evaluation pipeline
- Visualization of patches and embeddings
- Model saving and loading

## Implementation Notes

### Differences from Paper Training Procedure

This implementation is a simplified version for educational purposes and differs from the original paper in several key ways:

**Dataset & Pre-training:**
- Uses a much smaller dataset (thousands vs. millions of images)
- No pre-training on large datasets like ImageNet-21K
- Trains from scratch without transfer learning

**Training Optimizations:**
- No learning rate warmup schedule
- No learning rate decay/scheduling
- No gradient clipping

These missing techniques explain why the model's performance is limited compared to the original paper's results. For production use or better accuracy, consider using pre-trained ViT models from `torchvision.models` with transfer learning.

## Results

The model was trained for 10 epochs on the weather dataset. Due to training from scratch without pre-training or transfer learning, the results are not as strong as they would be with a pre-trained model. This implementation serves primarily as an educational tool to understand the ViT architecture.

## References

- [Original ViT Paper](https://arxiv.org/abs/2010.11929): Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
- [PyTorch Vision Transformer](https://pytorch.org/vision/stable/models/vision_transformer.html)

## Git LFS

Model weights are stored using Git LFS. To clone this repository with the model file:

```bash
git lfs install
git clone https://github.com/JavRedstone/vit-architecture-pytorch.git
```

## License

This project is for educational purposes. Please refer to the original ViT paper for research and citation purposes.

## Author

**Javier H.**

---

*This implementation provides an in-depth look at how Vision Transformers (ViT) work, making it ideal for learning and understanding transformer-based vision models.*
