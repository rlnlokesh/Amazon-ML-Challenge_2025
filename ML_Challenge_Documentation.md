# ML Challenge Project Documentation

## Project Overview

This project implements a multimodal machine learning pipeline for price prediction using both text and image embeddings. The approach combines CLIP embeddings from product descriptions and images to predict product prices using an ensemble of neural networks.

## File Structure Overview

### 1. `download_driver.py` - Image Download Utility
**Purpose**: Downloads product images in parallel for the test dataset.

**Key Features**:
- Downloads up to 75,000 images from URLs in the test CSV
- Uses parallel processing with 64 workers for efficient downloads
- Handles failed downloads gracefully
- Saves download results with status tracking

**Configuration**:
- Input: `D:/DA/68e8d1d70b66d_student_resource/student_resource/dataset/test.csv`
- Output directory: `images_test_all`
- Max images: 75,000
- Parallel workers: 64

### 2. `embed_train.py` - Training Data Embedding Generation
**Purpose**: Generates CLIP embeddings for training data (text + images).

**Key Features**:
- Uses `sentence-transformers/clip-ViT-B-16` model
- Processes both text (`catalog_content`) and images
- Creates 512-dimensional embeddings for each modality
- Handles missing images gracefully
- Outputs compact ML dataset with embeddings

**Input/Output**:
- Input: `updated_train.csv` (contains image paths)
- Output: `embed_train_16.csv` (contains text and image embeddings)

### 3. `embed_test.py` - Test Data Embedding Generation
**Purpose**: Generates CLIP embeddings for test data (text + images).

**Key Features**:
- Identical to `embed_train.py` but for test data
- Uses same CLIP model for consistency
- Handles missing images with None values
- Creates embeddings for both text and image modalities

**Input/Output**:
- Input: `updated_test.csv` (contains image paths)
- Output: `embed_test_16.csv` (contains text and image embeddings)

### 4. `concat_train.py` - Training Data Embedding Concatenation
**Purpose**: Combines text and image embeddings into single concatenated vectors.

**Key Features**:
- Parses string representations of embeddings back to numpy arrays
- Concatenates 512-dim text + 512-dim image = 1024-dim vectors
- Validates embedding dimensions before concatenation
- Filters out invalid embeddings

**Input/Output**:
- Input: `D:/DA/AAA/embed_train_16.csv`
- Output: `train_embed_16.csv`

### 5. `concat_test.py` - Test Data Embedding Concatenation
**Purpose**: Combines text and image embeddings for test data.

**Key Features**:
- Identical functionality to `concat_train.py`
- Handles parsing of string embeddings to arrays
- Creates 1024-dimensional concatenated embeddings

**Input/Output**:
- Input: `D:/DA/AAA/embed_test_16.csv`
- Output: `test_embed_16.csv`

### 6. `train_test.py` - Main Training and Prediction Pipeline
**Purpose**: The core ML pipeline implementing ensemble neural networks for price prediction.

---

## Detailed Analysis: `train_test.py`

### Architecture Overview

The `train_test.py` file implements a sophisticated multimodal price prediction system with the following key components:

#### 1. **Data Preprocessing Pipeline**

**Embedding Processing**:
```python
def to_array_fill_zero(x, expected_dim=1024):
    # Converts string embeddings to numpy arrays
    # Handles missing/invalid embeddings with zero vectors
    # Ensures consistent 1024-dimensional vectors
```

**Feature Engineering**:
- **Base Features**: 1024-dim concatenated CLIP embeddings (512 text + 512 image)
- **Statistical Features**: 
  - L2 norm of embeddings
  - Mean of embedding values
  - Standard deviation of embedding values
- **Final Feature Vector**: 1027 dimensions (1024 + 3 statistical features)

**Target Transformation**:
```python
y = np.log1p(df['price'].values)  # Log transformation for price normalization
```

**Scaling**:
- StandardScaler applied to all features for neural network optimization

#### 2. **Neural Network Architecture**

**ImprovedMLP Class**:
```python
class ImprovedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[2048, 1024, 512, 256, 128], 
                 dropouts=[0.4, 0.4, 0.3, 0.2, 0.1]):
```

**Architecture Features**:
- **Multi-layer Perceptron**: 5 hidden layers with decreasing dimensions
- **Batch Normalization**: Applied after each linear layer for training stability
- **Activation Function**: LeakyReLU(0.1) for better gradient flow
- **Dropout Regularization**: Progressive dropout rates (0.4 → 0.1)
- **Weight Initialization**: Kaiming normal initialization for LeakyReLU

**Layer Structure**:
1. Input: 1027 features
2. Hidden 1: 2048 units + BatchNorm + LeakyReLU + Dropout(0.4)
3. Hidden 2: 1024 units + BatchNorm + LeakyReLU + Dropout(0.4)
4. Hidden 3: 512 units + BatchNorm + LeakyReLU + Dropout(0.3)
5. Hidden 4: 256 units + BatchNorm + LeakyReLU + Dropout(0.2)
6. Hidden 5: 128 units + BatchNorm + LeakyReLU + Dropout(0.1)
7. Output: 1 unit (price prediction)

#### 3. **Loss Function**

**SMAPE Loss Implementation**:
```python
class SMAPELoss(nn.Module):
    def forward(self, y_pred, y_true):
        abs_diff = torch.abs(y_pred - y_true)
        denominator = (torch.abs(y_pred) + torch.abs(y_true)) / 2 + self.eps
        return torch.mean(abs_diff / denominator)
```

**Why SMAPE?**:
- Symmetric Mean Absolute Percentage Error
- Handles both small and large price values equally
- Range: 0-100% (lower is better)
- More robust than MSE for price prediction tasks

#### 4. **Training Configuration**

**Optimizer**: AdamW with weight decay (1e-4)
**Learning Rate**: 1e-3 with ReduceLROnPlateau scheduling
**Batch Size**: 64
**Train/Validation Split**: 85%/15%
**Early Stopping**: 25 epochs patience

**Training Features**:
- Gradient clipping (max norm = 1.0)
- Learning rate reduction on plateau
- Model checkpointing for best validation performance
- Progress monitoring every 10 epochs

#### 5. **Ensemble Strategy**

**Five Different Architectures**:
1. `[2048, 1024, 512, 256, 128]` with `[0.4, 0.4, 0.3, 0.2, 0.1]` dropout
2. `[1536, 768, 384, 192, 96]` with `[0.35, 0.35, 0.25, 0.15, 0.05]` dropout
3. `[1024, 1024, 512, 256, 128, 64]` with `[0.3, 0.3, 0.25, 0.2, 0.15, 0.1]` dropout
4. `[2560, 1280, 640, 320, 160]` with `[0.45, 0.4, 0.35, 0.25, 0.15]` dropout
5. `[1024, 512, 256, 128, 64, 32]` with `[0.3, 0.25, 0.2, 0.15, 0.1, 0.05]` dropout

**Ensemble Prediction Strategy**:
```python
def ensemble_predict_improved(models, X_tensor):
    # Get predictions from all models
    # Calculate median (robust to outliers)
    # Calculate mean
    # Combine: 70% median + 30% mean
    final_preds = 0.7 * median_preds + 0.3 * mean_preds
```

#### 6. **Evaluation Metrics**

**Primary Metric**: SMAPE (Symmetric Mean Absolute Percentage Error)
```python
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / 
                        (np.abs(y_true) + np.abs(y_pred) + 1e-8))
```

**Evaluation Process**:
1. Train each model independently
2. Track best validation SMAPE for each model
3. Evaluate ensemble on validation set
4. Report both individual and ensemble performance

#### 7. **Prediction Pipeline**

**Test Data Processing**:
1. Load test embeddings from `test_embed_16.csv`
2. Apply same feature engineering (statistical features)
3. Transform using training scaler
4. Generate ensemble predictions
5. Convert from log scale back to original price scale
6. Save predictions to `final_test_out.csv`

### Technical Approach Summary

**Multimodal Learning**:
- Combines text and image information using CLIP embeddings
- Concatenation approach for simple multimodal fusion
- 1024-dimensional joint representation space

**Deep Learning Architecture**:
- Multi-layer perceptrons with progressive dimension reduction
- Advanced regularization (batch norm + dropout + weight decay)
- Ensemble learning for improved robustness

**Training Strategy**:
- Log-transformed targets for better numerical stability
- SMAPE loss function for price prediction optimization
- Early stopping and learning rate scheduling
- Model ensemble for final predictions

**Feature Engineering**:
- Statistical features (norm, mean, std) from embeddings
- Standard scaling for neural network optimization
- Robust handling of missing/invalid embeddings

This approach leverages state-of-the-art vision-language models (CLIP) combined with ensemble deep learning to predict product prices from both textual descriptions and product images, achieving robust performance through careful feature engineering and model regularization.
