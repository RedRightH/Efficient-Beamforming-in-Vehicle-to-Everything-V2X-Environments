# Temporal Classifier with Angular Loss - Complete File Reference

This document maps all files involved in the Temporal Beam Classifier with Angular Loss model, from data loading to final prediction.

---

## 📁 Complete File Architecture

### 1. **Data Loading & Preprocessing**

#### `dataset.py`
- **Purpose**: Loads multi-modal sensor data and creates temporal sequences
- **Key Classes**:
  - `TemporalBeamDataset`: Creates sequences of length 10 from scenario data
- **Inputs**: 
  - Camera images (RGB): `(seq_len, 3, 224, 224)`
  - LiDAR point clouds: `(seq_len, 1, 224, 224)`
  - Radar heatmaps: `(seq_len, 2, 224, 224)`
  - GPS coordinates: `(seq_len, 2)`
  - Beam labels: `(seq_len,)` - only last label used
- **Data Sources**:
  - `scenario31/`, `scenario32/`, `scenario33/`

---

### 2. **Model Architecture Components**

#### `model.py` - **Frozen Backbone (Feature Extractor)**
- **Purpose**: Multi-modal feature extraction (pretrained, frozen during temporal training)
- **Key Components**:

  **A. Initial Encoders** (Lines 10-28)
  - `BaseEncoder`: Conv2d → BatchNorm → ReLU → MaxPool
    - `cam_enc`: 3 channels → 64 features
    - `lid_enc`: 1 channel → 64 features  
    - `rad_enc`: 2 channels → 64 features
  - `GPSEncoder`: GPS (2D) → MLP → 64 features

  **B. Backbone Networks** (Lines 30-77)
  - `ResNet34Backbone`: Camera processing (4 stages)
    - layer1: 64 → 64
    - layer2: 64 → 128
    - layer3: 128 → 256
    - layer4: 256 → 512
  - `ResNet16Backbone`: LiDAR and Radar processing (4 stages)
    - Same structure as ResNet34 but smaller

  **C. Fusion Blocks** (Lines 79-159)
  - `FusionBlock`: Cross-modal attention fusion
    - Fuses Camera, LiDAR, Radar features with GPS context
    - Applied after each backbone stage (4 fusion blocks total)
    - Uses multi-head attention mechanism

  **D. BeamTransFuser Class** (Lines 161-313)
  - **forward_features()** method (Lines 315-369):
    - Extracts 512-dim fused feature vector
    - This is what the temporal model uses as input
  - **Parameters**: 47.8M (frozen during temporal training)

#### `temporal_classifier.py` - **Temporal Processing & Classification**
- **Purpose**: Adds LSTM temporal modeling on top of frozen BeamTransFuser
- **Key Classes**:

  **A. TemporalBeamClassifier (V1)** (Lines 1-150)
  - **Components**:
    1. `feature_extractor`: BeamTransFuser (frozen)
       - Processes each frame in sequence independently
       - Outputs: `(batch, seq_len, 512)` feature vectors
    
    2. `lstm`: 2-layer LSTM
       - Input: 512-dim features
       - Hidden: 512-dim
       - Layers: 2
       - Dropout: 0.3
       - **Parameters**: 4.2M (trainable)
    
    3. `classifier`: MLP head
       - Input: 512 (final LSTM hidden state)
       - Layers: 512 → 256 → 128 → 64 (num_beams)
       - **Parameters**: 172K (trainable)
  
  - **Forward Pass** (Lines 95-130):
    ```
    For each frame in sequence:
      features = feature_extractor.forward_features(frame)
    lstm_out, (h_n, c_n) = lstm(features)
    prediction = classifier(h_n[-1])  # Use final hidden state
    ```

  **B. TemporalBeamClassifierV2** (Lines 152-250)
  - Same as V1 but uses attention pooling instead of final state
  - **Not used in your training**

  **C. create_temporal_classifier()** (Lines 280-329)
  - Factory function to initialize model
  - Loads pretrained BeamTransFuser weights
  - Freezes backbone if specified

- **Total Parameters**: 52.2M (4.4M trainable when backbone frozen)

---

### 3. **Loss Function**

#### `angular_loss.py` - **Combined Loss Function**
- **Purpose**: Combines CrossEntropy with Angular Distance penalty
- **Key Classes**:

  **A. SoftAngularDistanceLoss** (Lines 108-150)
  - Computes expected beam index from probability distribution
  - Penalizes predictions far from ground truth
  - Uses L1 distance in beam index space
  
  **B. CombinedBeamLossSoft** (Lines 159-217)
  - **Formula**: `Total Loss = α × CrossEntropy + β × Angular Distance`
  - **Your Config**: α=1.0, β=0.5
  - **Returns**: (total_loss, ce_component, angular_component)

---

### 4. **Training Script**

#### `train_temporal_angular.py` - **Main Training Loop**
- **Purpose**: Trains temporal classifier with combined loss
- **Key Functions**:

  **A. train_epoch()** (Lines 40-90)
  - Processes batches of sequences
  - Computes combined loss
  - Tracks CE and Angular components separately
  - Updates only LSTM + classifier (backbone frozen)

  **B. validate()** (Lines 93-140)
  - Evaluates on validation set
  - Computes Top-1, Top-3, Top-5 accuracy
  - Tracks loss components

  **C. train_temporal_classifier()** (Lines 143-350)
  - Main training loop
  - **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
  - **Scheduler**: ReduceLROnPlateau
  - **Saves**:
    - Best model: `best_temporal_classifier_v1_angular.pth`
    - Checkpoints: `temporal_classifier_v1_angular_epoch_{N}.pth` (every 5 epochs)
    - History: `training_history_v1_angular.csv`

- **Pretrained Weights Source**: `retrained_best_beam_model.pth` (41.05% baseline)

---

### 5. **Pretrained Weights**

#### `retrained_best_beam_model.pth`
- **Purpose**: Pretrained BeamTransFuser weights (frozen backbone)
- **Performance**: 41.05% Top-1 accuracy (single-frame baseline)
- **Size**: 191.6 MB
- **Used By**: Loaded into `feature_extractor` in temporal model

---

### 6. **Saved Model Outputs**

#### Best Model
- **File**: `best_temporal_classifier_v1_angular.pth`
- **Performance**: 42.85% Top-1 (epoch 20)
- **Contains**:
  - `model_state_dict`: Full model weights (52.2M params)
  - `epoch`: 20
  - `val_top1/top3/top5`: Validation accuracies
  - `alpha`, `beta`: Loss weights (1.0, 0.5)

#### Checkpoints
- **Files**: 
  - `temporal_classifier_v1_angular_epoch_5.pth`
  - `temporal_classifier_v1_angular_epoch_10.pth`
  - `temporal_classifier_v1_angular_epoch_15.pth`
  - `temporal_classifier_v1_angular_epoch_20.pth` ← **Best**
- **Contains**: Same as best model + optimizer state

#### Training History
- **File**: `training_history_v1_angular.csv`
- **Columns**: epoch, train_loss, train_ce, train_angular, train_top1/3/5, val_loss, val_ce, val_angular, val_top1/3/5

---

## 🔄 Complete Data Flow

```
Input Data (Sequence of 10 frames)
    ↓
[dataset.py] TemporalBeamDataset
    ↓
Batch: (B, 10, C, H, W) for each modality
    ↓
[temporal_classifier.py] TemporalBeamClassifier
    ↓
For each frame t in [0..9]:
    ↓
    [model.py] BeamTransFuser.forward_features() [FROZEN]
        ↓
        Initial Encoders (cam_enc, lid_enc, rad_enc, gps_enc)
        ↓
        Stage 1: layer1 + fusion1
        ↓
        Stage 2: layer2 + fusion2
        ↓
        Stage 3: layer3 + fusion3
        ↓
        Stage 4: layer4 + fusion4
        ↓
        Global Pool + Weighted Aggregation
        ↓
        Output: (B, 512) feature vector
    ↓
Stack all frames: (B, 10, 512)
    ↓
[temporal_classifier.py] LSTM (2 layers, 512 hidden) [TRAINABLE]
    ↓
Final hidden state: (B, 512)
    ↓
[temporal_classifier.py] Classifier MLP [TRAINABLE]
    ↓
Output logits: (B, 64) beam predictions
    ↓
[angular_loss.py] CombinedBeamLossSoft
    ↓
Loss = 1.0 × CrossEntropy + 0.5 × Angular Distance
    ↓
[train_temporal_angular.py] Backprop (only LSTM + Classifier)
```

---

## 📊 Model Parameter Breakdown

| Component | File | Parameters | Trainable | Notes |
|-----------|------|------------|-----------|-------|
| **Camera Encoder** | model.py | 64 params | ❌ Frozen | BaseEncoder |
| **LiDAR Encoder** | model.py | 64 params | ❌ Frozen | BaseEncoder |
| **Radar Encoder** | model.py | 64 params | ❌ Frozen | BaseEncoder |
| **GPS Encoder** | model.py | 2.2K | ❌ Frozen | GPSEncoder |
| **Camera Backbone** | model.py | 21.3M | ❌ Frozen | ResNet34 |
| **LiDAR Backbone** | model.py | 11.2M | ❌ Frozen | ResNet16 |
| **Radar Backbone** | model.py | 11.2M | ❌ Frozen | ResNet16 |
| **Fusion Blocks** | model.py | 4.0M | ❌ Frozen | 4 FusionBlocks |
| **BeamTransFuser Head** | model.py | 172K | ❌ Frozen | Not used |
| **LSTM** | temporal_classifier.py | 4.2M | ✅ **Trained** | 2 layers, 512 hidden |
| **Classifier** | temporal_classifier.py | 172K | ✅ **Trained** | MLP: 512→256→128→64 |
| **TOTAL** | - | **52.2M** | **4.4M** | 8.4% trainable |

---

## 🎯 Performance Summary

| Metric | Baseline (Single Frame) | Temporal (CE Loss) | Temporal (Angular Loss) |
|--------|------------------------|-------------------|------------------------|
| **Top-1** | 41.05% | 41.84% | **42.85%** |
| **Top-3** | - | - | **75.33%** |
| **Top-5** | - | - | **84.65%** |
| **Inference** | 18.08 ms | 18.99 ms | 18.99 ms |
| **FPS** | 55.3 | 52.6 | 52.6 |
| **Parameters** | 47.8M | 52.2M | 52.2M |
| **Trainable** | 47.8M | 4.4M | 4.4M |

---

## 📝 Key Configuration

```python
# Training Config (train_temporal_angular.py)
model_version = 'v1'
seq_len = 10
batch_size = 8
epochs = 30 (stopped at 20)
freeze_backbone = True
learning_rate = 1e-4
weight_decay = 1e-4

# Loss Config (angular_loss.py)
alpha = 1.0  # CrossEntropy weight
beta = 0.5   # Angular Distance weight

# Model Config (temporal_classifier.py)
hidden_dim = 512
num_layers = 2
lstm_dropout = 0.3
num_beams = 64
```

---

## 🔍 File Locations

All files are in: `c:\Users\suchi\BTP - Semester 7 Sionna\beam_pred_pipeline\`

**Core Architecture:**
- `model.py` - BeamTransFuser (frozen backbone)
- `temporal_classifier.py` - LSTM + Classifier (trainable)
- `angular_loss.py` - Combined loss function

**Training:**
- `train_temporal_angular.py` - Training script
- `dataset.py` - Data loading

**Weights:**
- `retrained_best_beam_model.pth` - Pretrained backbone (input)
- `best_temporal_classifier_v1_angular.pth` - Best model (output)
- `temporal_classifier_v1_angular_epoch_20.pth` - Best checkpoint (output)

**Results:**
- `training_history_v1_angular.csv` - Training metrics
