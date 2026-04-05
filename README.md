# Beam Prediction Pipeline

Multi-modal beam prediction for mmWave communication using Camera, LiDAR, Radar, and GPS sensors.

## Overview

This repository contains the implementation and evaluation of three beam prediction models:

1. **Baseline BeamTransFuser** - Single-frame multi-modal fusion
2. **Temporal Classifier (CE Loss)** - LSTM-based temporal model with CrossEntropy loss
3. **Temporal Classifier (Angular Loss)** - LSTM-based temporal model with combined CE + Angular Distance loss

See [FINAL_MODEL_COMPARISON.md](FINAL_MODEL_COMPARISON.md) for detailed results and analysis.

## Key Results

| Model | Top-1 Accuracy | Top-5 Accuracy | Inference Time | FPS |
|-------|----------------|----------------|----------------|-----|
| Baseline BeamTransFuser | 41.05% | 83.34% | 22.46 ms | 44.5 |
| Temporal CE Loss | 41.84% | 83.57% | 63.71 ms | 15.7 |
| **Temporal Angular Loss** | **42.96%** | **84.32%** | 73.98 ms | 13.5 |

**Best Model**: Temporal Angular Loss achieves highest accuracy with potential for real-time performance through data pipeline optimizations.

## Repository Structure

```
beam_pred_pipeline/
├── model.py                          # Baseline BeamTransFuser architecture
├── temporal_classifier.py            # Temporal LSTM-based classifier
├── angular_loss.py                   # Angular distance loss implementation
├── dataset.py                        # Dataset classes (BeamDataset, SequenceBeamDataset)
├── train_temporal_angular.py         # Training script for temporal models
├── eval_baseline.py                  # Evaluation script for baseline model
├── eval_temporal.py                  # Evaluation script for temporal models
├── FINAL_MODEL_COMPARISON.md         # Comprehensive results and analysis
├── TEMPORAL_ANGULAR_MODEL_FILES.md   # Documentation of temporal model files
└── README.md                         # This file
```

## Model Architectures

### Baseline BeamTransFuser
- Multi-modal fusion with ResNet backbones (ResNet34 for Camera/LiDAR, ResNet18 for Radar)
- Cross-modal transformer fusion at multiple stages
- ~47.8M parameters
- Input: Single frame (Camera, LiDAR, Radar, GPS)

### Temporal Classifier
- Frozen BeamTransFuser backbone + LSTM (512 hidden, 2 layers)
- ~52.2M total parameters (~4.4M LSTM)
- Input: 10-frame sequences
- Two variants: CE Loss and Angular Loss

### Angular Loss
- Combined loss: α × CrossEntropy + β × Angular Distance
- Default: α=1.0, β=0.5
- Angular distance: Normalized beam index difference
- Helps model understand beam proximity in 1D beam array

## Installation

```bash
# Clone repository
git clone <repository-url>
cd beam_pred_pipeline

# Install dependencies
pip install torch torchvision
pip install pandas numpy pillow tqdm
pip install plyfile  # For LiDAR processing
```

## Usage

### Training Temporal Model

```bash
# Train with CrossEntropy loss
python train_temporal_angular.py --loss_type ce --epochs 20

# Train with Angular loss
python train_temporal_angular.py --loss_type angular --epochs 20 --alpha 1.0 --beta 0.5
```

### Evaluation

```bash
# Evaluate baseline model
python eval_baseline.py

# Evaluate temporal models
python eval_temporal.py
```

## Dataset Format

The models expect data organized as follows:

```
scenario_name/
├── scenario_name_dev.csv          # CSV with file paths and labels
├── unit1_rgb/                     # RGB images
├── unit1_lidar/                   # LiDAR point clouds (.ply or .npy)
├── unit1_radar/                   # Radar data (.npy)
├── unit1_gps/                     # GPS coordinates (.txt)
└── unit1_pwr_60ghz/              # Power vectors (.txt)
```

CSV format:
```csv
unit1_rgb,unit1_lidar,unit1_radar,unit1_gps,unit1_pwr_60ghz,unit1_beam
path/to/image.png,path/to/lidar.ply,path/to/radar.npy,path/to/gps.txt,path/to/power.txt,beam_index
```

## Key Features

### Multi-Modal Fusion
- **Camera**: ResNet34 backbone with ImageNet pretraining
- **LiDAR**: ResNet34 backbone processing voxelized point clouds
- **Radar**: ResNet18 backbone for 2-channel range-Doppler maps
- **GPS**: MLP projection to feature space

### Temporal Modeling
- LSTM processes sequence of fused features
- Captures temporal dependencies in beam selection
- Final state pooling for classification

### Angular Distance Loss
- Novel loss function for beam prediction
- Considers spatial proximity of beams
- Improves accuracy over standard CrossEntropy

## Performance Optimization

The temporal models show slower inference in current evaluation due to data loading overhead. For production deployment:

1. **Sequence Buffering**: Keep last 9 frames in memory, only load 1 new frame
2. **Preprocessing Cache**: Pre-process and cache sequences
3. **Batch Processing**: Process multiple sequences in parallel
4. **Model Optimization**: TensorRT, FP16, quantization

With these optimizations, temporal models can achieve ~19ms inference (52.6 FPS) as shown in synthetic benchmarks.

## Citation

If you use this code, please cite:

```bibtex
@misc{beam_prediction_2026,
  title={Multi-Modal Temporal Beam Prediction with Angular Loss},
  author={Your Name},
  year={2026}
}
```

## License

[Add your license here]

## Acknowledgments

- ResNet architectures from torchvision
- Inspired by multi-modal fusion techniques in autonomous driving
