# Final Model Comparison Results

## Summary Table

| Model | Checkpoint | Top-1 | Top-3 | Top-5 | Top-10 | Inference (ms) | FPS |
|-------|-----------|-------|-------|-------|--------|----------------|-----|
| **Baseline BeamTransFuser** | retrained_best_beam_model.pth | 41.05% | 74.83% | 83.34% | 89.07% | 18.08 | 55.3 |
| **Temporal CE Loss** | best_temporal_classifier_v1.pth | 41.84% | 73.88% | 83.57% | 89.00% | 18.99 | 52.6 |
| **Temporal Angular Loss** | best_temporal_classifier_v1_angular.pth | **42.96%** | **75.14%** | **84.32%** | 88.81% | 18.99 | 52.6 |

## Key Findings

### Accuracy
1. **Best Overall**: Temporal Angular Loss
   - Top-1: 42.96% (+1.91% vs Baseline, +1.12% vs Temporal CE)
   - Top-3: 75.14% (+0.31% vs Baseline, +1.26% vs Temporal CE)
   - Top-5: 84.32% (+0.98% vs Baseline, +0.75% vs Temporal CE)

2. **Temporal CE vs Baseline**
   - Marginal improvements: +0.79% Top-1, -0.95% Top-3
   - Temporal modeling provides slight benefit

3. **Angular Loss Benefit**
   - Consistently outperforms CE loss across Top-1/3/5
   - Angular distance helps with beam selection accuracy

### Inference Time

#### Synthetic Benchmark (Previous - Pure Model Inference)
- **Baseline**: 18.08 ms (55.3 FPS)
- **Temporal**: 18.99 ms (52.6 FPS)
- **Overhead**: Only +0.91 ms for LSTM processing
   - **Actual LSTM overhead: Only 0.91 ms!**

## Recommendations

### For Production Deployment

1. **If Accuracy is Priority**
   - Use **Temporal Angular Loss** model
   - Best Top-1 accuracy: 42.96%
   - Implement data pipeline optimizations (see below)

2. **If Real-Time is Critical**
   - Use **Baseline BeamTransFuser**
   - 55.3 FPS with optimized pipeline
   - Still achieves 41.05% Top-1 accuracy

3. **Balanced Approach**
   - Use **Temporal CE Loss** model with optimizations
   - 41.84% accuracy (close to Angular)

## Validation Dataset

- **Scenarios**: scenario31, scenario32, scenario33
- **Total Samples**: 
  - Baseline: 6,541 single frames
  - Temporal: 2,691 sequences (10 frames each)
- **Split**: 20% validation, 80% training
- **Preprocessing**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Model Details

### Baseline BeamTransFuser
- **Architecture**: Multi-modal fusion with ResNet backbones
- **Parameters**: ~47.8M
- **Input**: Single frame (Camera, LiDAR, Radar, GPS)
- **Training**: Retrained on scenario31/32/33

### Temporal Classifier V1
- **Architecture**: Frozen BeamTransFuser + LSTM (512 hidden, 2 layers)
- **Parameters**: ~52.2M total (~4.4M LSTM)
- **Input**: 10-frame sequences
- **Backbone**: Frozen pretrained BeamTransFuser
- **Training Epochs**: 12

### Loss Functions
- **CE Loss**: Standard CrossEntropyLoss
- **Angular Loss**: Combined CrossEntropyLoss (α=1.0) + Angular Distance (β=0.5)
  - Angular distance: Normalized beam index difference
  - Helps model understand beam proximity

## Conclusion

**Best Model Choice Depends on Use Case:**

1. **Maximum Accuracy**: Temporal Angular Loss (42.96% Top-1)
   - Requires data pipeline optimization for real-time
   - Worth the effort if accuracy is critical

2. **Real-Time Ready**: Baseline BeamTransFuser (41.05% Top-1, 44.5 FPS)
   - Works out-of-the-box with real data
   - Minimal accuracy trade-off (1.91% lower)
