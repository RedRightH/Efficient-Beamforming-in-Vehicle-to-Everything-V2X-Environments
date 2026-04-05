"""
Temporal Beam Classifier with LSTM
===================================
A hybrid architecture that combines:
1. BeamTransFuser backbone for multi-modal feature extraction
2. LSTM for temporal sequence modeling
3. Supervised classification head (NOT DQN)

This model processes sequences of multi-modal observations and predicts
the optimal beam index using cross-entropy loss, similar to the baseline
BeamTransFuser but with added temporal modeling capability.

Key Differences from DQN approach:
- Uses supervised learning (cross-entropy) instead of Q-learning
- Simpler training procedure without replay buffers or epsilon-greedy
- Should achieve better accuracy than DQN while capturing temporal patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import BeamTransFuser


class TemporalBeamClassifier(nn.Module):
    """
    Temporal extension of BeamTransFuser using LSTM for sequence modeling.
    
    Architecture:
    1. BeamTransFuser extracts 512-dim features per timestep
    2. LSTM processes the sequence of features
    3. Classification head predicts beam index from final LSTM state
    
    Args:
        feature_extractor: Pre-trained BeamTransFuser model
        hidden_dim: LSTM hidden dimension (default: 512)
        num_layers: Number of LSTM layers (default: 2)
        num_beams: Number of beam classes (default: 64)
        dropout: Dropout rate (default: 0.3)
        bidirectional: Use bidirectional LSTM (default: False)
        freeze_backbone: Freeze feature extractor weights (default: False)
    """
    
    def __init__(self, feature_extractor, hidden_dim=512, num_layers=2, 
                 num_beams=64, dropout=0.3, bidirectional=False, freeze_backbone=False):
        super().__init__()
        
        self.feature_extractor = feature_extractor
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Optionally freeze the backbone
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            print("Feature extractor backbone frozen")
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=512,  # BeamTransFuser outputs 512-dim features
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        # Calculate final LSTM output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_beams)
        )
        
    def forward(self, imgs, lids, rads, gpss):
        """
        Forward pass through temporal classifier.
        
        Args:
            imgs: (B, T, C, H, W) - Sequence of camera images
            lids: (B, T, 1, H, W) - Sequence of lidar images
            rads: (B, T, 2, H, W) - Sequence of radar images
            gpss: (B, T, 2) - Sequence of GPS coordinates
            
        Returns:
            logits: (B, num_beams) - Classification logits for beam prediction
        """
        B, T, C, H, W = imgs.shape
        
        # Flatten batch and time dimensions to process all frames
        imgs_flat = imgs.view(B * T, C, H, W)
        lids_flat = lids.view(B * T, 1, H, W)
        rads_flat = rads.view(B * T, 2, H, W)
        gpss_flat = gpss.view(B * T, 2)
        
        # Extract features using BeamTransFuser backbone
        # Use forward_features to get 512-dim embeddings without classification
        features = self.feature_extractor.forward_features(
            imgs_flat, lids_flat, rads_flat, gpss_flat
        )  # (B*T, 512)
        
        # Reshape back to sequence format
        features_seq = features.view(B, T, -1)  # (B, T, 512)
        
        # Process sequence with LSTM
        lstm_out, (hn, cn) = self.lstm(features_seq)
        # lstm_out: (B, T, hidden_dim * num_directions)
        # hn: (num_layers * num_directions, B, hidden_dim)
        
        # Use the final hidden state from the last layer
        if self.bidirectional:
            # Concatenate forward and backward final states
            final_state = torch.cat([hn[-2], hn[-1]], dim=1)  # (B, hidden_dim*2)
        else:
            final_state = hn[-1]  # (B, hidden_dim)
        
        # Classification
        logits = self.classifier(final_state)  # (B, num_beams)
        
        return logits
    
    def forward_with_attention(self, imgs, lids, rads, gpss):
        """
        Forward pass with attention mechanism over LSTM outputs.
        Uses all timesteps instead of just the final state.
        
        Returns:
            logits: (B, num_beams)
            attention_weights: (B, T) - Attention weights over timesteps
        """
        B, T, C, H, W = imgs.shape
        
        # Extract features (same as regular forward)
        imgs_flat = imgs.view(B * T, C, H, W)
        lids_flat = lids.view(B * T, 1, H, W)
        rads_flat = rads.view(B * T, 2, H, W)
        gpss_flat = gpss.view(B * T, 2)
        
        features = self.feature_extractor.forward_features(
            imgs_flat, lids_flat, rads_flat, gpss_flat
        )
        features_seq = features.view(B, T, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features_seq)  # (B, T, hidden_dim)
        
        # Attention mechanism
        # Simple attention: learn weights for each timestep
        attention_scores = torch.bmm(
            lstm_out,
            lstm_out[:, -1, :].unsqueeze(2)  # Use last timestep as query
        ).squeeze(2)  # (B, T)
        
        attention_weights = F.softmax(attention_scores, dim=1)  # (B, T)
        
        # Weighted sum of LSTM outputs
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            lstm_out
        ).squeeze(1)  # (B, hidden_dim)
        
        # Classification
        logits = self.classifier(context)
        
        return logits, attention_weights


class TemporalBeamClassifierV2(nn.Module):
    """
    Enhanced version with attention pooling over sequence.
    
    Instead of using only the final LSTM state, this version:
    1. Processes the full sequence with LSTM
    2. Uses attention to weight all timesteps
    3. Pools the attended features for classification
    """
    
    def __init__(self, feature_extractor, hidden_dim=512, num_layers=2,
                 num_beams=64, dropout=0.3, freeze_backbone=False):
        super().__init__()
        
        self.feature_extractor = feature_extractor
        self.hidden_dim = hidden_dim
        
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        # Bidirectional LSTM for richer temporal context
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        lstm_output_dim = hidden_dim * 2  # Bidirectional
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_beams)
        )
    
    def forward(self, imgs, lids, rads, gpss):
        B, T, C, H, W = imgs.shape
        
        # Extract features
        imgs_flat = imgs.view(B * T, C, H, W)
        lids_flat = lids.view(B * T, 1, H, W)
        rads_flat = rads.view(B * T, 2, H, W)
        gpss_flat = gpss.view(B * T, 2)
        
        features = self.feature_extractor.forward_features(
            imgs_flat, lids_flat, rads_flat, gpss_flat
        )
        features_seq = features.view(B, T, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features_seq)  # (B, T, hidden_dim*2)
        
        # Attention pooling
        attention_scores = self.attention(lstm_out)  # (B, T, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (B, T, 1)
        
        # Weighted sum
        context = (lstm_out * attention_weights).sum(dim=1)  # (B, hidden_dim*2)
        
        # Classification
        logits = self.classifier(context)
        
        return logits


def create_temporal_classifier(pretrained_path, model_version='v1', hidden_dim=512,
                               num_layers=2, dropout=0.3, freeze_backbone=False,
                               pruning_ratio=0.25, num_beams=64):
    """
    Factory function to create temporal classifier models.
    
    Args:
        pretrained_path: Path to pretrained BeamTransFuser weights
        model_version: 'v1' (final state) or 'v2' (attention pooling)
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        freeze_backbone: Whether to freeze feature extractor
        pruning_ratio: Pruning ratio for BeamTransFuser
        num_beams: Number of beam classes
        
    Returns:
        model: TemporalBeamClassifier instance
    """
    # Create feature extractor
    feature_extractor = BeamTransFuser(num_beams=num_beams, pruning_ratio=pruning_ratio)
    
    # Load pretrained weights if available
    if pretrained_path and torch.cuda.is_available():
        device = torch.device('cuda')
        state_dict = torch.load(pretrained_path, map_location=device)
        feature_extractor.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {pretrained_path}")
    elif pretrained_path:
        state_dict = torch.load(pretrained_path, map_location='cpu')
        feature_extractor.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {pretrained_path}")
    
    # Create temporal model
    if model_version == 'v1':
        model = TemporalBeamClassifier(
            feature_extractor=feature_extractor,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_beams=num_beams,
            dropout=dropout,
            freeze_backbone=freeze_backbone
        )
        print(f"Created TemporalBeamClassifier V1 (final state pooling)")
    elif model_version == 'v2':
        model = TemporalBeamClassifierV2(
            feature_extractor=feature_extractor,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_beams=num_beams,
            dropout=dropout,
            freeze_backbone=freeze_backbone
        )
        print(f"Created TemporalBeamClassifier V2 (attention pooling)")
    else:
        raise ValueError(f"Unknown model version: {model_version}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing TemporalBeamClassifier...")
    
    # Create dummy feature extractor
    feature_extractor = BeamTransFuser(num_beams=64, pruning_ratio=0.25)
    
    # Test V1
    model_v1 = TemporalBeamClassifier(feature_extractor, hidden_dim=256, num_layers=2)
    print(f"\nV1 Model parameters: {sum(p.numel() for p in model_v1.parameters()):,}")
    
    # Test V2
    model_v2 = TemporalBeamClassifierV2(feature_extractor, hidden_dim=256, num_layers=2)
    print(f"V2 Model parameters: {sum(p.numel() for p in model_v2.parameters()):,}")
    
    # Test forward pass
    B, T = 2, 10
    imgs = torch.randn(B, T, 3, 224, 224)
    lids = torch.randn(B, T, 1, 224, 224)
    rads = torch.randn(B, T, 2, 224, 224)
    gpss = torch.randn(B, T, 2)
    
    with torch.no_grad():
        out_v1 = model_v1(imgs, lids, rads, gpss)
        out_v2 = model_v2(imgs, lids, rads, gpss)
    
    print(f"\nV1 Output shape: {out_v1.shape}")
    print(f"V2 Output shape: {out_v2.shape}")
    print("\nAll tests passed!")
