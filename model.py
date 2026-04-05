import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, resnet18

# -----------------------------------------------------------------------------
# 1. Encoders & Backbones
# -----------------------------------------------------------------------------

class BaseEncoder(nn.Module):
    """
    Unified architectural design: Conv -> BN -> ReLU -> MaxPool.
    Projects inputs to 64 channels.
    """
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # x: (B, C_in, 224, 224) -> (B, 64, 56, 56)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class ResNet34Backbone(nn.Module):
    """
    Standard ResNet34 stages (Layers 1-4).
    """
    def __init__(self):
        super().__init__()
        basenet = resnet34(pretrained=True)
        self.layer1 = basenet.layer1 # 64 -> 64
        self.layer2 = basenet.layer2 # 64 -> 128
        self.layer3 = basenet.layer3 # 128 -> 256
        self.layer4 = basenet.layer4 # 256 -> 512
        
        # We need access to channel counts for fusion blocks
        self.dims = [64, 128, 256, 512]

    def forward(self, x):
        # Returns list of features at each stage for fusion
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]

class ResNet16Backbone(nn.Module):
    """
    Lightweight ResNet variant (using ResNet18 structure but potentially fewer layers or channels).
    For simplicity, we use ResNet18 here but can interpret it as the 'lightweight' choice relative to 34.
    The paper cites ResNet16 [22], usually structurally similar to 18/34 but shallower.
    We will use ResNet18 to ensure pretrained weights availability and stability.
    """
    def __init__(self):
        super().__init__()
        basenet = resnet18(pretrained=True)
        self.layer1 = basenet.layer1
        self.layer2 = basenet.layer2
        self.layer3 = basenet.layer3
        self.layer4 = basenet.layer4
        self.dims = [64, 128, 256, 512]

    def forward(self, x):
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]

class GPSEncoder(nn.Module):
    """
    Project GPS [lat, lon] to feature space matching the fusion tokens.
    """
    def __init__(self, token_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, token_dim)
        )
    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------------------------
# 2. Multi-Modal Fusion Block
# -----------------------------------------------------------------------------

class FusionBlock(nn.Module):
    def __init__(self, channel_dim, token_dim=64, num_heads=4):
        super().__init__()
        
        # Projections with Static Pruning (Reduced Dim) support
        # We allow token_dim to be different from channel_dim
        self.cam_proj = nn.Conv2d(channel_dim, token_dim, kernel_size=1)
        self.lid_proj = nn.Conv2d(channel_dim, token_dim, kernel_size=1)
        self.rad_proj = nn.Conv2d(channel_dim, token_dim, kernel_size=1)
        self.gps_proj = nn.Linear(channel_dim if channel_dim==64 else 64, token_dim)
        
        # Transformer with Dropout
        self.transformer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=num_heads, dim_feedforward=token_dim*4, dropout=0.3, batch_first=True)
        
        # Back-projection
        self.cam_back = nn.Conv2d(token_dim, channel_dim, kernel_size=1)
        self.lid_back = nn.Conv2d(token_dim, channel_dim, kernel_size=1)
        self.rad_back = nn.Conv2d(token_dim, channel_dim, kernel_size=1)
        
        self.dropout = nn.Dropout(p=0.3)
        
        # Pool size
        self.pool_size = 4 

    def forward(self, feats_list, gps_feat):
        # ... (same logic)
        cam, lid, rad = feats_list
        B, C, H, W = cam.shape
        
        # Tokenize
        cam_tok = F.adaptive_avg_pool2d(self.cam_proj(cam), self.pool_size).flatten(2).transpose(1, 2) 
        lid_tok = F.adaptive_avg_pool2d(self.lid_proj(lid), self.pool_size).flatten(2).transpose(1, 2)
        rad_tok = F.adaptive_avg_pool2d(self.rad_proj(rad), self.pool_size).flatten(2).transpose(1, 2)
        
        gps_embed = self.gps_proj(gps_feat)
        gps_tok = gps_embed.unsqueeze(1)
        
        tokens = torch.cat([cam_tok, lid_tok, rad_tok, gps_tok], dim=1)
        
        # Transformer
        fused_tokens = self.transformer(tokens)
        fused_tokens = self.dropout(fused_tokens) # Add dropout after fusion
        
        # 4. Split & Reintegrate
        # We only really care about updating the spatial features of the backbones.
        # GPS update is often ignored for backbone flow in these architectures unless specificed.
        # The paper says: "reintegrated into their corresponding branches via residual connections"
        
        # Extract chunks
        seq_len = self.pool_size * self.pool_size
        cam_out_tok = fused_tokens[:, 0:seq_len, :].transpose(1, 2).reshape(B, -1, self.pool_size, self.pool_size)
        lid_out_tok = fused_tokens[:, seq_len:2*seq_len, :].transpose(1, 2).reshape(B, -1, self.pool_size, self.pool_size)
        rad_out_tok = fused_tokens[:, 2*seq_len:3*seq_len, :].transpose(1, 2).reshape(B, -1, self.pool_size, self.pool_size)
        
        # Upsample back to H, W and Project
        cam_res = self.cam_back(F.interpolate(cam_out_tok, size=(H, W), mode='bilinear', align_corners=False))
        lid_res = self.lid_back(F.interpolate(lid_out_tok, size=(H, W), mode='bilinear', align_corners=False))
        rad_res = self.rad_back(F.interpolate(rad_out_tok, size=(H, W), mode='bilinear', align_corners=False))
        
        # Residual Addition
        cam_new = cam + cam_res
        lid_new = lid + lid_res
        rad_new = rad + rad_res
        
        return [cam_new, lid_new, rad_new]

# -----------------------------------------------------------------------------
# 3. Overall Framework
# -----------------------------------------------------------------------------

class BeamTransFuser(nn.Module):
    def __init__(self, num_beams=64, pruning_ratio=0.0):
        super().__init__()
        
        # 1. Initial Encoders (64 channels)
        self.cam_enc = BaseEncoder(in_channels=3, out_channels=64)
        self.lid_enc = BaseEncoder(in_channels=1, out_channels=64)
        self.rad_enc = BaseEncoder(in_channels=2, out_channels=64)
        self.gps_enc = GPSEncoder(token_dim=64) # Project GPS directly to matching DIM for later fusion? 
                                                # Paper says GPS -> MLP -> Fusion.
                                                # The Fusion block will handle further projection.
        
        # 2. Backbones
        self.cam_backbone = ResNet34Backbone()
        self.lid_backbone = ResNet16Backbone()
        self.rad_backbone = ResNet16Backbone()
        
        # 3. Fusion Blocks (Interleaved)
        # Inserted between layers.
        # Layer dimensions for ResNets: [64, 128, 256, 512]
        # We place fusion blocks after Layer 1, Layer 2, Layer 3, Layer 4?
        # Paper: "between Layer 1 and Layer 2" etc.
        # So we fuse the outputs of Layer 1, then pass to Layer 2.
        
        # 3. Fusion Blocks (Interleaved) with Compression Support
        # Pruning Ratio r reduces the token_dim inside fusion blocks.
        # Original dims: 64, 128, 256, 512
        # If pruning_ratio=0.5 -> dims become 32, 64, 128, 256
        
        self.pruning_ratio = pruning_ratio
        
        # Helper to calc dim
        def get_dim(d, r, heads=4):
            # Dim must be divisible by num_heads
            val = int(d * (1 - r))
            # ensure divisible by heads
            val = (val // heads) * heads
            if val < heads: val = heads
            return val
            
        d1 = get_dim(64, pruning_ratio)
        d2 = get_dim(128, pruning_ratio)
        d3 = get_dim(256, pruning_ratio)
        d4 = get_dim(512, pruning_ratio)
        
        self.fusion1 = FusionBlock(channel_dim=64, token_dim=d1)
        self.fusion2 = FusionBlock(channel_dim=128, token_dim=d2)
        self.fusion3 = FusionBlock(channel_dim=256, token_dim=d3)
        self.fusion4 = FusionBlock(channel_dim=512, token_dim=d4)
        
        # 4. Aggregation
        # Final feature map size (B, 512, 7, 7) usually
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Softmax Weighting
        self.modality_weights = nn.Parameter(torch.ones(4)) # Cam, Lid, Rad, GPS
        
        # 5. Beam Generator Head
        # Concatenated dimension -> MLP
        # Fused_F dim.
        # If we weighted-average the modalities, dim is 512.
        # If we concat, it's 512*3 + 64?
        # Paper: "softmax-weighted fusion ... adaptively weigh each modality's contribution"
        # "aggregated via a softmax-weighted fusion mechanism... denoted by (x) in Fig. 2"
        # This implies a weighted SUM or similar.
        # Let's assume we project GPS up to 512 to match others, then weighted sum.
        
        self.gps_final_proj = nn.Linear(64, 512)
        
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5), # Regularization
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # Regularization
            nn.Linear(128, num_beams)
        )

    def forward(self, img, lid, rad, gps):
        # 1. Encode
        f_cam = self.cam_enc(img) # (B, 64, 56, 56)
        f_lid = self.lid_enc(lid)
        f_rad = self.rad_enc(rad)
        f_gps = self.gps_enc(gps) # (B, 64)
        
        # 2. Stage 1
        c1 = self.cam_backbone.layer1(f_cam)
        l1 = self.lid_backbone.layer1(f_lid)
        r1 = self.rad_backbone.layer1(f_rad)
        
        # Fusion 1
        c1, l1, r1 = self.fusion1([c1, l1, r1], f_gps)
        
        # Stage 2
        c2 = self.cam_backbone.layer2(c1)
        l2 = self.lid_backbone.layer2(l1)
        r2 = self.rad_backbone.layer2(r1)
        
        # Fusion 2
        # GPS projection logic: GPS vector is static/global. We can keep feeding the same f_gps 
        # or a projected version. The block handles projection.
        # We need to project GPS to 128 dim if using it as token in fusion2?
        # My FusionBlock handles projection from 'channel_dim' (128) to 'token_dim' (128).
        # But GPS is stuck at 64 dim. 
        # Fix: We should probably project GPS appropriately outside or inside based on block.
        # Let's assume FusionBlock expects a `gps_feat` it can project.
        # My FusionBlock implementation: `self.gps_proj = nn.Linear(channel_dim if ... else 64)`
        # This handles the mismatch.
        
        c2, l2, r2 = self.fusion2([c2, l2, r2], f_gps)
        
        # Stage 3
        c3 = self.cam_backbone.layer3(c2)
        l3 = self.lid_backbone.layer3(l2)
        r3 = self.rad_backbone.layer3(r2)
        
        # Fusion 3
        c3, l3, r3 = self.fusion3([c3, l3, r3], f_gps)
        
        # Stage 4
        c4 = self.cam_backbone.layer4(c3)
        l4 = self.lid_backbone.layer4(l3)
        r4 = self.rad_backbone.layer4(r3)
        
        # Fusion 4
        c4, l4, r4 = self.fusion4([c4, l4, r4], f_gps)
        
        # 3. Global Pool & Weighted Aggregation
        # Pool to vectors (B, 512)
        v_cam = self.pool(c4).flatten(1)
        v_lid = self.pool(l4).flatten(1)
        v_rad = self.pool(r4).flatten(1)
        
        # Project GPS to 512
        v_gps = self.gps_final_proj(f_gps)
        
        # Stack: (B, 4, 512)
        stack = torch.stack([v_cam, v_lid, v_rad, v_gps], dim=1)
        
        # Weights
        weights = F.softmax(self.modality_weights, dim=0) # (4,)
        weights = weights.view(1, 4, 1)
        
        # Weighted Sum
        fused_vector = (stack * weights).sum(dim=1) # (B, 512)
        
        # 4. Head
        out = self.head(fused_vector)
        return out

    def forward_features(self, img, lid, rad, gps):
        """
        Extracts the fused feature vector (B, 512) without the classification head.
        """
        # 1. Encode
        f_cam = self.cam_enc(img) 
        f_lid = self.lid_enc(lid)
        f_rad = self.rad_enc(rad)
        f_gps = self.gps_enc(gps) 
        
        # 2. Stage 1
        c1 = self.cam_backbone.layer1(f_cam)
        l1 = self.lid_backbone.layer1(f_lid)
        r1 = self.rad_backbone.layer1(f_rad)
        
        # Fusion 1
        c1, l1, r1 = self.fusion1([c1, l1, r1], f_gps)
        
        # Stage 2
        c2 = self.cam_backbone.layer2(c1)
        l2 = self.lid_backbone.layer2(l1)
        r2 = self.rad_backbone.layer2(r1)
        
        c2, l2, r2 = self.fusion2([c2, l2, r2], f_gps)
        
        # Stage 3
        c3 = self.cam_backbone.layer3(c2)
        l3 = self.lid_backbone.layer3(l2)
        r3 = self.rad_backbone.layer3(r2)
        
        c3, l3, r3 = self.fusion3([c3, l3, r3], f_gps)
        
        # Stage 4
        c4 = self.cam_backbone.layer4(c3)
        l4 = self.lid_backbone.layer4(l3)
        r4 = self.rad_backbone.layer4(r3)
        
        c4, l4, r4 = self.fusion4([c4, l4, r4], f_gps)
        
        # 3. Global Pool & Weighted Aggregation
        v_cam = self.pool(c4).flatten(1)
        v_lid = self.pool(l4).flatten(1)
        v_rad = self.pool(r4).flatten(1)
        
        v_gps = self.gps_final_proj(f_gps)
        
        stack = torch.stack([v_cam, v_lid, v_rad, v_gps], dim=1)
        
        weights = F.softmax(self.modality_weights, dim=0)
        weights = weights.view(1, 4, 1)
        
        fused_vector = (stack * weights).sum(dim=1) # (B, 512)
        return fused_vector

class DuelingHead(nn.Module):
    """
    Dueling Network Architecture:
    Splits the network into two streams: Value (V) and Advantage (A).
    Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
    """
    def __init__(self, input_dim=512, action_dim=64):
        super().__init__()
        
        # Value Stream (Scalar)
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1) 
        )
        
        # Advantage Stream (Vector of action_dim)
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
    def forward(self, x):
        value = self.value_stream(x) # (B, 1)
        advantage = self.advantage_stream(x) # (B, action_dim)
        
        # Combine
        # Q = V + (A - mean(A))
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals

class TemporalBeamTransFuser(nn.Module):
    """
    Wraps BeamTransFuser to handle sequences.
    1. Extracts features for each frame in sequence.
    2. Aggregates features using LSTM/GRU.
    3. Feeds aggregated state to DuelingHead.
    """
    def __init__(self, feature_extractor, hidden_dim=512, num_beams=64):
        super().__init__()
        self.feature_extractor = feature_extractor
        
        # Freeze feature extractor if needed (optional)
        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False
            
        # Temporal Encoder
        self.rnn = nn.LSTM(input_size=512, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        
        # Dueling Head
        self.head = DuelingHead(input_dim=hidden_dim, action_dim=num_beams)
        
    def forward(self, imgs, lids, rads, gpss):
        # Input shape: (B, T, C, H, W) or (B, T, 2)
        B, T, C, H, W = imgs.shape
        
        # Flatten B and T to process all frames
        imgs_flat = imgs.view(B*T, C, H, W)
        lids_flat = lids.view(B*T, 1, H, W)
        rads_flat = rads.view(B*T, 2, H, W)
        gpss_flat = gpss.view(B*T, 2)
        
        # Extract features
        # We need to rely on forward_features
        features = self.feature_extractor.forward_features(imgs_flat, lids_flat, rads_flat, gpss_flat) # (B*T, 512)
        
        # Reshape back to sequence
        features_seq = features.view(B, T, -1) # (B, T, 512)
        
        # RNN
        # out: (B, T, hidden_dim), (hn, cn)
        # We take the last hidden state
        _, (hn, _) = self.rnn(features_seq) # hn: (num_layers, B, hidden_dim)
        
        final_state = hn[-1] # (B, hidden_dim)
        
        # Dueling Head
        q_values = self.head(final_state) # (B, num_beams)
        
        return q_values
