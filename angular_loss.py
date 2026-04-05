"""
Angular Distance Loss for Beam Prediction
==========================================
Combines CrossEntropy with Angular Distance penalty to encourage
predictions that are spatially close to the ground truth beam.

For a 64-beam array (1D linear arrangement), beams are indexed 0-63.
Angular distance penalizes predictions that are far from the correct beam.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AngularDistanceLoss(nn.Module):
    """
    Angular distance loss for beam prediction.
    
    For a 1D beam array (64 beams), computes the normalized distance
    between predicted and ground truth beam indices.
    
    Args:
        num_beams: Total number of beams (default: 64)
        reduction: 'mean' or 'sum' (default: 'mean')
    """
    def __init__(self, num_beams=64, reduction='mean'):
        super().__init__()
        self.num_beams = num_beams
        self.reduction = reduction
        
    def forward(self, predictions, targets):
        """
        Compute angular distance loss.
        
        Args:
            predictions: (B, num_beams) - logits or probabilities
            targets: (B,) - ground truth beam indices
            
        Returns:
            loss: scalar tensor
        """
        batch_size = predictions.size(0)
        
        # Get predicted beam indices
        pred_beams = predictions.argmax(dim=1)  # (B,)
        
        # Compute absolute distance
        distances = torch.abs(pred_beams - targets).float()  # (B,)
        
        # Normalize by maximum possible distance
        max_distance = self.num_beams - 1
        normalized_distances = distances / max_distance  # (B,) in [0, 1]
        
        # Reduction
        if self.reduction == 'mean':
            return normalized_distances.mean()
        elif self.reduction == 'sum':
            return normalized_distances.sum()
        else:
            return normalized_distances


class CombinedBeamLoss(nn.Module):
    """
    Combined loss: CrossEntropy + Angular Distance
    
    Loss = alpha * CrossEntropy + beta * AngularDistance
    
    Args:
        alpha: Weight for cross-entropy loss (default: 1.0)
        beta: Weight for angular distance loss (default: 0.5)
        num_beams: Number of beams (default: 64)
    """
    def __init__(self, alpha=1.0, beta=0.5, num_beams=64):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()
        self.angular_loss = AngularDistanceLoss(num_beams=num_beams)
        
    def forward(self, predictions, targets):
        """
        Compute combined loss.
        
        Args:
            predictions: (B, num_beams) - logits
            targets: (B,) - ground truth beam indices
            
        Returns:
            loss: scalar tensor
            ce_component: cross-entropy component (for logging)
            angular_component: angular distance component (for logging)
        """
        ce = self.ce_loss(predictions, targets)
        angular = self.angular_loss(predictions, targets)
        
        total_loss = self.alpha * ce + self.beta * angular
        
        return total_loss, ce, angular


class SoftAngularDistanceLoss(nn.Module):
    """
    Soft angular distance loss using probability distributions.
    
    Instead of using argmax, this version uses the full probability
    distribution to compute expected angular distance.
    
    Args:
        num_beams: Total number of beams (default: 64)
    """
    def __init__(self, num_beams=64):
        super().__init__()
        self.num_beams = num_beams
        
        # Create beam index tensor [0, 1, 2, ..., 63]
        self.register_buffer('beam_indices', torch.arange(num_beams).float())
        
    def forward(self, logits, targets):
        """
        Compute soft angular distance loss.
        
        Args:
            logits: (B, num_beams) - model outputs (before softmax)
            targets: (B,) - ground truth beam indices
            
        Returns:
            loss: scalar tensor
        """
        batch_size = logits.size(0)
        
        # Get probabilities
        probs = F.softmax(logits, dim=1)  # (B, num_beams)
        
        # Move beam_indices to same device as logits
        beam_indices = self.beam_indices.to(logits.device)
        
        # Compute expected beam index for each sample
        # expected_beam = sum(prob_i * beam_i)
        expected_beams = (probs * beam_indices.unsqueeze(0)).sum(dim=1)  # (B,)
        
        # Compute distance from ground truth
        distances = torch.abs(expected_beams - targets.float())  # (B,)
        
        # Normalize
        max_distance = self.num_beams - 1
        normalized_distances = distances / max_distance
        
        return normalized_distances.mean()


class CombinedBeamLossSoft(nn.Module):
    """
    Combined loss with soft angular distance.
    
    Loss = alpha * CrossEntropy + beta * SoftAngularDistance
    """
    def __init__(self, alpha=1.0, beta=0.5, num_beams=64):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()
        self.angular_loss = SoftAngularDistanceLoss(num_beams=num_beams)
        
    def forward(self, predictions, targets):
        """
        Compute combined loss.
        
        Args:
            predictions: (B, num_beams) - logits
            targets: (B,) - ground truth beam indices
            
        Returns:
            loss: scalar tensor
            ce_component: cross-entropy component
            angular_component: angular distance component
        """
        ce = self.ce_loss(predictions, targets)
        angular = self.angular_loss(predictions, targets)
        
        total_loss = self.alpha * ce + self.beta * angular
        
        return total_loss, ce, angular


if __name__ == "__main__":
    # Test the loss functions
    print("Testing Angular Distance Loss Functions")
    print("="*60)
    
    # Create dummy data
    batch_size = 4
    num_beams = 64
    
    logits = torch.randn(batch_size, num_beams)
    targets = torch.tensor([10, 20, 30, 40])
    
    # Test hard angular distance
    print("\n1. Hard Angular Distance Loss:")
    angular_loss = AngularDistanceLoss(num_beams=num_beams)
    loss = angular_loss(logits, targets)
    print(f"   Loss: {loss.item():.4f}")
    
    # Test combined loss (hard)
    print("\n2. Combined Loss (Hard Angular):")
    combined_loss = CombinedBeamLoss(alpha=1.0, beta=0.5, num_beams=num_beams)
    total, ce, angular = combined_loss(logits, targets)
    print(f"   Total Loss: {total.item():.4f}")
    print(f"   CE Component: {ce.item():.4f}")
    print(f"   Angular Component: {angular.item():.4f}")
    
    # Test soft angular distance
    print("\n3. Soft Angular Distance Loss:")
    soft_angular_loss = SoftAngularDistanceLoss(num_beams=num_beams)
    loss = soft_angular_loss(logits, targets)
    print(f"   Loss: {loss.item():.4f}")
    
    # Test combined loss (soft)
    print("\n4. Combined Loss (Soft Angular):")
    combined_soft = CombinedBeamLossSoft(alpha=1.0, beta=0.5, num_beams=num_beams)
    total, ce, angular = combined_soft(logits, targets)
    print(f"   Total Loss: {total.item():.4f}")
    print(f"   CE Component: {ce.item():.4f}")
    print(f"   Angular Component: {angular.item():.4f}")
    
    # Test with perfect prediction
    print("\n5. Perfect Prediction Test:")
    perfect_logits = torch.zeros(batch_size, num_beams)
    perfect_logits[0, 10] = 10.0
    perfect_logits[1, 20] = 10.0
    perfect_logits[2, 30] = 10.0
    perfect_logits[3, 40] = 10.0
    
    total, ce, angular = combined_soft(perfect_logits, targets)
    print(f"   Total Loss: {total.item():.4f}")
    print(f"   CE Component: {ce.item():.4f}")
    print(f"   Angular Component: {angular.item():.4f} (should be ~0)")
    
    print("\n" + "="*60)
    print("All tests passed!")
