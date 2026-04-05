"""
Training Script for Temporal Beam Classifier with Angular Loss
===============================================================
Trains the LSTM-enhanced BeamTransFuser using:
    CrossEntropy + Angular Distance Loss

This encourages predictions that are spatially close to the ground truth,
which is important for beam prediction where nearby beams have similar coverage.

Output files are saved with '_angular' suffix to avoid overwriting previous models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import argparse
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

from dataset import SequenceBeamDataset
from temporal_classifier import create_temporal_classifier
from angular_loss import CombinedBeamLossSoft


def top_k_accuracy(output, target, topk=(1, 3, 5)):
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    
    running_loss = 0.0
    running_ce = 0.0
    running_angular = 0.0
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for batch_idx, (imgs, lids, rads, gpss, labels, _) in enumerate(pbar):
        imgs = imgs.to(device)
        lids = lids.to(device)
        rads = rads.to(device)
        gpss = gpss.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs, lids, rads, gpss)
        
        # Compute combined loss
        loss, ce_component, angular_component = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        running_ce += ce_component.item()
        running_angular += angular_component.item()
        
        accs = top_k_accuracy(outputs, labels, topk=(1, 3, 5))
        top1_correct += accs[0] * labels.size(0) / 100.0
        top3_correct += accs[1] * labels.size(0) / 100.0
        top5_correct += accs[2] * labels.size(0) / 100.0
        total += labels.size(0)
        
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'ce': running_ce / (batch_idx + 1),
                'ang': running_angular / (batch_idx + 1),
                'top1': 100.0 * top1_correct / total
            })
    
    epoch_loss = running_loss / len(loader)
    epoch_ce = running_ce / len(loader)
    epoch_angular = running_angular / len(loader)
    epoch_top1 = 100.0 * top1_correct / total
    epoch_top3 = 100.0 * top3_correct / total
    epoch_top5 = 100.0 * top5_correct / total
    
    return epoch_loss, epoch_ce, epoch_angular, epoch_top1, epoch_top3, epoch_top5


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    
    running_loss = 0.0
    running_ce = 0.0
    running_angular = 0.0
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, lids, rads, gpss, labels, _ in tqdm(loader, desc='Validation'):
            imgs = imgs.to(device)
            lids = lids.to(device)
            rads = rads.to(device)
            gpss = gpss.to(device)
            labels = labels.to(device)
            
            outputs = model(imgs, lids, rads, gpss)
            loss, ce_component, angular_component = criterion(outputs, labels)
            
            running_loss += loss.item()
            running_ce += ce_component.item()
            running_angular += angular_component.item()
            
            accs = top_k_accuracy(outputs, labels, topk=(1, 3, 5))
            top1_correct += accs[0] * labels.size(0) / 100.0
            top3_correct += accs[1] * labels.size(0) / 100.0
            top5_correct += accs[2] * labels.size(0) / 100.0
            total += labels.size(0)
    
    val_loss = running_loss / len(loader)
    val_ce = running_ce / len(loader)
    val_angular = running_angular / len(loader)
    val_top1 = 100.0 * top1_correct / total
    val_top3 = 100.0 * top3_correct / total
    val_top5 = 100.0 * top5_correct / total
    
    return val_loss, val_ce, val_angular, val_top1, val_top3, val_top5


def train_temporal_classifier(args):
    """Main training function."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("\nLoading datasets...")
    scenarios = [
        (r"c:/Users/suchi/BTP - Semester 7 Sionna/scenario31", "scenario31_dev.csv"),
        (r"c:/Users/suchi/BTP - Semester 7 Sionna/scenario32", "scenario32_dev.csv"),
        (r"c:/Users/suchi/BTP - Semester 7 Sionna/scenario33", "scenario33_dev.csv")
    ]
    
    train_datasets = []
    val_datasets = []
    
    for scenario_dir, csv_name in scenarios:
        csv_file = os.path.join(scenario_dir, csv_name)
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found, skipping...")
            continue
        
        print(f"Loading {scenario_dir}...")
        
        train_ds = SequenceBeamDataset(
            csv_file=csv_file,
            root_dir=scenario_dir,
            transform=transform,
            mode='train',
            val_split=args.val_split,
            seq_len=args.seq_len
        )
        train_datasets.append(train_ds)
        
        val_ds = SequenceBeamDataset(
            csv_file=csv_file,
            root_dir=scenario_dir,
            transform=transform,
            mode='val',
            val_split=args.val_split,
            seq_len=args.seq_len
        )
        val_datasets.append(val_ds)
    
    train_set = ConcatDataset(train_datasets)
    val_set = ConcatDataset(val_datasets)
    
    print(f"Training sequences: {len(train_set)}")
    print(f"Validation sequences: {len(val_set)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    print(f"\nCreating {args.model_version} model...")
    model = create_temporal_classifier(
        pretrained_path=args.pretrained_path,
        model_version=args.model_version,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        pruning_ratio=args.pruning_ratio,
        num_beams=64
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Combined loss with angular distance
    print(f"\nUsing Combined Loss: CE (alpha={args.alpha}) + Angular (beta={args.beta})")
    criterion = CombinedBeamLossSoft(alpha=args.alpha, beta=args.beta, num_beams=64)
    
    # Optimizer
    if args.freeze_backbone:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        backbone_params = list(model.feature_extractor.parameters())
        new_params = list(model.lstm.parameters()) + list(model.classifier.parameters())
        
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': args.lr * 0.1},
            {'params': new_params, 'lr': args.lr}
        ], weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    print("\nStarting training with Angular Loss...")
    best_val_top1 = 0.0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_ce, train_ang, train_top1, train_top3, train_top5 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        print(f"\nTraining Results:")
        print(f"  Total Loss: {train_loss:.4f} (CE: {train_ce:.4f}, Angular: {train_ang:.4f})")
        print(f"  Top-1: {train_top1:.2f}%")
        print(f"  Top-3: {train_top3:.2f}%")
        print(f"  Top-5: {train_top5:.2f}%")
        
        # Validate
        val_loss, val_ce, val_ang, val_top1, val_top3, val_top5 = validate(
            model, val_loader, criterion, device
        )
        
        print(f"\nValidation Results:")
        print(f"  Total Loss: {val_loss:.4f} (CE: {val_ce:.4f}, Angular: {val_ang:.4f})")
        print(f"  Top-1: {val_top1:.2f}%")
        print(f"  Top-3: {val_top3:.2f}%")
        print(f"  Top-5: {val_top5:.2f}%")
        
        # Update scheduler
        scheduler.step(val_top1)
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_ce': train_ce,
            'train_angular': train_ang,
            'train_top1': train_top1,
            'train_top3': train_top3,
            'train_top5': train_top5,
            'val_loss': val_loss,
            'val_ce': val_ce,
            'val_angular': val_ang,
            'val_top1': val_top1,
            'val_top3': val_top3,
            'val_top5': val_top5
        })
        
        # Save best model (with _angular suffix)
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            save_path = f'best_temporal_classifier_{args.model_version}_angular.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_top1': val_top1,
                'val_top3': val_top3,
                'val_top5': val_top5,
                'alpha': args.alpha,
                'beta': args.beta,
                'args': args
            }, save_path)
            print(f"\n✓ Saved best model: {save_path} (Top-1: {val_top1:.2f}%)")
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = f'temporal_classifier_{args.model_version}_angular_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_top1': val_top1,
                'args': args
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(f'training_history_{args.model_version}_angular.csv', index=False)
    print(f"\nTraining history saved to training_history_{args.model_version}_angular.csv")
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Validation Top-1 Accuracy: {best_val_top1:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Temporal Beam Classifier with Angular Loss')
    
    # Model arguments
    parser.add_argument('--model_version', type=str, default='v1', choices=['v1', 'v2'],
                        help='Model version: v1 (final state) or v2 (attention)')
    parser.add_argument('--pretrained_path', type=str, default='retrained_best_beam_model.pth',
                        help='Path to pretrained BeamTransFuser weights')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='LSTM hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze feature extractor backbone')
    parser.add_argument('--pruning_ratio', type=float, default=0.25,
                        help='Pruning ratio for BeamTransFuser')
    
    # Loss arguments
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for cross-entropy loss')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Weight for angular distance loss')
    
    # Data arguments
    parser.add_argument('--seq_len', type=int, default=10,
                        help='Sequence length')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Temporal Beam Classifier Training with Angular Loss")
    print("="*60)
    print(f"Model Version: {args.model_version}")
    print(f"Loss: CE (α={args.alpha}) + Angular (β={args.beta})")
    print(f"Sequence Length: {args.seq_len}")
    print(f"Hidden Dim: {args.hidden_dim}")
    print(f"LSTM Layers: {args.num_layers}")
    print(f"Freeze Backbone: {args.freeze_backbone}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print("="*60)
    
    train_temporal_classifier(args)
