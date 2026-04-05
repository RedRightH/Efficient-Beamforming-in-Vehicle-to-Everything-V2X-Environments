"""
Evaluate Temporal Classifier Models
"""
import torch
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset

from temporal_classifier import create_temporal_classifier
from dataset import SequenceBeamDataset


def top_k_accuracy(outputs, targets, k=1):
    """Calculate top-k accuracy."""
    _, pred = outputs.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    correct_k = correct[:k].reshape(-1).float().sum(0)
    return correct_k.mul_(100.0 / targets.size(0)).item()


def custom_collate(batch):
    """Custom collate function to handle 6 return values from SequenceBeamDataset."""
    imgs, lids, rads, gpss, labels, powers = zip(*batch)
    
    imgs = torch.stack(imgs)
    lids = torch.stack(lids)
    rads = torch.stack(rads)
    gpss = torch.stack(gpss)
    labels = torch.tensor(labels)
    # We don't need powers for evaluation, so we can drop it
    
    return imgs, lids, rads, gpss, labels


def benchmark_inference(model, dataloader, device, num_samples=100):
    """Benchmark inference time."""
    model.eval()
    times = []
    
    with torch.no_grad():
        for i, (img, lid, rad, gps, labels) in enumerate(dataloader):
            if i >= num_samples:
                break
            
            img = img.to(device)
            lid = lid.to(device)
            rad = rad.to(device)
            gps = gps.to(device)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(img, lid, rad, gps)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    return {
        'mean': times.mean(),
        'std': times.std(),
        'min': times.min(),
        'max': times.max(),
        'fps': 1000.0 / times.mean()
    }


def evaluate_model(model_path, model_name, val_datasets, device):
    """Evaluate a temporal model."""
    print("\n" + "="*60)
    print(f"{model_name}")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = create_temporal_classifier(
        pretrained_path=None,
        model_version='v1',
        hidden_dim=512,
        num_layers=2,
        freeze_backbone=True,
        num_beams=64
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded from: {model_path}")
    print(f"Epoch: {checkpoint['epoch']}")
    
    # Create dataloader with custom collate
    combined_dataset = ConcatDataset(val_datasets)
    val_loader = DataLoader(
        combined_dataset, 
        batch_size=8, 
        shuffle=False, 
        num_workers=0,
        collate_fn=custom_collate
    )
    
    # Evaluate accuracy
    print("\n" + "="*60)
    print("Evaluating Accuracy")
    print("="*60)
    
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for img, lid, rad, gps, labels in tqdm(val_loader, desc="Validation"):
            img = img.to(device)
            lid = lid.to(device)
            rad = rad.to(device)
            gps = gps.to(device)
            
            outputs = model(img, lid, rad, gps)
            all_outputs.append(outputs.cpu())
            all_labels.append(labels)
    
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    top1 = top_k_accuracy(all_outputs, all_labels, k=1)
    top3 = top_k_accuracy(all_outputs, all_labels, k=3)
    top5 = top_k_accuracy(all_outputs, all_labels, k=5)
    top10 = top_k_accuracy(all_outputs, all_labels, k=10)
    
    print(f"\nAccuracy Results:")
    print(f"  Top-1:  {top1:.2f}%")
    print(f"  Top-3:  {top3:.2f}%")
    print(f"  Top-5:  {top5:.2f}%")
    print(f"  Top-10: {top10:.2f}%")
    
    # Benchmark inference
    print("\n" + "="*60)
    print("Benchmarking Inference Time")
    print("="*60)
    
    inference_loader = DataLoader(
        combined_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0,
        collate_fn=custom_collate
    )
    inference_stats = benchmark_inference(model, inference_loader, device, num_samples=100)
    
    print(f"\nInference Time (100 samples):")
    print(f"  Mean: {inference_stats['mean']:.2f} ms")
    print(f"  Std:  {inference_stats['std']:.2f} ms")
    print(f"  Min:  {inference_stats['min']:.2f} ms")
    print(f"  Max:  {inference_stats['max']:.2f} ms")
    print(f"  FPS:  {inference_stats['fps']:.1f}")
    
    return {
        'Model': model_name,
        'Checkpoint': model_path,
        'Epoch': checkpoint['epoch'],
        'Top-1': f"{top1:.2f}%",
        'Top-3': f"{top3:.2f}%",
        'Top-5': f"{top5:.2f}%",
        'Top-10': f"{top10:.2f}%",
        'Mean Time (ms)': f"{inference_stats['mean']:.2f}",
        'FPS': f"{inference_stats['fps']:.1f}"
    }


def main():
    print("="*60)
    print("Temporal Classifier Evaluation")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Define transforms (matching training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load validation datasets
    print("\nLoading validation datasets...")
    scenarios = [
        ('c:/Users/suchi/BTP - Semester 7 Sionna/scenario31', 'c:/Users/suchi/BTP - Semester 7 Sionna/scenario31/scenario31_dev.csv'),
        ('c:/Users/suchi/BTP - Semester 7 Sionna/scenario32', 'c:/Users/suchi/BTP - Semester 7 Sionna/scenario32/scenario32_dev.csv'),
        ('c:/Users/suchi/BTP - Semester 7 Sionna/scenario33', 'c:/Users/suchi/BTP - Semester 7 Sionna/scenario33/scenario33_dev.csv')
    ]
    
    val_datasets = []
    for root_dir, csv_file in scenarios:
        print(f"  Loading {root_dir}...")
        dataset = SequenceBeamDataset(
            csv_file=csv_file,
            root_dir=root_dir,
            transform=transform,
            mode='val',
            val_split=0.2,
            seq_len=10
        )
        val_datasets.append(dataset)
        print(f"    Found {len(dataset)} sequences")
    
    total_samples = sum(len(ds) for ds in val_datasets)
    print(f"\nTotal validation sequences: {total_samples}")
    
    # Evaluate both temporal models
    results = []
    
    # 1. Temporal CE Loss
    results.append(evaluate_model(
        'best_temporal_classifier_v1.pth',
        'Temporal Classifier (CE Loss)',
        val_datasets,
        device
    ))
    
    # 2. Temporal Angular Loss
    results.append(evaluate_model(
        'best_temporal_classifier_v1_angular.pth',
        'Temporal Classifier (Angular Loss)',
        val_datasets,
        device
    ))
    
    # Print summary
    print("\n" + "="*60)
    print("Summary - All Temporal Models")
    print("="*60)
    
    for result in results:
        print(f"\n{result['Model']}:")
        for key, value in result.items():
            if key != 'Model':
                print(f"  {key:20s}: {value}")
    
    return results


if __name__ == '__main__':
    main()
