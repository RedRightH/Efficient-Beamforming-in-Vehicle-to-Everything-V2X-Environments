"""
Evaluate Baseline BeamTransFuser Model
"""
import torch
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

from model import BeamTransFuser
from dataset import BeamDataset


def top_k_accuracy(outputs, targets, k=1):
    """Calculate top-k accuracy."""
    _, pred = outputs.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    correct_k = correct[:k].reshape(-1).float().sum(0)
    return correct_k.mul_(100.0 / targets.size(0)).item()


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


def main():
    print("="*60)
    print("Baseline BeamTransFuser Evaluation")
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
        ('c:/Users/suchi/BTP - Semester 7 Sionna/scenario31', 'scenario31_dev.csv'),
        ('c:/Users/suchi/BTP - Semester 7 Sionna/scenario32', 'scenario32_dev.csv'),
        ('c:/Users/suchi/BTP - Semester 7 Sionna/scenario33', 'scenario33_dev.csv')
    ]
    
    val_datasets = []
    for scenario_dir, csv_name in scenarios:
        csv_file = f"{scenario_dir}/{csv_name}"
        print(f"  Loading {scenario_dir}...")
        dataset = BeamDataset(
            csv_file=csv_file,
            root_dir=scenario_dir,
            transform=transform,
            mode='val',
            val_split=0.2
        )
        val_datasets.append(dataset)
        print(f"    Found {len(dataset)} samples")
    
    # Combine datasets
    from torch.utils.data import ConcatDataset
    combined_dataset = ConcatDataset(val_datasets)
    print(f"\nTotal validation samples: {len(combined_dataset)}")
    
    # Load model
    print("\nLoading model...")
    model = BeamTransFuser(num_beams=64, pruning_ratio=0.25).to(device)
    checkpoint = torch.load('retrained_best_beam_model.pth', map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully")
    
    # Evaluate accuracy
    print("\n" + "="*60)
    print("Evaluating Accuracy")
    print("="*60)
    
    val_loader = DataLoader(combined_dataset, batch_size=32, shuffle=False, num_workers=0)
    
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
    
    inference_loader = DataLoader(combined_dataset, batch_size=1, shuffle=False, num_workers=0)
    inference_stats = benchmark_inference(model, inference_loader, device, num_samples=100)
    
    print(f"\nInference Time (100 samples):")
    print(f"  Mean: {inference_stats['mean']:.2f} ms")
    print(f"  Std:  {inference_stats['std']:.2f} ms")
    print(f"  Min:  {inference_stats['min']:.2f} ms")
    print(f"  Max:  {inference_stats['max']:.2f} ms")
    print(f"  FPS:  {inference_stats['fps']:.1f}")
    
    # Save results
    results = {
        'Model': 'Baseline BeamTransFuser',
        'Checkpoint': 'retrained_best_beam_model.pth',
        'Top-1': f"{top1:.2f}%",
        'Top-3': f"{top3:.2f}%",
        'Top-5': f"{top5:.2f}%",
        'Top-10': f"{top10:.2f}%",
        'Mean Time (ms)': f"{inference_stats['mean']:.2f}",
        'FPS': f"{inference_stats['fps']:.1f}"
    }
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for key, value in results.items():
        print(f"{key:20s}: {value}")
    
    return results


if __name__ == '__main__':
    main()
