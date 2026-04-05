"""
Radar Signal Processing Module for FMCW Radar Data

This module provides functions to compute Range-Doppler and Range-Angle maps
from raw complex radar data.

Input Data Format:
- Shape: (Rx_Antennas, Range_Samples, Slow_Time/Chirps) = (4, 256, 250)
- Dtype: complex64 (raw IF signal after ADC)
"""

import numpy as np
import torch

def compute_range_doppler(data: np.ndarray) -> np.ndarray:
    """
    Compute Range-Doppler map from raw radar data.
    
    Steps:
    1. Apply FFT along Range axis (axis=1) -> Range bins
    2. Apply FFT along Slow-Time axis (axis=2) -> Doppler bins
    3. Take magnitude and sum across antennas
    
    Args:
        data: Complex array of shape (Rx, Range_Samples, Slow_Time)
    
    Returns:
        Range-Doppler map of shape (Range_Bins, Doppler_Bins)
    """
    # Apply Window functions to reduce spectral leakage
    range_window = np.hanning(data.shape[1])
    doppler_window = np.hanning(data.shape[2])
    
    # Apply windows
    data_windowed = data * range_window[np.newaxis, :, np.newaxis]
    data_windowed = data_windowed * doppler_window[np.newaxis, np.newaxis, :]
    
    # 2D FFT: Range then Doppler
    # Range FFT
    range_fft = np.fft.fft(data_windowed, axis=1)
    # Doppler FFT
    range_doppler = np.fft.fft(range_fft, axis=2)
    
    # Shift zero-frequency to center (for Doppler)
    range_doppler = np.fft.fftshift(range_doppler, axes=2)
    
    # Take magnitude
    rd_mag = np.abs(range_doppler)
    
    # Sum across antennas for a combined view
    rd_map = np.sum(rd_mag, axis=0)  # (Range, Doppler)
    
    # Convert to dB scale for better dynamic range
    rd_map = 20 * np.log10(rd_map + 1e-10)
    
    # Normalize to [0, 1]
    rd_map = (rd_map - rd_map.min()) / (rd_map.max() - rd_map.min() + 1e-10)
    
    return rd_map.astype(np.float32)


def compute_range_angle(data: np.ndarray) -> np.ndarray:
    """
    Compute Range-Angle map from raw radar data.
    
    Steps:
    1. Apply FFT along Range axis (axis=1) -> Range bins
    2. Apply FFT along Antenna axis (axis=0) -> Angle bins (Beamforming)
    3. Sum or select across Doppler (axis=2) to compress
    
    Args:
        data: Complex array of shape (Rx, Range_Samples, Slow_Time)
    
    Returns:
        Range-Angle map of shape (Range_Bins, Angle_Bins)
    """
    # Range FFT
    range_fft = np.fft.fft(data, axis=1)
    
    # Average over slow-time (Doppler) to simplify - take max for better targets
    # Or, sum over Doppler for a static scene representation
    range_data = np.mean(range_fft, axis=2)  # (Rx, Range)
    
    # Zero-pad antenna axis for better angle resolution
    # E.g., pad from 4 to 64 virtual antennas
    num_angle_bins = 64
    padded = np.zeros((num_angle_bins, range_data.shape[1]), dtype=np.complex64)
    padded[:range_data.shape[0], :] = range_data
    
    # Angle FFT (across antenna dimension)
    angle_fft = np.fft.fft(padded, axis=0)
    angle_fft = np.fft.fftshift(angle_fft, axes=0)
    
    # Take magnitude
    ra_map = np.abs(angle_fft)  # (Angle_Bins, Range)
    
    # Transpose to (Range, Angle) for consistency
    ra_map = ra_map.T
    
    # Convert to dB
    ra_map = 20 * np.log10(ra_map + 1e-10)
    
    # Normalize
    ra_map = (ra_map - ra_map.min()) / (ra_map.max() - ra_map.min() + 1e-10)
    
    return ra_map.astype(np.float32)


def process_radar_data(path: str, target_size=(224, 224)) -> torch.Tensor:
    """
    Load radar data and compute Range-Doppler and Range-Angle maps.
    
    Args:
        path: Path to .npy file containing complex radar data
        target_size: Output spatial dimensions (H, W)
    
    Returns:
        Tensor of shape (2, H, W) containing [Range-Doppler, Range-Angle]
    """
    try:
        data = np.load(path)
        
        # Compute maps
        rd_map = compute_range_doppler(data)  # (Range, Doppler)
        ra_map = compute_range_angle(data)    # (Range, Angle)
        
        # Resize individual maps to target size
        # Add simpler helper to resize: (H, W) -> (1, 1, H, W) -> interp -> (H_new, W_new)
        def resize_map(m, size):
            t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0)
            t = torch.nn.functional.interpolate(t, size=size, mode='bilinear', align_corners=False)
            return t.squeeze(0).squeeze(0)

        rd_tensor = resize_map(rd_map, target_size)
        ra_tensor = resize_map(ra_map, target_size)
        
        # Stack as channels
        tensor = torch.stack([rd_tensor, ra_tensor], axis=0)  # (2, H, W)
        
        return tensor
        
    except Exception as e:
        print(f"Error processing radar {path}: {e}")
        return torch.zeros((2, target_size[0], target_size[1]), dtype=torch.float32)


if __name__ == "__main__":
    # Test
    import matplotlib.pyplot as plt
    
    test_path = r'c:/Users/suchi/BTP_Semester8/DeepSense6G_TII/Dataset/MultiModeBeamforming/Multi_Modal/scenario31/unit1/radar_data/radar_data_1000.npy'
    
    data = np.load(test_path)
    print(f"Raw data shape: {data.shape}, dtype: {data.dtype}")
    
    rd = compute_range_doppler(data)
    ra = compute_range_angle(data)
    
    print(f"Range-Doppler shape: {rd.shape}")
    print(f"Range-Angle shape: {ra.shape}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(rd, aspect='auto', cmap='jet')
    axes[0].set_title("Range-Doppler Map")
    axes[0].set_xlabel("Doppler Bin")
    axes[0].set_ylabel("Range Bin")
    
    axes[1].imshow(ra, aspect='auto', cmap='jet')
    axes[1].set_title("Range-Angle Map")
    axes[1].set_xlabel("Angle Bin")
    axes[1].set_ylabel("Range Bin")
    
    plt.tight_layout()
    plt.savefig("radar_processing_test.png")
    print("Saved radar_processing_test.png")
