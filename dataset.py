import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
from plyfile import PlyData
from functools import lru_cache

class BeamDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, mode='train', val_split=0.2):
        self.root_dir = root_dir
        self.transform = transform
        self.df = pd.read_csv(csv_file)
        
        # Filter valid files if necessary or just use all
        # self.df = self.df.dropna(subset=['unit1_beam'])
        
        # Split
        dataset_size = len(self.df)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        
        # Sequential split (no shuffling)
        if mode == 'train':
            self.indices = indices[split:]
        else:
            self.indices = indices[:split]
            
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        row = self.df.iloc[real_idx]
        
        # 1. Image (RGB)
        img_path = os.path.join(self.root_dir, row['unit1_rgb'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        else:
            # Basic transform if none provided
            image = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float() / 255.0

        # 2. LiDAR (PLY to Grid)
        lidar_path = os.path.join(self.root_dir, row['unit1_lidar'])
        lidar_grid = self.process_lidar(lidar_path)

        # 3. Radar (NPY)
        radar_path = os.path.join(self.root_dir, row['unit1_radar'])
        radar_tensor = self.process_radar(radar_path)

        # 4. GPS (from Unit 2)
        gps_path = os.path.join(self.root_dir, row['unit2_loc'])
        gps_vec = self.process_gps(gps_path)

        # Label (Beam Index 1-64 -> 0-63)
        # Assuming values in CSV are 1-based integers.
        label = int(row['unit1_beam']) - 1 # Adjust if labels are already 0-based
        if label < 0: label = 0 # Safety
        if label > 63: label = 63

        return image, lidar_grid, radar_tensor, gps_vec, label

    def process_lidar(self, path):
        # Optimizaed: Check for preprocessed .npy first
        npy_path = path.replace('.ply', '.npy')
        
        # DEBUG: Check if we are finding the file
        # print(f"Checking {npy_path} -> Exists: {os.path.exists(npy_path)}")
        
        if os.path.exists(npy_path):
            try:
                grid = np.load(npy_path)
                return torch.from_numpy(grid)
            except Exception as e:
                print(f"Failed to load NPY {npy_path}: {e}")
        else:
            print(f"ERROR: Missing preprocessed NPY file {npy_path}. Run preprocess_lidar.py first!")
            
        return torch.zeros((1, 224, 224), dtype=torch.float32)

    def process_radar(self, path):
        """
        Process radar data using FFT-based Range-Doppler and Range-Angle maps.
        Uses the radar_processing module with caching for faster loading.
        """
        # Check for preprocessed radar cache
        cache_path = path.replace('.npy', '_processed.pt')
        
        if os.path.exists(cache_path):
            try:
                return torch.load(cache_path)
            except Exception as e:
                print(f"Failed to load cached radar {cache_path}: {e}")
        
        # Process and cache
        try:
            from radar_processing import process_radar_data
            radar_tensor = process_radar_data(path, target_size=(224, 224))
            # Save cache for next time
            try:
                torch.save(radar_tensor, cache_path)
            except:
                pass  # Don't fail if caching fails
            return radar_tensor
        except Exception as e:
            print(f"Error processing radar {path}: {e}")
            return torch.zeros((2, 224, 224), dtype=torch.float32)

    def process_gps(self, path):
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
                lat = float(lines[0].strip())
                lon = float(lines[1].strip())
            
            # Normalize approx based on dataset stats or global coords
            # Simple min-max or zero-center if we knew the bounds.
            # For now, just raw values or scaled slightly.
            # Arizona coords approx: 33.4, -111.9
            
            lat_norm = (lat - 33.0) 
            lon_norm = (lon + 111.0) 
            
            return torch.tensor([lat_norm, lon_norm], dtype=torch.float32)
        except Exception as e:
            return torch.tensor([0.0, 0.0], dtype=torch.float32)

class SequenceBeamDataset(Dataset):
    """
    Loads a sequence of frames (seq_len) for temporal models.
    """
    def __init__(self, csv_file, root_dir, transform=None, mode='train', val_split=0.2, seq_len=10):
        self.root_dir = root_dir
        self.transform = transform
        self.df = pd.read_csv(csv_file)
        self.seq_len = seq_len
        
        # Sort by seq_index and time_stamp to ensure order
        if 'seq_index' in self.df.columns and 'time_stamp' in self.df.columns:
            self.df = self.df.sort_values(by=['seq_index', 'time_stamp'])
        
        # Create sequences
        self.sequences = []
        
        current_seq_idx = -1
        current_buffer_indices = []
        
        # Iterate and build sequences
        # Only sequences belonging to the same 'seq_index' are valid.
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            s_idx = row.get('seq_index', 0) # Default to 0 if missing
            
            if s_idx != current_seq_idx:
                # New sequence group
                current_seq_idx = s_idx
                current_buffer_indices = []
            
            current_buffer_indices.append(idx)
            
            if len(current_buffer_indices) >= seq_len:
                # valid window
                # We store the INDICES of the window [i-seq_len+1, i]
                self.sequences.append(list(current_buffer_indices[-seq_len:]))
                
        # Split sequences (not individual frames)
        dataset_size = len(self.sequences)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        
        # Sequential split (no shuffling)
        if mode == 'train':
            self.indices = indices[split:]
        else:
            self.indices = indices[:split]
            
        print(f"Dataset({mode}): Found {len(self.indices)} sequences of length {seq_len}.")

    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        # returns tensors of shape (T, C, H, W)
        seq_indices = self.sequences[self.indices[idx]]
        
        imgs_list = []
        lids_list = []
        rads_list = []
        gpss_list = []
        power_list = []
        
        # Target label is usually the label of the LAST frame in the sequence
        last_idx = seq_indices[-1]
        last_row = self.df.iloc[last_idx]
        label = int(last_row['unit1_beam']) - 1
        if label < 0: label = 0
        if label > 63: label = 63
        
        # Helper reuse (we can instantiate a helper BeamDataset or just copy methods)
        # To avoid code dupe, we can call static methods or mixins, but here we'll just instantiate one helper
        # Or better, just refactor process_* to be standalone or use the logic here.
        # For simplicity/speed, I'll copy the logic or use a helper instance.
        
        # Optimization: We are loading T files. 
        for i in seq_indices:
            row = self.df.iloc[i]
            
            # Image
            img_path = os.path.join(self.root_dir, row['unit1_rgb'])
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            else:
                image = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float() / 255.0
            imgs_list.append(image)
            
            # Lidar
            lidar_path = os.path.join(self.root_dir, row['unit1_lidar'])
            # Re-implement simple process_lidar logic here or refactor. 
            # I will assume we can access BeamDataset code. 
            # Let's dynamically add the method or copy it.
            # Copying core logic for safety.
            npy_path = lidar_path.replace('.ply', '.npy')
            if os.path.exists(npy_path):
                 grid = torch.from_numpy(np.load(npy_path))
            else:
                 # Fallback empty
                 grid = torch.zeros((1, 224, 224), dtype=torch.float32)
            lids_list.append(grid)

            # Radar (with caching)
            radar_path = os.path.join(self.root_dir, row['unit1_radar'])
            cache_path = radar_path.replace('.npy', '_processed.pt')
            
            if os.path.exists(cache_path):
                try:
                    data = torch.load(cache_path)
                except:
                    try:
                        from radar_processing import process_radar_data
                        data = process_radar_data(radar_path, target_size=(224, 224))
                        torch.save(data, cache_path)
                    except Exception as e:
                        data = torch.zeros((2, 224, 224), dtype=torch.float32)
            else:
                try:
                    from radar_processing import process_radar_data
                    data = process_radar_data(radar_path, target_size=(224, 224))
                    torch.save(data, cache_path)
                except Exception as e:
                    data = torch.zeros((2, 224, 224), dtype=torch.float32)
            rads_list.append(data)
            
            # GPS
            gps_path = os.path.join(self.root_dir, row['unit2_loc'])
            try:
                with open(gps_path, 'r') as f:
                    lines = f.readlines()
                    lat = float(lines[0].strip())
                    lon = float(lines[1].strip())
                lat_norm = (lat - 33.0) 
                lon_norm = (lon + 111.0) 
                gps_vec = torch.tensor([lat_norm, lon_norm], dtype=torch.float32)
            except:
                gps_vec = torch.tensor([0.0, 0.0], dtype=torch.float32)
            gpss_list.append(gps_vec)
            
            # Power Vector for each frame in sequence
            pwr_path = os.path.join(self.root_dir, row['unit1_pwr_60ghz'])
            try:
                with open(pwr_path, 'r') as f:
                    pwr_lines = f.readlines()
                    pwr_vec = []
                    for l in pwr_lines[:64]:
                        val = float(l.strip())
                        pwr_vec.append(val if not np.isnan(val) else 0.0)
                    if len(pwr_vec) < 64:
                        pwr_vec += [0.0] * (64 - len(pwr_vec))
                    power_vec = torch.tensor(pwr_vec, dtype=torch.float32)
            except Exception:
                power_vec = torch.zeros(64, dtype=torch.float32)
            power_list.append(power_vec)

        # Stack
        imgs = torch.stack(imgs_list) # (T, 3, 224, 224)
        lids = torch.stack(lids_list)
        rads = torch.stack(rads_list)
        gpss = torch.stack(gpss_list)
        powers = torch.stack(power_list) # (T, 64)
        
        return imgs, lids, rads, gpss, label, powers
