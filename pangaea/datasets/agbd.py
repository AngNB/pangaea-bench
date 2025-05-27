"""
AGBD Dataset for PANGAEA

Adapted from the official AGBD GEDIDataset implementation (see TODO Insert link)
This version is designed to be as close as possible to the proven AGBD code, with only minimal changes for PANGAEA config-driven initialization and output conventions.

- All normalization, band order, and nodata logic is preserved.
- Handles multiple HDF5 files and years, using split files and mapping.
- Patch extraction, normalization, and stacking for all modalities.
- Label extraction and normalization for regression.

DO NOT MODIFY unless you have read both the AGBD and PANGAEA documentation and understand the integration risks.
"""
# TODO clean up imports, remove unused ones
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from os.path import join, exists
import pickle
import pandas as pd
import glob
import math
from typing import Sequence
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# NODATAVALS and REF_BIOMES as in original
NODATAVALS = {
    'S2_bands' : 0, 
    'CH': 255, 
    'ALOS_bands': 0, 
    'DEM': -9999, 
    'LC': 255
}

# TODO: Check where used in original AGBD code
REF_BIOMES = {
    20: 'Shrubs', 
    30: 'Herbaceous vegetation', 
    40: 'Cultivated', 
    90: 'Herbaceous wetland', 
    111: 'Closed-ENL', 
    112: 'Closed-EBL', 
    114: 'Closed-DBL', 
    115: 'Closed-mixed', 
    116: 'Closed-other', 
    121: 'Open-ENL', 
    122: 'Open-EBL', 
    124: 'Open-DBL', 
    125: 'Open-mixed', 
    126: 'Open-other'
}

# --- Begin copied helper functions from AGBD/Models/dataset.py ---
def initialize_index(fnames, mode, chunk_size, path_mapping, path_h5):
    with open(join('/cluster/work/igp_psr/gsialelli/Data/AGB/', 'biomes_splits_to_name.pkl'), 'rb') as f:
        tile_mapping = pickle.load(f)
    idx = {}
    for fname in fnames:
        idx[fname] = {}
        with h5py.File(join(path_h5, fname), 'r') as f:
            all_tiles = list(f.keys())
            tiles = np.intersect1d(all_tiles, tile_mapping[mode])
            for tile in tiles:
                n_patches = len(f[tile]['GEDI']['agbd'])
                idx[fname][tile] = n_patches // chunk_size
    total_length = sum(sum(v for v in d.values()) for d in idx.values())
    return idx, total_length

def find_index_for_chunk(index, n, total_length):
    assert n < total_length, "The chunk index is out of bounds"
    cumulative_sum = 0
    for file_name, file_data in index.items():
        for tile_name, num_rows in file_data.items():
            if cumulative_sum + num_rows > n:
                chunk_within_tile = n - cumulative_sum
                return file_name, tile_name, chunk_within_tile
            cumulative_sum += num_rows

def normalize_data(data, norm_values, norm_strat, nodata_value=None):
    if norm_strat == 'mean_std':
        mean, std = norm_values['mean'], norm_values['std']
        # Handle zero standard deviation case (constant values)
        if std == 0:
            if nodata_value is not None:
                data = np.where(data == nodata_value, 0, 0)
            else:
                data = np.zeros_like(data)
        else:
            if nodata_value is not None:
                data = np.where(data == nodata_value, 0, (data - mean) / std)
            else:
                data = (data - mean) / std
    elif norm_strat == 'pct':
        p1, p99 = norm_values['p1'], norm_values['p99']
        if nodata_value is not None:
            data = np.where(data == nodata_value, 0, (data - p1) / (p99 - p1))
        else:
            data = (data - p1) / (p99 - p1)
        data = np.clip(data, 0, 1)
    elif norm_strat == 'min_max':
        min_val, max_val = norm_values['min'], norm_values['max']
        if nodata_value is not None:
            data = np.where(data == nodata_value, 0, (data - min_val) / (max_val - min_val))
        else:
            data = (data - min_val) / (max_val - min_val)
    else:
        raise ValueError(f'Normalization strategy `{norm_strat}` is not valid.')
    return data

def normalize_bands(bands_data, norm_values, order, norm_strat, nodata_value=None):
    for i, band in enumerate(order):
        band_norm = norm_values[band]
        bands_data[:, :, i] = normalize_data(bands_data[:, :, i], band_norm, norm_strat, nodata_value)
    return bands_data

def encode_lc(lc_data):
    """
    Encode the land cover classes into sin/cosine values and scale the class probabilities to [0,1].
    
    Args:
    - lc_data (np.array): the land cover data with shape (H, W, 2) where [:, :, 0] is class, [:, :, 1] is probability
    
    Returns:
    - lc_cos (np.array): the cosine values of the land cover classes
    - lc_sin (np.array): the sine values of the land cover classes
    - lc_prob (np.array): the land cover class probabilities
    """
    # Get the land cover classes
    lc_map = lc_data[:, :, 0]
    
    # Encode the LC classes with sin/cosine values and scale the data to [0,1]
    lc_cos = np.where(lc_map == NODATAVALS['LC'], 0, (np.cos(2 * np.pi * lc_map / 100) + 1) / 2)
    lc_sin = np.where(lc_map == NODATAVALS['LC'], 0, (np.sin(2 * np.pi * lc_map / 100) + 1) / 2)
    
    # Scale the class probabilities to [0,1]
    lc_prob = lc_data[:, :, 1]
    lc_prob = np.where(lc_prob == NODATAVALS['LC'], 0, lc_prob / 100)
    
    return lc_cos, lc_sin, lc_prob
# --- End copied helper functions ---

class AGBD(Dataset):
    """
    PANGAEA-compatible dataset for AGBD, adapted from GEDIDataset.
    Uses only biomes_splits_to_name.pkl for split logic, matching the original AGBD code.
    Reads config fields as in agbd.yaml, builds index from tile mapping, and loads patches from multiple HDF5 files.
    """
    def __init__(self, split, dataset_name, root_path, hdf5_dir, hdf5_pattern, split_files, img_size, multi_modal, multi_temporal, bands, data_mean, data_std, data_min, data_max, label_group, label_name, ignore_index=-1, **kwargs):
        # Parse config and set up paths
        self.mode = split
        self.root_path = root_path
        self.h5_path = hdf5_dir
        self.chunk_size = 1  # AGBD is patch-based, so chunk_size=1
        self.img_size = img_size
        self.patch_size = (img_size, img_size)
        self.bands = bands['optical']  # S2 optical bands
        self.alos_bands = bands['sar']  # SAR bands
        self.auxiliary_bands = bands.get('auxiliary', [])  # Auxiliary features (DEM, CH, LC, SCL)
        
        # Set up normalization values for all modalities
        self.norm_values = {'S2_bands': {}, 'ALOS_bands': {}, 'auxiliary': {}}
        # Set up normalization values for all modalities
        self.norm_values = {'S2_bands': {}, 'ALOS_bands': {}, 'auxiliary': {}}
        for i, b in enumerate(self.bands):
            self.norm_values['S2_bands'][b] = {'mean': data_mean['optical'][i], 'std': data_std['optical'][i], 'min': data_min['optical'][i], 'max': data_max['optical'][i]}
        for i, b in enumerate(self.alos_bands):
            self.norm_values['ALOS_bands'][b] = {'mean': data_mean['sar'][i], 'std': data_std['sar'][i], 'min': data_min['sar'][i], 'max': data_max['sar'][i]}
        for i, b in enumerate(self.auxiliary_bands):
            self.norm_values['auxiliary'][b] = {'mean': data_mean['auxiliary'][i], 'std': data_std['auxiliary'][i], 'min': data_min['auxiliary'][i], 'max': data_max['auxiliary'][i]}
        self.label_group = label_group
        self.label_name = label_name
        self.split = split  # For PANGAEA compatibility (used by subset_sampler etc.)
        # For PANGAEA compatibility: evaluator expects a `classes` attribute (see BioMassters etc.)
        # This is a regression dataset, so we set classes to ['regression']
        self.classes = ['regression']
        # For PANGAEA compatibility: evaluator expects an`ignore_index` attribute (see BioMassters etc.)
        self.ignore_index = ignore_index
        # For PANGAEA compatibility: evaluator expects a `num_classes` attribute (see BioMassters etc.)
        # This is a regression dataset, so set num_classes to 1
        self.num_classes = 1
        # Use only biomes_splits_to_name.pkl for split logic
        mapping_path = join('/cluster/work/igp_psr/gsialelli/Data/AGB/', 'biomes_splits_to_name.pkl')
        with open(mapping_path, 'rb') as f:
            tile_mapping = pickle.load(f)
        self.split_tiles = set(tile_mapping[split])
        # List all HDF5 files in hdf5_dir matching hdf5_pattern
        h5_files = glob.glob(join(hdf5_dir, hdf5_pattern))
        self.fnames = [f.split('/')[-1] for f in h5_files]
        self.fnames = sorted(list(set(self.fnames)))
        # Build index as in initialize_index, but only for tiles in split_tiles
        self.index, self.length = initialize_index(self.fnames, self.mode, self.chunk_size, root_path, hdf5_dir)
        self.handles = {fname: h5py.File(join(hdf5_dir, fname), 'r') for fname in self.index.keys()}
        self.s2_order = self.bands
        self.alos_order = self.alos_bands
        self.norm_strat = 'mean_std'  # TODO: make configurable? idk

    def __len__(self):
        return self.length

    def __getitem__(self, n):
        file_name, tile_name, idx = find_index_for_chunk(self.index, n, self.length)
        f = self.handles[file_name]
        
        # Initialize data list for auxiliary features
        auxiliary_data = []
        
        # S2 bands
        s2_bands = f[tile_name]['S2_bands'][idx].astype(np.float32)

        # --- PATCH: Convert S2 bands to reflectance as in original AGBD code ---
        if 'Sentinel_metadata' in f[tile_name] and 'S2_boa_offset' in f[tile_name]['Sentinel_metadata']:
            s2_boa_offset = f[tile_name]['Sentinel_metadata']['S2_boa_offset'][idx]
        else:
            s2_boa_offset = 0

        # [PATCH: S2_BANDS/BOA_OFFSET DTYPE SAFETY + DIAGNOSTIC]
        s2_bands = s2_bands.astype(np.float32)
        s2_boa_offset = np.array(s2_boa_offset).astype(np.float32)
        # [END PATCH]

        s2_bands = (s2_bands - s2_boa_offset * 1000) / 10000
        s2_bands[s2_bands < 0] = 0
        s2_bands[s2_bands == 0] = 0
        # --- END PATCH ---

        s2_bands = normalize_bands(s2_bands, self.norm_values['S2_bands'], self.s2_order, self.norm_strat, NODATAVALS['S2_bands'])
        
        # ALOS bands
        alos_bands = f[tile_name]['ALOS_bands'][idx].astype(np.float32)

        # Mask zeros and negatives before log10 to avoid divide by zero (caused crash)
        alos_bands = np.where(alos_bands <= 0, 1e-6, alos_bands)
        alos_bands = np.where(alos_bands == NODATAVALS['ALOS_bands'], -9999.0, 10 * np.log10(np.power(alos_bands, 2)) - 83.0)
        alos_bands = normalize_bands(alos_bands, self.norm_values['ALOS_bands'], self.alos_order, self.norm_strat, -9999.0)
        
        # Load auxiliary features if available
        if len(self.auxiliary_bands) > 0:
            # DEM data
            if 'DEM' in self.auxiliary_bands:
                dem = f[tile_name]['DEM'][idx].astype(np.float32)
                dem = normalize_data(dem, self.norm_values['auxiliary']['DEM'], self.norm_strat, NODATAVALS['DEM'])
                auxiliary_data.append(dem[..., np.newaxis])
            
            # Canopy Height data
            if 'CH_ch' in self.auxiliary_bands:
                ch = f[tile_name]['CH']['ch'][idx].astype(np.float32)
                ch = normalize_data(ch, self.norm_values['auxiliary']['CH_ch'], self.norm_strat, NODATAVALS['CH'])
                auxiliary_data.append(ch[..., np.newaxis])
            
            if 'CH_std' in self.auxiliary_bands:
                ch_std = f[tile_name]['CH']['std'][idx].astype(np.float32)
                ch_std = normalize_data(ch_std, self.norm_values['auxiliary']['CH_std'], self.norm_strat, NODATAVALS['CH'])
                auxiliary_data.append(ch_std[..., np.newaxis])
            
            # Land Cover data with sin/cosine encoding
            if 'LC_1' in self.auxiliary_bands and 'LC_2' in self.auxiliary_bands:
                lc = f[tile_name]['LC'][idx].astype(np.float32)
                lc_cos, lc_sin, lc_prob = encode_lc(lc)
                auxiliary_data.append(lc_cos[..., np.newaxis])
                auxiliary_data.append(lc_sin[..., np.newaxis])
            
            # Scene Classification Layer (S2_SCL)
            if 'S2_SCL' in self.auxiliary_bands:
                if 'S2_SCL' in f[tile_name]:
                    scl = f[tile_name]['S2_SCL'][idx].astype(np.float32)
                    scl = normalize_data(scl, self.norm_values['auxiliary']['S2_SCL'], self.norm_strat, None)
                    auxiliary_data.append(scl[..., np.newaxis])
                else:
                    # If SCL not available as separate dataset, try to extract from S2_bands
                    # This is a fallback - in practice, SCL should be available separately
                    scl = np.zeros(self.patch_size, dtype=np.float32)
                    auxiliary_data.append(scl[..., np.newaxis])
        
        # Target
        agbd = f[tile_name]['GEDI']['agbd'][idx]

        # --- PATCH: Normalize target to match original AGBD code (images all black otherwise! check this though TODO) ---
        target_mean = getattr(self, 'data_mean', {}).get('target', 66.97265625)
        target_std = getattr(self, 'data_std', {}).get('target', 98.66587829589844)
        agbd = (agbd - target_mean) / target_std
        agbd = torch.full(self.patch_size, float(agbd), dtype=torch.float32)
        
        # Build image dictionary - stack and return in PANGAEA format (add time dimension T=1)
        image = {
            'optical': torch.from_numpy(s2_bands).permute(2, 0, 1).unsqueeze(1).float(),
            'sar': torch.from_numpy(alos_bands).permute(2, 0, 1).unsqueeze(1).float()
        }
        
        # Add auxiliary features if available
        if auxiliary_data:
            auxiliary_stacked = np.concatenate(auxiliary_data, axis=-1)
            image['auxiliary'] = torch.from_numpy(auxiliary_stacked).permute(2, 0, 1).unsqueeze(1).float()
        
        # --- PATCH COORDINATE/INDEX METADATA ---
        metadata = {'tile_name': tile_name, 'patch_index': idx}
        # Try to get lat/lon if available
        try:
            lat_offset = f[tile_name]['GEDI']['lat_offset'][idx]
            lat_decimal = f[tile_name]['GEDI']['lat_decimal'][idx]
            lon_offset = f[tile_name]['GEDI']['lon_offset'][idx]
            lon_decimal = f[tile_name]['GEDI']['lon_decimal'][idx]
            lat = float(np.sign(lat_decimal) * (np.abs(lat_decimal) + lat_offset))
            lon = float(np.sign(lon_decimal) * (np.abs(lon_decimal) + lon_offset))
            metadata['lat'] = lat
            metadata['lon'] = lon
        except Exception:
            pass
        # --- END PATCH COORDINATE/INDEX METADATA ---
        
        return {'image': image, 'target': agbd, 'metadata': metadata}