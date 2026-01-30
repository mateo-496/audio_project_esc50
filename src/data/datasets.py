import torch
import numpy as np
from torch.utils.data import Dataset

cnn_input_length = 128

class SpectrogramDataset(Dataset):
    def __init__(self, spectrograms, labels, patch_length=cnn_input_length, mode='train'):
        self.spectrograms = spectrograms
        self.labels = labels
        self.patch_length = patch_length
        self.mode = mode
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spec = self.spectrograms[idx]
        label = self.labels[idx]

        if self.mode == 'train':
            n_frames = spec.shape[0]

            if n_frames >= self.patch_length:
                start = np.random.randint(0, n_frames - self.patch_length + 1)
                patch = spec[start:start + self.patch_length]
            else:
                pad = self.patch_length - n_frames
                patch = np.pad(spec, ((0, pad), (0, 0)), mode='constant')
            
            patch = patch[np.newaxis, :, :]
            return torch.tensor(patch, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        
        else:
            return spec, label
            
class FullTFPatchesDataset(Dataset):
    def __init__(self, spectrograms, labels, patch_length=128):
        self.patch_length = patch_length
        self.patch_indices = []

        for spec_idx, spec in enumerate(spectrograms):
            n_frames = spec.shape[0]
            label = labels[spec_idx]

            if n_frames >= patch_length:
                for start_frame in range(n_frames - patch_length + 1):
                    self.patch_indices.append((spec_idx, start_frame, label))
            else:
                self.patch_indices.append((spec_idx, 0, label))
            
        self.spectrograms = spectrograms
    
    def __len__(self):
        return len(self.patch_indices)
    
    def __getitem__(self, idx):
        spec_idx, start_frame, label = self.patch_indices[idx]
        spec = self.spectrograms[spec_idx]
        
        n_frames = spec.shape[0]
        
        if n_frames >= self.patch_length:
            patch = spec[start_frame:start_frame + self.patch_length]
        else:
            pad = self.patch_length - n_frames
            patch = np.pad(spec, ((0, pad), (0, 0)), mode='constant')
        
        patch = patch[np.newaxis, :, :]
        
        return torch.tensor(patch, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class RandomPatchDataset(Dataset):
    def __init__(self, spectrograms, labels, patch_length=128):
        self.spectrograms = spectrograms
        self.labels = labels
        self.patch_length = patch_length
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        spec = self.spectrograms[idx]
        label = self.labels[idx]
        n_frames = spec.shape[0]
        
        if n_frames >= self.patch_length:
            start = np.random.randint(0, n_frames - self.patch_length + 1)
            patch = spec[start:start + self.patch_length]
        else:
            pad = self.patch_length - n_frames
            patch = np.pad(spec, ((0, pad), (0, 0)), mode='constant')
        
        patch = patch[np.newaxis, :, :]
        return torch.tensor(patch, dtype=torch.float32), torch.tensor(label, dtype=torch.long)