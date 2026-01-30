import numpy as np
import torch

cnn_input_length = 128

def predict_with_overlapping_patches(model, spectrogram, patch_length=cnn_input_length, hop=1, batch_size=100, device="cuda"):
    model.eval()

    n_frames, n_mels = spectrogram.shape

    if n_frames < patch_length:
        pad = patch_length - n_frames
        spectrogram = np.pad(spectrogram, ((0, pad), (0, 0)), mode='constant')
        n_frames = patch_length

    patches = []
    for start in range(0, n_frames - patch_length + 1, hop):
        patch = spectrogram[start:start + patch_length]
        patch = patch[np.newaxis, np.newaxis, :, :]
        patches.append(patch)

    patches = np.concatenate(patches, axis=0)
    patches = torch.tensor(patches, dtype=torch.float32).to(device)

    all_outputs = []
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i + batch_size]
            outputs = model(batch)
            all_outputs.append(outputs)
    
    all_outputs = torch.cat(all_outputs, dim=0)

    mean_activations = all_outputs.mean(dim=0)
    predicted_class = mean_activations.argmax().item()

    return predicted_class
