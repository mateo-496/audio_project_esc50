import numpy as np
import torch
import torch.nn as nn
import essentia.standard as es 
import argparse
import os
import sys

from src.models.cnn import CNN
from src.data.augment import data_treatment
from src.config.config import sample_rate, parameters, cnn_input_length, esc50_labels

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

def predict_top_k(model, spectrogram, patch_length=cnn_input_length, hop=1, batch_size=100, device="cpu", top_k=5):  
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
    mean_logits = all_outputs.mean(dim=0)
    probabilities = torch.nn.functional.softmax(mean_logits, dim=0)

    top_probs, top_indices = torch.topk(probabilities, min(top_k, 50))
    top_probs = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()

    return top_probs, top_indices

def predict_file(model, audio_file, device="cpu", top_k=5):
    parameters = {
        "n_bands"  : 128,
        "n_mels" : 128,
        "frame_size" : 1024,
        "hop_size": 1024,
        "sample_rate": sample_rate,
        "fft_size": 8192,
    }
    spectrogram, label = data_treatment(audio_file, **parameters)

    spectrogram = np.array(spectrogram)
    print(f"  Spectrogram shape from data_treatment: {spectrogram.shape}")

    spectrogram = spectrogram.squeeze()


    predicted_class = predict_with_overlapping_patches(
        model, spectrogram, patch_length=128, hop=1, batch_size=100, device=device
    )
    top_probs, top_indices = predict_top_k(
        model, spectrogram, patch_length=128, hop=1, batch_size=100, device=device, top_k=top_k
    )

    return predicted_class, label, top_probs, top_indices

def load_model(model_path, device='cpu'):
    print(f"Loading model from {model_path}...")
    
    model = CNN(n_classes=50)
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'best_val_acc' in checkpoint:
                print(f"Model validation accuracy: {checkpoint['best_val_acc']:.4f}")
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print("Model loaded successfully!\n")
    return model



def main():
    parser = argparse.ArgumentParser(
        description='Predict environmental sound class using trained ESC-50 model'
    )
    parser.add_argument(
        'audio_file',
        type=str,
        help='Path to .wav file to classify'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='best_model.pt',
        help='Path to trained model checkpoint (default: best_model.pt)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top predictions to show (default: 5)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Load model
    try:
        model = load_model(args.model, device=args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Predict
    try:
        predicted_class, label, top_probs, top_indices = predict_file(
            model, args.audio_file, device=args.device, top_k=args.top_k
        )
        
        # Display results
        print("\n" + "=" * 60)
        print(f"Top {args.top_k} Predictions:")
        print("=" * 60)
        
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            class_name = esc50_labels[idx]
            marker = "★" if idx == predicted_class else " "
            print(f"{marker} {i+1}. {class_name:20s} - {prob*100:6.2f}%")
        
        print("=" * 60)
        print(f"\n✓ Predicted class: {esc50_labels[predicted_class]}")
        
    except Exception as e:
        print(f"\nError during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()