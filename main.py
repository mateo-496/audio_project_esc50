import os
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.models.cnn import CNN
from src.models.train import train_cnn
from src.data.augment import create_augmented_datasets, create_log_mel

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    X_path = "data/preprocessed/X.npy"
    y_path = "data/preprocessed/y.npy"

    if os.path.exists(X_path) and os.path.exists(y_path):
        print("Loading existing processed data...")
        X = np.load(X_path, allow_pickle=True)
        y = np.load(y_path)
    else:
        print("Processing audio data...")
        audio_training_path = "data/audio/0"
        directories = os.listdir(audio_training_path)
        if len(directories) == 1: 
            print("Creating augmented datasets...")
            create_augmented_datasets("data/audio/0", "data/audio") 
        
        print("Creating log-mel spectrograms...")
        X, y = create_log_mel("data/audio", "data/preprocessed")

    print(f"Dataset size: {len(X)} samples, {len(np.unique(y))} classes")

    X_train, X_val, y_train, y_val = train_test_split( 
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    model = CNN(n_classes=len(np.unique(y)))

    best_val_acc = train_cnn(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=100,
        lr=1e-2,
        device=device,
        use_all_patches=True,
        samples_per_epoch_fraction=1/8,
        checkpoint_dir="models/checkpoints",
        save_every_n_epoch=1,
        resume_from=None
    )

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")

    return best_val_acc

def main_resume(checkpoint_dir="models/checkpoints", resume_from="models/checkpoints/latest_checkpoint.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading processed data...")
    X = np.load("data/log_mel/X.npy", allow_pickle=True)
    y = np.load("data/log_mel/y.npy")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    n_classes = len(np.unique(y))
    model = CNN(n_classes=n_classes)
    
    print(f"Resuming from: {resume_from}")
    best_val_acc = train_cnn(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=100,
        lr=0.01,
        device=device,
        use_all_patches=True,
        samples_per_epoch_fraction=1/8,
        checkpoint_dir=checkpoint_dir,
        save_every_n_epoch=1,
        resume_from=resume_from
    )
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
    return best_val_acc



main()