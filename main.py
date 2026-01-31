import os
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split

from src.data.donwload import download_clean
from src.data.augment import create_augmented_datasets, create_log_mel, data_treatment_testing
from src.models.cnn import CNN
from src.models.train import train_cnn
from src.models.predict import predict_with_overlapping_patches, predict_top_k, predict_file, load_model
from config.config import sample_rate, cnn_input_length, esc50_labels

def main():
    parser = argparse.ArgumentParser(
        description="ESC50 Audio Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    subparsers.required = True

    download_parser = subparser.add_parser('download', help='Download ESC50 dataset')
    download_paerser.set_defaults(func=cmd_download)

    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess audio dataset')
    preprocess_parser.add_argument('--input-dir', type=str, required=True, help='Input audio directory')
    preprocess_parser.add_argument('--output-dir', type=str, required=True, help='Output directory for preprocessed data')
    preprocess_parser.add_argument('--augment', action='store_true', help='Create augmented datasets')
    preprocess_parser.set_defaults(func=cmd_preprocess)

    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--audio-dir', type=str, help='Path to training audio directory')
    train_parser.add_argument('output-dir', type=str, help='Path to save preprocessed data')
    train_parser.add_argument('--X-path', type=str, help='Path to preprocessed X.npy')
    train_parser.add_argument('--y-path', type=str, help='Path to preprocessed y.npy')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    train_parser.add_argument('--batch-size', type=int, default=100, help='Batch size (default: 100)')
    train_parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    train_parser.add_argument('--sample-fraction', type=float, default=1/8, help='Fraction of samples per epoch (default: 1/8)')
    train_parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints', help='Checkpoint directory')
    train_parser.add_argument('--save-every', type=int, default=1, help='Save checkpoint every N epochs')
    train_parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    train_parser.set_defaults(func=cmd_train)

    resume_parser = subparsers.add_parser('resume', help='Resume training from checkpoint')
    resume_parser.add_argument('--resume-from', type=str, required=True, help='Path to checkpoint file')
    resume_parser.add_argument('--X-path', type=str, help='Path to preprocessed X.npy')
    resume_parser.add_argument('--y-path', type=str, help='Path to preprocessed y.npy')
    resume_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    resume_parser.add_argument('--batch-size', type=int, default=100, help='Batch size (default: 100)')
    resume_parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    resume_parser.add_argument('--sample-fraction', type=float, default=1/8, help='Fraction of samples per epoch')
    resume_parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints', help='Checkpoint directory')
    resume_parser.add_argument('--save-every', type=int, default=1, help='Save checkpoint every N epochs')
    resume_parser.set_defaults(func=cmd_resume)

    predict_parser = subparsers.add_parser('predict', help='Predict audio file class')
    predict_parser.add_argument('audio_file', type=str, help='Path to .wav file to classify')
    predict_parser.add_argument('--model', type=str, default='best_model.pt', help='Path to model checkpoint (default: best_model.pt)')
    predict_parser.add_argument('--top-k', type=int, default=5, help='Number of top predictions (default: 5)')
    predict_parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (default: auto)')
    predict_parser.set_defaults(func=cmd_predict)



    args = parser.parse_args()
    args.func(args)

def cmd_download(args):
    print("Download ESC50 audio data...")

    download_clean()

    print("Data downloaded and cleaned.")

def cmd_preprocess(args):
    print("Processing audio data...")
    
    if args.augment:
        print("Creating augmented datasets...")
        create_augmented_datasets(args.input_dir, args.output_dir)
    
    print("Creating log-mel spectrograms...")
    X, y = create_log_mel(args.input_dir, args.output_dir)
    
    print(f"Dataset size: {len(X)} samples, {len(np.unique(y))} classes")
    print(f"Saved to {args.output_dir}")


def cmd_train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    X_path = args.X_path or "data/preprocessed/X.npy"
    y_path = args.y_path or "data/preprocessed/y.npy"
    
    if os.path.exists(X_path) and os.path.exists(y_path):
        print("Loading existing processed data...")
        X = np.load(X_path, allow_pickle=True)
        y = np.load(y_path)
    else:
        print("Processing audio data...")
        audio_training_path = args.audio_dir or "data/audio/0"
        directories = os.listdir(audio_training_path)
        
        if len(directories) == 1 and args.augment:
            print("Creating augmented datasets...")
            create_augmented_datasets(audio_training_path, "data/audio")
        
        print("Creating log-mel spectrograms...")
        X, y = create_log_mel(args.audio_dir or "data/audio", args.output_dir or "data/preprocessed")
    
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
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        use_all_patches=True,
        samples_per_epoch_fraction=args.sample_fraction,
        checkpoint_dir=args.checkpoint_dir,
        save_every_n_epoch=args.save_every,
        resume_from=None
    )
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
    return best_val_acc

def cmd_resume(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading processed data...")
    X = np.load(args.X_path or "data/preprocessed/X.npy", allow_pickle=True)
    y = np.load(args.y_path or "data/preprocessed/y.npy")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    n_classes = len(np.unique(y))
    model = CNN(n_classes=n_classes)
    
    print(f"Resuming from: {args.resume_from}")
    
    best_val_acc = train_cnn(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        use_all_patches=True,
        samples_per_epoch_fraction=args.sample_fraction,
        checkpoint_dir=args.checkpoint_dir,
        save_every_n_epoch=args.save_every,
        resume_from=args.resume_from
    )
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
    return best_val_acc

def cmd_predict(args):
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    try:
        model = load_model(args.model, device=args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    try:
        predicted_class, top_probs, top_indices = predict_file(
            model, args.audio_file, device=args.device, top_k=args.top_k
        )
        
        print("\n" + "=" * 60)
        print(f"Top {args.top_k} Predictions:")
        print("=" * 60)
        
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            class_name = esc50_labels[idx]
            marker = "★" if idx == predicted_class else " "
            print(f"{marker} {i+1}. {class_name:20s} - {prob*100:6.2f}%")
        
    except Exception as e:
        print(f"\nError during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


main()