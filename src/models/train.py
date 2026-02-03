import os
import torch
import tqdm
import json
import numpy as np
from torch.utils.data import DataLoader

from src.models.predict import predict_with_overlapping_patches
from src.data.datasets import FullTFPatchesDataset, RandomPatchDataset

def train_cnn(
    model,
    X_train, y_train,
    X_val, y_val,
    fold_num,
    epochs=50,
    batch_size=100,
    lr=0.001,
    device="cuda",
    use_all_patches=True,
    samples_per_epoch_fraction=1/8,
    checkpoint_dir="models/checkpoints",
    save_every_n_epoch=1,
    resume_from=None
    ):
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.to(device)
    
    if use_all_patches:
        train_dataset = FullTFPatchesDataset(X_train, y_train, patch_length=128)
        print(f"\n{'='*60}")
        print("Using ALL PATCHES method (as per paper)")
        print(f"{'='*60}")
    else:
        train_dataset = RandomPatchDataset(X_train, y_train, patch_length=128)
        print(f"\n{'='*60}")
        print("Using RANDOM PATCHES method (simpler)")
        print(f"{'='*60}")
    
    # unique, counts = np.unique(y_train, return_counts=True)
    # print(f"\nClass distribution in y_train:")
    # print(f"Classes: {len(unique)}")
    # print(f"Min samples: {counts.min()}, Max samples: {counts.max()}, Mean: {counts.mean():.1f}")
    # print(f"\nPer-class counts:")
    # for cls, count in zip(unique, counts):
    #     print(f"Class {cls}: {count}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    total_patches = len(train_dataset)
    patches_per_epoch = int(total_patches * samples_per_epoch_fraction)
    batches_per_epoch = patches_per_epoch // batch_size
    
    print(f"Total available patches: {total_patches:,}")
    print(f"Patches per epoch ({samples_per_epoch_fraction}): {patches_per_epoch:,}")
    print(f"Batches per epoch: {batches_per_epoch:,}")
    print(f"{'='*60}\n")
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'weight_decay': 0.0},
        {'params': model.classifier.parameters(), 'weight_decay': 0.001}
    ], lr=lr)#, momentum=0.9)
    

    start_epoch = 0
    best_val_acc = 0.0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'epochs': []
    }

    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        training_history = checkpoint['history']

        print(f"Resuming training from epoch: {checkpoint['epoch']}")
        print(f"Best val acc: {best_val_acc:.4f}\n")



    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        batches_processed = 0
        
        for xb, yb in tqdm.tqdm(train_loader, f"Epoch {epoch+1} Train", leave=False):
            if batches_processed >= batches_per_epoch:
                break
            
            xb = xb.to(device)
            yb = yb.to(device)
            
            optimizer.zero_grad()
            out = model(xb)
                       
            loss = criterion(out, yb)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
     
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            _, pred = out.max(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
            batches_processed += 1
        
        train_loss /= total
        train_acc = correct / total
        
        model.eval()
        val_correct = 0
        val_total = len(y_val)
        

        for i in tqdm.tqdm(range(val_total), desc=f"Epoch {epoch+1} Val", leave=False):
            spec = X_val[i]
            true_label = y_val[i]
            
            pred_label = predict_with_overlapping_patches(model, spec, device=device)
            
            if pred_label == true_label:
                val_correct += 1
        
        val_acc = val_correct / val_total

        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_acc'].append(val_acc)
        training_history['epochs'].append(epoch + 1)

        is_best = val_acc > best_val_acc

        if is_best:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")
        
        print(
            f"Fold {fold_num} | Epoch {epoch+1}/{epochs} | "
            f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | "
            f"Val acc: {val_acc:.4f} (best: {best_val_acc:.4f})"
        )

        if (epoch + 1) % save_every_n_epoch == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc,
                'history': training_history,
                'config': {
                    'batch_size': batch_size,
                    'lr': lr,
                    'total_patches': total_patches,
                    'patches_per_epoch': patches_per_epoch,
                }
            }
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_epoch_{epoch+1}.pt"
            )
            torch.save(checkpoint, checkpoint_path)
            
            if is_best:
                best_path = os.path.join(checkpoint_dir, "best_model.pt")
                torch.save(checkpoint, best_path)
                #print("Saved best model")

            latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
            torch.save(checkpoint, latest_path)
            
            history_path = os.path.join(checkpoint_dir, "training_history.json")
            with open(history_path, 'w') as f:
                json.dump(training_history, f, indent=2)

    final_model_dir = "models/saved"
    os.makedirs(final_model_dir, exist_ok=True)
    final_model_path = os.path.join(final_model_dir, "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_val_acc': best_val_acc,
        'config': {
            'batch_size': batch_size,
            'lr': lr,
            'epochs': epochs,
        }
    }, final_model_path)
    print(f"\nTraining complete! Final model saved to {final_model_path}")

    return best_val_acc


def train_k_fold_cnn(
    model_class,
    X, y,
    epochs=50,
    batch_size=100,
    lr=0.01,
    k_fold=5,
    device="cuda",
    use_all_patches=True,
    samples_per_epoch_fraction=1/8,
    checkpoint_dir="models/checkpoints",
    save_every_n_epoch=1
    ):

    X = np.array(X)
    y = np.array(y)
    n_samples = len(y)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    fold_sizes = (n_samples // 5) * np.ones(5, dtype=int)
    fold_sizes[:n_samples % 5] += 1
    current = 0

    fold_accuracies = []

    for fold_num, fold_size in enumerate(fold_sizes, 1):
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        current = stop

        X_train, y_train = X[train_idx].tolist(), y[train_idx]
        X_val, y_val = X[val_idx].tolist(), y[val_idx]

        print(f"\n{'='*80}")
        print(f"FOLD {fold_num}/5 | Train: {len(X_train)}, Val: {len(X_val)}")
        print(f"{'='*80}\n")

        model = model_class()

        best_acc = train_cnn(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            fold_num=fold_num,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            use_all_patches=use_all_patches,
            samples_per_epoch_fraction=samples_per_epoch_fraction,
            checkpoint_dir=os.path.join(checkpoint_dir, f"fold_{fold_num}"),
            save_every_n_epoch=save_every_n_epoch
        )
        fold_accuracies.append(best_acc)

        print(f"\nFold {fold_num} Best Accuracy: {best_acc:.4f}\n")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    print(f"\n{'='*80}")
    print("FINAL 5-FOLD CROSS-VALIDATION RESULTS")
    print(f"Fold Accuracies: {fold_accuracies}")
    print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"{'='*80}\n")

    # Save results
    results_path = os.path.join(checkpoint_dir, "5fold_cv_results.json")
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({
            'fold_accuracies': fold_accuracies,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc
        }, f, indent=2)
    print(f"Results saved to {results_path}")

    return fold_accuracies, mean_acc