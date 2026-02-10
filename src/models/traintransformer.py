import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import os
import numpy as np
from tqdm import tqdm
import time


class TransformerTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        num_epochs=50,
        learning_rate=1e-4,
        weight_decay=1e-4,
        warmup_epochs=5,
        checkpoint_dir="models/transformer/checkpoints",
        device="cuda"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max = num_epochs - warmup_epochs,
            eta_min=1e-6
        )

        self.warmup_epochs = warmup_epochs
        self.base_lr = learning_rate

        self.criterion = nn.CrossEntropyLoss()

        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.best_val_acc = 0
        self.best_epoch = 0

    def warmup_lr(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
    
    def train_epoch(self, epoch):
        self.model.train()

        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, des=f"Epoch {epoch}/{self.num_epochs}")):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
        
            total_loss += loss.item()
            total += target.size()

            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total,
                'lr': self.optimizer.param_groups[0]['lr']
            })

        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100. * correct / total

        return avg_loss, avg_acc
        
    def validate(self):
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
        }
        
        # Save latest checkpoint
        path = os.path.join(self.checkpoint_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, path)
        
        # Save best checkpoint
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, path)
            print(f'✓ Saved best model with val_acc: {val_acc:.2f}%')
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accs = checkpoint['train_accs']
        self.val_accs = checkpoint['val_accs']
        
        print(f'✓ Loaded checkpoint from epoch {checkpoint["epoch"]}')
        return checkpoint['epoch']
    
    def train(self, resume_from=None):
        """
        Main training loop.
        
        Args:
            resume_from: Path to checkpoint to resume from
        
        Returns:
            Best validation accuracy
        """
        start_epoch = 1
        
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1
        
        print(f'\nStarting training for {self.num_epochs} epochs')
        print(f'Device: {self.device}')
        print(f'Training samples: {len(self.train_loader.dataset)}')
        print(f'Validation samples: {len(self.val_loader.dataset)}')
        print('-' * 60)
        
        start_time = time.time()
        
        for epoch in range(start_epoch, self.num_epochs + 1):
            # Warmup learning rate
            if epoch <= self.warmup_epochs:
                self._warmup_lr(epoch - 1)
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler (after warmup)
            if epoch > self.warmup_epochs:
                self.scheduler.step()
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Print epoch summary
            print(f'\nEpoch {epoch}/{self.num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
            
            self.save_checkpoint(epoch, val_acc, is_best)
            
            # Early stopping check (optional)
            if epoch - self.best_epoch > 30:
                print(f'\nEarly stopping: no improvement for 30 epochs')
                break
        
        elapsed_time = time.time() - start_time
        print(f'\n{"="*60}')
        print(f'Training completed in {elapsed_time/3600:.2f} hours')
        print(f'Best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}')
        print(f'{"="*60}')
        
        return self.best_val_acc

