
def plot_training_history(checkpoint_dir):
    
    history_path = os.path.join(checkpoint_dir, "training_history.json")
    
    if not os.path.exists(history_path):
        print(f"No training history found at {history_path}")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = history['epochs']
    train_loss = history['train_loss']
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, train_acc, 'b-', label='Train Accuracy')
    ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(checkpoint_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved training curves to {plot_path}")
    plt.show()
