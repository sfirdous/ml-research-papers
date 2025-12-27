import matplotlib.pyplot as plt
train_losses = [0.2336, 0.1715, 0.1633, 0.1650, 0.1383, 0.1358, 0.1294, 0.1268, 0.1201, 0.1144, 0.1112, 0.1093, 0.1061, 0.1013, 0.1059, 0.1023, 0.0953, 0.0959, 0.0949, 0.0904]

val_losses = [0.1838, 0.1970, 0.1426, 0.1401, 0.1488, 0.1619, 0.1326, 0.1306, 0.1347, 0.1198, 0.1169, 0.1146, 0.1128, 0.1291, 0.1101, 0.1078, 0.1138, 0.1048, 0.1190, 0.1078]

val_accuracies = [94.90, 95.02, 95.56, 95.23, 95.40, 95.11, 95.78, 95.73, 95.80, 95.98, 95.98, 96.07, 96.14, 95.95, 96.20, 96.25, 96.23, 96.37, 96.13, 96.40]

def plot_training_history(train_losses, val_losses, val_accuracies):
    """Visualize training progress"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)
    
    # Accuracy curve
    ax2.plot(epochs, val_accuracies, 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax2.set_title('Validation Accuracy Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()

plot_training_history(train_losses, val_losses, val_accuracies)
