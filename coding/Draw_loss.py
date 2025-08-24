import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losees):
    fig, axl = plt.subplots(figsize=(5,3))
    
    # Plot training and validation loss against epochs
    axl.plot(epochs_seen, train_losses, label="Training loss")
    axl.plot(epochs_seen, val_losees, linestyle="-.",label="Validation loss")
    axl.set_xlabel("Epochs")
    axl.set_ylabel("Loss")
    axl.legend(loc="upper right")
    axl.xaxis.set_major_locator(MaxNLocator(integer=True)) # Only show integer labels on x-axis
    
    # Create a second x-axis for tokens seen
    ax2 = axl.twiny() # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses,alpha=0) # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")
    
    fig.tight_layout() # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()