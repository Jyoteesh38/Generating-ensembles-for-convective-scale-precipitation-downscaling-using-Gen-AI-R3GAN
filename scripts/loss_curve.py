import matplotlib.pyplot as plt

# Read the data from the file
with open("history.txt", "r") as file:
    lines = file.readlines()
    
# Extract data values
epochs = []
gloss = []
dloss = []
mse = []
val_gloss = []
val_dloss = []
val_mse = []

# Skip the header line
for line in lines[1:]:
    parts = line.split(',')
    epochs.append(int(parts[0]))
    gloss.append(float(parts[1]))
    dloss.append(float(parts[2]))
    mse.append(float(parts[3]))
    val_gloss.append(float(parts[6]))
    val_dloss.append(float(parts[7]))
    val_mse.append(float(parts[8]))
    
# Create a grid of subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 10))
fig.tight_layout(pad=5.0)

# Increase font size for labels
font_size = 14

# Plot the data on respective subplots
axs[0].plot(epochs, gloss, '-o', label = 'Generator train loss')
axs[0].plot(epochs, val_gloss, '-o', label = 'Generator test loss')
axs[0].set_ylabel('R3GAN Generator loss', fontsize=font_size)

axs[1].plot(epochs, dloss, '-o', label = 'Discriminator train Loss')
axs[1].plot(epochs, val_dloss, '-o', label = 'Discriminator test Loss')
axs[1].set_ylabel('R3GAN Discriminator loss', fontsize=font_size)

# Add a vertical line at epoch 129
for ax in axs:
    ax.axvline(x=129, color='black', linestyle='--')

axes = axs.flatten()
for ax in axes:
    ax.grid(True)
    ax.legend(fontsize=font_size)
    ax.set_xlim(1, 165)
    ax.tick_params(axis='both', labelsize=font_size-2)
    ax.set_xlabel('Epochs', fontsize=font_size)
plt.savefig('GAN_Loss_curve.png')
