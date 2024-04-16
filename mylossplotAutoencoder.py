import numpy as np
import matplotlib.pyplot as plt

# Replace these lists with the actual numbers of neurons and training examples you used
neurons_list = [250,500,1000,2000]
img_list = [1000, 3000, 10000, 30000]

plt.figure()

# Loop over all combinations of neurons and images
for n_neurons in neurons_list:
    for n_img in img_list:
        # Load the validation loss array
        val_loss = np.load(f'ANNpictures/Autoencoder_validation_loss_{n_neurons}n_{n_img}img_norm.npy')
        
        # Plot the validation loss
        plt.plot(val_loss, label=f'{n_neurons} neurons, {n_img} images')

# Add a legend
plt.legend()

# Add labels and a title
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Autoencoder-Validation Loss vs Epoch for Different Numbers of Neurons and Training Examples')
#savefig
plt.savefig(f'ANNpictures/Autoencoder_validation_loss_n_img_norm.png',bbox_inches='tight')

# Show the plot
plt.show()

#do the same for training loss
plt.figure()

# Loop over all combinations of neurons and images
for n_neurons in neurons_list:
    for n_img in img_list:
        # Load the training loss array
        train_loss = np.load(f'ANNpictures/Autoencoder_training_loss_{n_neurons}n_{n_img}img_norm.npy')
        
        # Plot the training loss
        plt.plot(train_loss, label=f'{n_neurons} neurons, {n_img} images')

# Add a legend
plt.legend()

# Add labels and a title
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Autoencoder-Training Loss vs Epoch for Different Numbers of Neurons and Training Examples')
#savefig
plt.savefig(f'ANNpictures/Autoencoder_training_loss_n_img_norm.png',bbox_inches='tight')
# Show the plot
plt.show()

# Initialize a 2D array to store the loss values
# Initialize a 2D array to store the loss values
loss_values_train = np.zeros((len(neurons_list), len(img_list)))
loss_values_val = np.zeros((len(neurons_list), len(img_list)))

# Loop over all combinations of neurons and images
for i, n_neurons in enumerate(neurons_list):
    for j, n_img in enumerate(img_list):
        # Load the training loss array
        train_loss = np.load(f'ANNpictures/Autoencoder_training_loss_{n_neurons}n_{n_img}img_norm.npy')
        val_loss = np.load(f'ANNpictures/Autoencoder_validation_loss_{n_neurons}n_{n_img}img_norm.npy')
        
        # Store the mean loss value in the 2D array
        loss_values_train[i, j] = np.mean(train_loss)
        loss_values_val[i, j] = np.mean(val_loss)

# Create a figure and two subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

#I want to know the maximum value of the loss for the colorbar
max_val = max(np.max(loss_values_train), np.max(loss_values_val))

# Create a 2D color plot of the training loss values
im1 = axs[0].imshow(loss_values_train, cmap='hot', interpolation='nearest')
# fig.colorbar(im1, ax=axs[0], label='Mean Training Loss', location='bottom')
# im1.set_clim(0, max_val)
im1.set_clim(0, 1)
axs[0].set_xticks(np.arange(len(img_list)))
axs[0].set_yticks(np.arange(len(neurons_list)))
axs[0].set_xticklabels(img_list)
axs[0].set_yticklabels(neurons_list)
axs[0].set_xlabel('Number of Training Examples')
axs[0].set_ylabel('Number of Neurons')
axs[0].set_title('Autoencoder-Mean Training Loss for # (Neurons, Training Examples)')

# Create a 2D color plot of the validation loss values
im2 = axs[1].imshow(loss_values_val, cmap='hot', interpolation='nearest')
# Create a colorbar for both plots
fig.colorbar(im2, ax=axs, label='Mean Validation Loss', location='bottom')
# im2.set_clim(0, max_val)
im2.set_clim(0, 1)
axs[1].set_xticks(np.arange(len(img_list)))
axs[1].set_yticks(np.arange(len(neurons_list)))
axs[1].set_xticklabels(img_list)
axs[1].set_yticklabels(neurons_list)
axs[1].set_xlabel('Number of Validation Examples')
axs[1].set_ylabel('Number of Neurons')
axs[1].set_title('Autoencoder-Mean Validation Loss for # (Neurons, Training Examples)')
plt.tight_layout

# Save the figure
plt.savefig(f'ANNpictures/Autoencoder_mean_loss_norm.png',bbox_inches='tight')

# Show the plot
plt.show()