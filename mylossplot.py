import numpy as np
import matplotlib.pyplot as plt

# Replace these lists with the actual numbers of neurons and training examples you used
neurons_list = [500,1000,2000]
img_list = [1000, 3000, 10000, 30000]

plt.figure()

# Loop over all combinations of neurons and images
for n_neurons in neurons_list:
    for n_img in img_list:
        # Load the validation loss array
        val_loss = np.load(f'ANNpictures/validation_loss_{n_neurons}n_{n_img}img.npy')
        
        # Plot the validation loss
        plt.plot(val_loss, label=f'{n_neurons} neurons, {n_img} images')

# Add a legend
plt.legend()

# Add labels and a title
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss vs Epoch for Different Numbers of Neurons and Training Examples')
#savefig
plt.savefig(f'ANNpictures/validation_loss_{n_neurons}n_{n_img}img.png')

# Show the plot
plt.show()

#do the same for training loss
plt.figure()

# Loop over all combinations of neurons and images
for n_neurons in neurons_list:
    for n_img in img_list:
        # Load the training loss array
        train_loss = np.load(f'ANNpictures/training_loss_{n_neurons}n_{n_img}img.npy')
        
        # Plot the training loss
        plt.plot(train_loss, label=f'{n_neurons} neurons, {n_img} images')

# Add a legend
plt.legend()

# Add labels and a title
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss vs Epoch for Different Numbers of Neurons and Training Examples')
#savefig
plt.savefig(f'ANNpictures/training_loss_{n_neurons}n_{n_img}img.png')
# Show the plot
plt.show()