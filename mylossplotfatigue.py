import numpy as np
import matplotlib.pyplot as plt

# Replace these lists with the actual numbers of neurons and training examples you used
n_neurons = 1000
img_list = [1000, 3000, 10000, 30000]
fatigue_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
n_neurons_list = [250,500,1000,2000]

for n_neurons in n_neurons_list:
    # Loop over all combinations of neurons and images
    for n_img in img_list:
        for fatigue in fatigue_list:
            # Load the validation loss array
            val_loss = np.load(f'ANNpictures/validation_loss_{n_neurons}n_{n_img}img_{fatigue}fatigue.npy')
            
            # Plot the validation loss
            plt.plot(val_loss, label=f'{n_img} img, {fatigue} fatigue')

    # Add a legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add labels and a title
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss vs Epoch for Different fatigues and Images, 1000 neurons')
    #savefig
    plt.savefig(f'ANNpictures/validation_loss_{n_neurons}_fatigues.png',bbox_inches='tight') #ADD bbox_inches='tight' to savefig to avoid cropping the legend

    # Show the plot
    plt.show()

  

# Loop over all combinations of neurons and images
for n_neurons in n_neurons_list:
    for n_img in img_list:
        for fatigue in fatigue_list:
            # Load the training loss array
            train_loss = np.load(f'ANNpictures/training_loss_{n_neurons}n_{n_img}img_{fatigue}fatigue.npy')
            
            # Plot the training loss
            plt.plot(train_loss, label=f'{n_img} img, {fatigue} fatigue')

    # Add a legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add labels and a title
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(f'Training Loss vs Epoch for Different fatigues and Images, {n_neurons} neurons')
    #savefig
    plt.savefig(f'ANNpictures/training_loss_n_{n_neurons}_fatigues.png',bbox_inches='tight') #ADD bbox_inches='tight' to savefig to avoid cropping the legend
    # Show the plot
    plt.show()

# Initialize a 2D array to store the loss values
# Initialize a 2D array to store the loss values
max_val = []
absmax=0.3
for n_neurons in [250,500,1000,2000]:
    loss_values_train = np.zeros((len(fatigue_list), len(img_list)))
    loss_values_val = np.zeros((len(fatigue_list), len(img_list)))

    # Loop over all combinations of fatigues and images
    for i, fatigue in enumerate(fatigue_list):
        for j, n_img in enumerate(img_list):
            # Load the training loss array
            train_loss = np.load(f'ANNpictures/training_loss_{n_neurons}n_{n_img}img_{fatigue}fatigue.npy')
            val_loss = np.load(f'ANNpictures/validation_loss_{n_neurons}n_{n_img}img_{fatigue}fatigue.npy')
            
            # Store the mean loss value in the 2D array
            loss_values_train[i, j] = np.mean(train_loss)
            loss_values_val[i, j] = np.mean(val_loss)

    # Create a figure and two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    #I want to know the maximum value of the loss for the colorbar
    max_val.append(max(np.max(loss_values_train), np.max(loss_values_val)))

    # Create a 2D color plot of the training loss values
    im1 = axs[0].imshow(loss_values_train, cmap='hot', interpolation='nearest')
    # fig.colorbar(im1, ax=axs[0], label='Mean Training Loss', location='bottom')
    im1.set_clim(0, absmax)
    axs[0].set_xticks(np.arange(len(img_list)))
    axs[0].set_yticks(np.arange(len(fatigue_list)))
    axs[0].set_xticklabels(img_list)
    axs[0].set_yticklabels(fatigue_list)
    axs[0].set_xlabel('Number of Training Examples')
    axs[0].set_ylabel('fatigue coeff.')
    axs[0].set_title(f'Mean Training Loss (images x fatigue, {n_neurons} neurons)')

    # Create a 2D color plot of the validation loss values
    im2 = axs[1].imshow(loss_values_val, cmap='hot', interpolation='nearest')
    # Create a colorbar for both plots
    # fig.colorbar(im2, ax=axs, label='Mean Validation Loss', location='bottom')
    im2.set_clim(0, absmax)
    axs[1].set_xticks(np.arange(len(img_list)))
    axs[1].set_yticks(np.arange(len(fatigue_list)))
    axs[1].set_xticklabels(img_list)
    axs[1].set_yticklabels(fatigue_list)
    axs[1].set_xlabel('Number of Validation Examples')
    axs[1].set_ylabel('fatigue coeff.')
    axs[1].set_title(f'Mean Validation Loss (images x fatigue, {n_neurons} neurons)')

    # # Save the figure
    plt.tight_layout()
    fig.colorbar(im2, ax=axs, label='Mean Validation Loss', location='right')
    plt.savefig(f'ANNpictures/mean_loss_fatigue*images_{n_neurons}neurons.png',bbox_inches='tight')

    # Show the plot
    plt.show()
#Print the maximum value of the loss
absmax=np.max(max_val)
print(absmax)

max_val2 = []
absmax2=0.3
axs=[]
# For each fatigue level, neuron*images
for fatigue in fatigue_list:
    for n_neurons in n_neurons_list:
        loss_values_train = np.zeros((len(n_neurons_list), len(img_list)))
        loss_values_val = np.zeros((len(n_neurons_list), len(img_list)))

        for i, n in enumerate(n_neurons_list):
            for j, n_img in enumerate(img_list):
                train_loss = np.load(f'ANNpictures/training_loss_{n}n_{n_img}img_{fatigue}fatigue.npy')
                val_loss = np.load(f'ANNpictures/validation_loss_{n}n_{n_img}img_{fatigue}fatigue.npy')

                loss_values_train[i, j] = np.mean(train_loss)
                loss_values_val[i, j] = np.mean(val_loss)

        # Create a figure and two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    #I want to know the maximum value of the loss for the colorbar
    max_val2.append(max(np.max(loss_values_train), np.max(loss_values_val)))

    # Create a 2D color plot of the training loss values
    im1 = axs[0].imshow(loss_values_train, cmap='hot', interpolation='nearest')
    # fig.colorbar(im1, ax=axs[0], label='Mean Training Loss', location='bottom')
    im1.set_clim(0, absmax2)
    axs[0].set_xticks(np.arange(len(img_list)))
    axs[0].set_yticks(np.arange(len(n_neurons_list)))
    axs[0].set_xticklabels(img_list)
    axs[0].set_yticklabels(n_neurons_list)
    axs[0].set_xlabel('Number of Training Examples')
    axs[0].set_ylabel('Number of neurons')
    axs[0].set_title(f'Mean Training Loss (images x neurons, {fatigue} fatigue)')

    # Create a 2D color plot of the validation loss values
    im2 = axs[1].imshow(loss_values_val, cmap='hot', interpolation='nearest')
    # Create a colorbar for both plots
    # fig.colorbar(im2, ax=axs, label='Mean Validation Loss', location='bottom')
    im2.set_clim(0, absmax2)
    axs[1].set_xticks(np.arange(len(img_list)))
    axs[1].set_yticks(np.arange(len(n_neurons_list)))
    axs[1].set_xticklabels(img_list)
    axs[1].set_yticklabels(n_neurons_list)
    axs[1].set_xlabel('Number of Validation Examples')
    axs[1].set_ylabel('Number of neurons')
    axs[1].set_title(f'Mean Validation Loss (images x neurons, {fatigue} fatigue)')

    # # Save the figure
    plt.tight_layout()
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Create an axes divider for the given subplot
    divider = make_axes_locatable(axs[1])

    # Append an axes to the right of axs[1], with 5% width and with pad 0.1
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # Create the colorbar in the appended axes
    fig.colorbar(im2, cax=cax, label='Mean Validation Loss', location='right')
    # fig.colorbar(im2, ax=axs, label='Mean Validation Loss', location='right')
    plt.savefig(f'ANNpictures/mean_loss_images*neurons_{fatigue}fatigue.png',bbox_inches='tight')

    # Show the plot
    plt.show()
#Print the maximum value of the loss
absmax2=np.max(max_val2)
print(absmax2)

max_val3 = []
absmax3=0.3
axs=[]
# For each image value, neuron*fatigue
for n_img in img_list:
    for n_neurons in n_neurons_list:
        loss_values_train = np.zeros((len(n_neurons_list), len(fatigue_list)))
        loss_values_val = np.zeros((len(n_neurons_list), len(fatigue_list)))

        for i, n in enumerate(n_neurons_list):
            for j, fatigue in enumerate(fatigue_list):
                train_loss = np.load(f'ANNpictures/training_loss_{n}n_{n_img}img_{fatigue}fatigue.npy')
                val_loss = np.load(f'ANNpictures/validation_loss_{n}n_{n_img}img_{fatigue}fatigue.npy')

                loss_values_train[i, j] = np.mean(train_loss)
                loss_values_val[i, j] = np.mean(val_loss)

        # Create a figure and two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    #I want to know the maximum value of the loss for the colorbar
    max_val3.append(max(np.max(loss_values_train), np.max(loss_values_val)))

    # Create a 2D color plot of the training loss values
    im1 = axs[0].imshow(loss_values_train, cmap='hot', interpolation='nearest')
    # fig.colorbar(im1, ax=axs[0], label='Mean Training Loss', location='bottom')
    im1.set_clim(0, absmax)
    axs[0].set_xticks(np.arange(len(fatigue_list)))
    axs[0].set_yticks(np.arange(len(n_neurons_list)))
    axs[0].set_xticklabels(fatigue_list)
    axs[0].set_yticklabels(n_neurons_list)
    axs[0].set_xlabel('fatigue coefficient')
    axs[0].set_ylabel('Number of neurons')
    axs[0].set_title(f'Mean Training Loss (fatigue x neurons, {n_img} img)')

    # Create a 2D color plot of the validation loss values
    im2 = axs[1].imshow(loss_values_val, cmap='hot', interpolation='nearest')
    # Create a colorbar for both plots
    # fig.colorbar(im2, ax=axs, label='Mean Validation Loss', location='bottom')
    im2.set_clim(0, absmax)
    axs[1].set_xticks(np.arange(len(fatigue_list)))
    axs[1].set_yticks(np.arange(len(n_neurons_list)))
    axs[1].set_xticklabels(fatigue_list)
    axs[1].set_yticklabels(n_neurons_list)
    axs[1].set_xlabel('fatigue coefficient')
    axs[1].set_ylabel('Number of neurons')
    axs[1].set_title(f'Mean Validation Loss (fatigue x neurons, {n_img} img)')

    # # Save the figure
    plt.tight_layout()
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Create an axes divider for the given subplot
    divider = make_axes_locatable(axs[1])

    # Append an axes to the right of axs[1], with 5% width and with pad 0.1
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # Create the colorbar in the appended axes
    fig.colorbar(im2, cax=cax, label='Mean Validation Loss', location='right')
    plt.savefig(f'ANNpictures/mean_loss_fatigue*neurons_{n_img}img.png',bbox_inches='tight')

    # Show the plot
    plt.show()
#Print the maximum value of the loss
absmax3=np.max(max_val3)
print(absmax3)
