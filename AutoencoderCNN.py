import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# load the data and make training and testing/validation datasets
path = '~/data/Response_Simulation/A/'
path = os.path.expanduser(path)

n_neu = [250, 500, 1000, 2000]
numbers_of_images = [1000, 3000, 10000, 30000]

if not os.path.exists('ANNpictures'):
    os.makedirs('ANNpictures')

def load_data(fname, image_dim=32):
    data = np.genfromtxt(os.path.join(path, fname + '.csv'), delimiter=',')
    n_datapoints = data.shape[0]
    n_train = int(np.round(n_datapoints * 0.8))
    n_test = int(np.round(n_datapoints * 0.2))
    bools = np.concatenate((np.tile(True, n_train), np.tile(False, n_test)))
    np.random.shuffle(bools)
    x = data[:, :image_dim*image_dim].reshape(-1, image_dim, image_dim, 1)  # Reshape for CNN
    y = x  # Assuming we want to reconstruct the same images
    return (x[bools], y[bools]), (x[np.logical_not(bools)], y[np.logical_not(bools)])
maxval=[]
for n_neurons in n_neu:
    for n_img in numbers_of_images:
        fname = f'neurons_to_cifar_{n_neurons}n_1rep{n_img}n_img'

        # Load the data
        (x_train, y_train), (x_test, y_test) = load_data(fname)

        # Define a simple CNN model
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(32, 32, 1)),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D(size=(2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D(size=(2, 2)),
            tf.keras.layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.summary()

        history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

        # Save the model
        model.save(f'ANNpictures/Autoencoder_model_{fname}.h5')

        # Plotting the training and validation loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        ##Store the numpy of loss and val_loss
        np.save(f'ANNpictures/Autoencoder_training_loss_{n_neurons}n_{n_img}img.npy',history.history['loss'])
        np.save(f'ANNpictures/Autoencoder_validation_loss_{n_neurons}n_{n_img}img.npy',history.history['val_loss'])
        maxval.append(max([max(history.history['val_loss']),max(history.history['loss'])]))
        plt.title('Model Loss for ' + fname)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.ylim(0, 0.055)
        plt.legend(['Train', 'Test'], loc='upper right')
        #save the figure
        plt.savefig(f'ANNpictures/model_loss_Autoencoder_{fname}.png',bbox_inches='tight')
        plt.show()

        # Reconstruct images using the trained model
        reconstructed = model.predict(x_test)

        # Visualize the first N images for comparison
        n_images_to_show = 10
        fig, axs = plt.subplots(2, n_images_to_show, figsize=(20, 5))

        for i in range(n_images_to_show):
            axs[0, i].imshow(x_test[i].reshape(32, 32), cmap='gray')
            axs[0, i].set_title("Original")
            axs[0, i].axis('off')

            axs[1, i].imshow(reconstructed[i].reshape(32, 32), cmap='gray')
            axs[1, i].set_title("Reconstructed")
            axs[1, i].axis('off')
        #save the figure
        plt.savefig(f'ANNpictures/reconstruction_Autoencoder_{fname}.png',bbox_inches='tight')
        plt.show()
        # Find the first convolutional layer in the model
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                filters, biases = layer.get_weights()
                break

        # Proceed with filter visualization
        n_filters = filters.shape[3]
        n_display = min(n_filters, 10)  # Display the first 10 filters

        # Set up the plot dimensions
        fig, axs = plt.subplots(1, n_display, figsize=(20, 2))
        fig.suptitle('First Convolutional Layer Filters')

        for i in range(n_display):
            # Get the filter
            f = filters[:, :, 0, i]  # Assuming grayscale images, hence the channel is 0

            # Plot each filter
            axs[i].imshow(f, cmap='gray')
            axs[i].set_title(f'Filter {i+1}')
            axs[i].axis('off')
        #save the figure
        plt.savefig(f'ANNpictures/First_Conv_Layer_Filters_Autoencoder_{fname}.png',bbox_inches='tight')
        plt.show()
print(maxval)
print(max(maxval))