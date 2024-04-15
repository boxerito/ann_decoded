import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gabor_kernel
import os
import tensorflow as tf

# Set the seed for TensorFlow's random module. To guarantee consistent results, you can set the seed to a fixed value.
# tf.random.set_seed(0)  # Replace 0 with your desired seed

# ... rest of your code ...

# load the data and make training and testing/validation datasets
path = '~/data/Response_Simulation/A/'
path = os.path.expanduser(path)

n_neu = [250, 500, 1000, 2000]
numbers_of_images = [1000, 3000, 10000, 30000]

if not os.path.exists('ANNpictures'):
    os.makedirs('ANNpictures')

def load_data(fname):
    data = np.genfromtxt(os.path.join(path, fname + '.csv'), delimiter=',')
    n_datapoints = data.shape[0]
    n_train = int(np.round(n_datapoints * 0.8))
    n_test = int(np.round(n_datapoints * 0.2))
    bools = np.concatenate((np.tile(True, n_train), np.tile(False, n_test)))
    np.random.shuffle(bools)
    return ((data[bools, 1024:], data[bools, 0:1024]), (data[np.logical_not(bools), 1024:], data[np.logical_not(bools), 0:1024]))

def load_rf(fname):
    return np.genfromtxt(path + fname + '_neuron_prop.csv', delimiter=',')

def make_rf(props, width, height):
    pos_x, pos_y, freq, theta, sigma_x, sigma_y, offset = props
    rf = np.zeros((height, width))
    k = np.real(gabor_kernel(freq, theta=theta, sigma_x=sigma_x, sigma_y=sigma_y, offset=offset))
    x_idx = np.arange(pos_x, pos_x + k.shape[1]) % width
    y_idx = np.arange(pos_y, pos_y + k.shape[0]) % height
    x_idx, y_idx = np.meshgrid(x_idx, y_idx)
    rf[y_idx.astype(int), x_idx.astype(int)] = k
    return rf / np.sqrt(np.sum(rf**2))
maxval=[]
for n_neurons in n_neu:
    for n_img in numbers_of_images:
        fname = f'neurons_to_cifar_{n_neurons}n_1rep{n_img}n_img'

        # Load the data
        (x_train, y_train), (x_test, y_test) = load_data(fname)

        # Neural network model
        input_layer = tf.keras.Input(shape=(x_train.shape[1],))
        layer2 = tf.keras.layers.Dense(128, activation='relu')(input_layer)
        layer3 = tf.keras.layers.Dense(64, activation='relu')(layer2)
        layer4 = tf.keras.layers.Dense(32, activation='relu')(layer3)
        output_layer = tf.keras.layers.Dense(y_train.shape[1], activation='sigmoid')(layer4)

        decoder = tf.keras.Model(inputs=input_layer, outputs=output_layer, name="decoder")
        
        # Check if the output layer's dimension matches the target output dimension
        output_neurons = decoder.layers[-1].output_shape[-1]  # Get the number of neurons in the output layer
        target_output_dim = y_train.shape[1]  # Get the target output dimension

        if output_neurons == target_output_dim:
            print("Dimensions match. Output layer has same # neurons as target output: ", output_neurons)
        else:
            print(f"Mismatch in dimensions: Output layer has {output_neurons} neurons, target output has {target_output_dim} dimensions.")

        # Compile and train the model
        decoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
        #SUGGEST ALTERNATIVE FOR LOSS FUNCTION
        # decoder.compile(optimizer='adam',loss=tf.keras.losses.MeanAbsoluteError())
        numberepochs=20
        history = decoder.fit(x_train, y_train, epochs=numberepochs, shuffle=True, validation_data=(x_test, y_test))

        # Save and plot training history, reconstructed images, etc.
        decoder.save(f'ANNpictures/decoderMLP_{fname}.h5')
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss for ' + fname+ ' (MLP)')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        #ylim
        plt.ylim(0, 0.07)
        plt.legend(['train', 'test'], loc='upper right')
        #savefig
        plt.savefig(f'ANNpictures/MLP_model_loss_{fname}.png')
        #Store the numpy of loss and val_loss
        np.save(f'ANNpictures/MLP_training_loss_{n_neurons}n_{n_img}img.npy', history.history['loss'])
        np.save(f'ANNpictures/MLP_validation_loss_{n_neurons}n_{n_img}img.npy', history.history['val_loss'])
        plt.show()
        maxval.append(max([max(history.history['val_loss']),max(history.history['loss'])]))

        # Reconstruct images using the decoder
        reconstructed = decoder.predict(x_test)

        # Visualize the first N images for comparison
        n_images_to_show = 10
        fig, axs = plt.subplots(2, n_images_to_show, figsize=(20, 4))

        for i in range(n_images_to_show):
            # Display original image
            axs[0, i].imshow(y_test[i].reshape(32, 32), cmap='gray')
            axs[0, i].set_title(f"Original {i+1}")
            axs[0, i].axis('off')

            # Display reconstructed image
            axs[1, i].imshow(reconstructed[i].reshape(32, 32), cmap='gray')
            axs[1, i].set_title(f"Reconstructed {i+1}")
            axs[1, i].axis('off')

        plt.tight_layout()
        fig.suptitle('Reconstructed images for ' + fname+ ' (MLP)',y=1.05)
        plt.tight_layout()
        #savefig
        plt.savefig(f'ANNpictures/reconstruction_MLP_{fname}.png')
        plt.show()


print(maxval)
print(max(maxval))


# #uSEFUL ADDITIONAL INFO
# from tensorflow.keras.models import load_model

# # Replace 'model.h5' with the path to your HDF5 file
# model = load_model('model.h5')
# predictions = model.predict(data)
# from tensorflow.keras.models import load_model

# # Load the model
# model = load_model('model.h5')

# # Evaluate the model
# # Replace 'test_data' and 'test_labels' with your test dataset and labels
# loss, metrics = model.evaluate(test_data, test_labels)

# print(f'Loss: {loss}')
# print(f'Metrics: {metrics}')