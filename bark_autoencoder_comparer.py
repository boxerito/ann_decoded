import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def load_images(directory, target_size=(300, 300)):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):  # Asumiendo que son PNG, ajusta según sea necesario
            path = os.path.join(directory, filename)
            image = load_img(path, color_mode='grayscale', target_size=target_size)
            image = img_to_array(image)
            images.append(image)
    images = np.array(images, dtype='float32')
    images /= 255.0  # Normalización a [0, 1]
    return images

# Carga las imágenes
images = load_images('barkpictures')
epochsnum = 5
print("Número de imágenes cargadas:", images.shape[0])

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.layers import Cropping2D

from tensorflow.keras.layers import UpSampling2D

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D
from tensorflow.keras.models import Model

def build_autoencoder_transpose(input_shape=(300, 300, 1)):
    input_img = Input(shape=input_shape)

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Ajustar las dimensiones si es necesario con Cropping
    decoded = Cropping2D(cropping=((2, 2), (2, 2)))(decoded)

    autoencoder = Model(input_img, decoded)
    return autoencoder

autoencoder_transpose = build_autoencoder_transpose()
autoencoder_transpose.compile(optimizer='adam', loss='mean_squared_error')
autoencoder_transpose.fit(images, images, epochs=epochsnum, batch_size=32, validation_split=0.2)

from tensorflow.keras.layers import UpSampling2D

def build_autoencoder_upsampling(input_shape=(300, 300, 1)):
    input_img = Input(shape=input_shape)

        # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)

    # Ajustar las dimensiones si es necesario con Cropping
    decoded = Cropping2D(cropping=((2, 2), (2, 2)))(decoded)
    autoencoder = Model(input_img, decoded)
    return autoencoder

autoencoder_upsampling = build_autoencoder_upsampling()
autoencoder_upsampling.compile(optimizer='adam', loss='mean_squared_error')
autoencoder_upsampling.fit(images, images, epochs=epochsnum, batch_size=32, validation_split=0.2)


import matplotlib.pyplot as plt

def display_comparisons(model1, model2, images, n=10):
    indices = np.random.randint(len(images), size=n)
    test_images = images[indices]
    reconstructions1 = model1.predict(test_images)
    reconstructions2 = model2.predict(test_images)

    plt.figure(figsize=(20, 6))
    for i, image_idx in enumerate(indices):
        # Display original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(images[image_idx].reshape(300, 300), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        #add a title
        ax.set_title('Original')

        # Display reconstruction from model 1 (Conv2DTranspose)
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(reconstructions1[i].reshape(300, 300), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        #add a title
        ax.set_title('Conv2DTranspose')

        # Display reconstruction from model 2 (UpSampling2D + Conv2D)
        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(reconstructions2[i].reshape(300, 300), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        #add a title
        ax.set_title('UpSampling2D + Conv2D')
    #add a good title
    plt.suptitle('Comparison of reconstructions with ' + str(epochsnum) + ' epochs')
    #save image
    plt.tight_layout()
    plt.savefig('ANNpictures/comparison.png')
    plt.show()

# Suponiendo que `autoencoder_transpose` y `autoencoder_upsampling` son tus modelos entrenados:
display_comparisons(autoencoder_transpose, autoencoder_upsampling, images)
#display_comparisons(autoencoder, autoencoder_upsampling, images)