import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import combineloss
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
print("Número de imágenes cargadas:", images.shape[0])

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.layers import Cropping2D

def build_autoencoder(input_shape=(300, 300, 1)):
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

    # Si es necesario ajustar el tamaño de salida, puedes añadir una capa de Cropping2D aquí
    # Ejemplo: si el tamaño es 304x304 y necesitas 300x300
    decoded = Cropping2D(cropping=((2, 2), (2, 2)))(decoded)

    autoencoder = Model(input_img, decoded)
    return autoencoder


autoencoder = build_autoencoder()
autoencoder.summary()
lossmethod='ssim'
if lossmethod=="mean_squared_error":
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
elif lossmethod.lower() == "ssim":
    import tensorflow as tf

    # def ssim_loss(y_true, y_pred):
    #     return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    # autoencoder.compile(optimizer='adam', loss=ssim_loss)
    from SSIM import calculate_ssim
    autoencoder.compile(optimizer='adam', loss=calculate_ssim)
else:
    #generate error message
    print("Error: loss method not recognized")



modo='manual'
numepochs=5
if modo=="automatic":
    # Entrenamiento
    autoencoder.fit(images, images, epochs=numepochs, batch_size=32, validation_split=0.2) #for automatic splitting
elif modo=="manual":
    #for manual spliting
    from sklearn.model_selection import train_test_split

    # Dividir los datos en conjuntos de entrenamiento y validación
    images_train, images_val = train_test_split(images, test_size=0.2, random_state=42)
    # Entrenamiento
    x_train=images_train
    y_train=images_train
    x_test=images_val
    y_test=images_val
    # Entrenamiento
    combineloss_pre=combineloss.extract_and_combine_losses(autoencoder,x_train, y_train, x_test, y_test,"pre")
    history=autoencoder.fit(images_train, images_train, epochs=numepochs, batch_size=32, validation_data=(images_val, images_val))
    combineloss_post=combineloss.extract_and_combine_losses(autoencoder,x_train, y_train, x_test, y_test,"post",history)
    _,_,fig = combineloss.combine_pretraining_and_training_losses(combineloss_pre,combineloss_post)
    #SAVE FIGURE
    fig.savefig('ANNpictures/bark_Autoencoder_losses.png')
else:
    #generate error message
    print("Error: mode not recognized")

import matplotlib.pyplot as plt
# Visualización de las reconstrucciones
def display_reconstructions(model, images, n=10):
    indices = np.random.randint(len(images), size=n)
    test_images = images[indices]
    reconstructions = model.predict(test_images)

    plt.figure(figsize=(20, 4))
    for i, image_idx in enumerate(indices):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(images[image_idx].reshape(300, 300))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructions[i].reshape(300, 300))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    #SAVE FIGURE
    plt.savefig('ANNpictures/bark_Autoencoder_reconstructions.png')
    plt.show()

display_reconstructions(autoencoder, images_val)

