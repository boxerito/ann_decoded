import tensorflow as tf
from tensorflow.keras import layers, models

def build_attention_autoencoder(input_shape=(32, 32, 1)):
    """
    Build an attentional autoencoder model.

    Parameters:
        input_shape (tuple): The shape of the input data. Default is (32, 32, 1).

    Returns:
        keras.models.Model: The attentional autoencoder model.

    """
    # Encoder
    input_layer = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Attention Mechanism
    attention = layers.Conv2D(64, (3, 3), activation='sigmoid', padding='same')(encoded)
    attended_encoding = layers.multiply([encoded, attention])

    # Decoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(attended_encoding)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Autoencoder Model
    autoencoder = models.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    return autoencoder

# Build the model
autoencoder_model = build_attention_autoencoder()

# Model summary
autoencoder_model.summary()
