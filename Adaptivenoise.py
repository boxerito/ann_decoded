import tensorflow as tf

class ActivityDependentNoise(tf.keras.layers.Layer):
    """
    Custom layer that applies activity-dependent noise to the inputs during training.

    Args:
        noise_factor (float): The factor to control the amount of noise to be added.

    Attributes:
        noise_factor (float): The factor to control the amount of noise to be added.

    Methods:
        build(input_shape): Builds the layer by creating the variables.
        call(inputs, training=None): Applies activity-dependent noise to the inputs.

    Usage:
        input_layer = tf.keras.layers.Input(shape=(32, 32, 1))
        x = tf.keras.layers.Flatten()(input_layer)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = ActivityDependentNoise(noise_factor=0.1)(x, training=True)  # Apply adaptive noise
        output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='mean_squared_error')
    """

    def __init__(self, noise_factor=0.1, **kwargs):
        super(ActivityDependentNoise, self).__init__(**kwargs)
        self.noise_factor = noise_factor

    def build(self, input_shape):
        """
        Builds the layer by creating the variables.

        Args:
            input_shape (tuple): The shape of the input tensor.

        Returns:
            None
        """
        super(ActivityDependentNoise, self).build(input_shape)

    def call(self, inputs, training=None):
        """
        Applies activity-dependent noise to the inputs.

        Args:
            inputs (tensor): The input tensor.
            training (bool): Whether the layer is in training mode or not.

        Returns:
            tensor: The output tensor with activity-dependent noise applied.
        """
        if training:
            stddev = self.noise_factor * tf.math.reduce_std(inputs)
            noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=stddev)
            return inputs + noise
        else:
            return inputs

# Using the layer in a model
# input_layer = tf.keras.layers.Input(shape=(32, 32, 1))
# x = tf.keras.layers.Flatten()(input_layer)
# x = tf.keras.layers.Dense(128, activation='relu')(x)
# x = ActivityDependentNoise(noise_factor=0.1)(x, training=True)  # Apply adaptive noise
# output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
# model.compile(optimizer='adam', loss='mean_squared_error')
