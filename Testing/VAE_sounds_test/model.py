import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import (
    layers,
    models,
    callbacks,
    losses,
    utils,
    metrics,
    optimizers,
)
import tensorflow.python.keras.backend as K


INPUT_SHAPE = (256,64,1)
CONV_FILTERS = (512, 256, 128, 64, 32)
N_CHANNELS = INPUT_SHAPE[2]
BATCH_SIZE = 128
LATENT_DIM = 200
BETA = 100
LEARNING_RATE = 0.0005
EPOCHS = 5
VALIDATION_SPLIT = 0.2



class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        
        #sameple epsilon from normal dist
        epsilon = K.random_normal(shape=(batch,dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def encoder(INPUT_SHAPE, CONV_FILTERS, LATENT_DIM):
        N_CHANNELS = INPUT_SHAPE[2]
        encoder_input = layers.Input(shape= INPUT_SHAPE)
        x = layers.Conv2D(CONV_FILTERS[0], kernel_size = 3, strides = 2, padding = "same")(encoder_input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x) #default leakyrelu alpha value = 0.3

        x = layers.Conv2D(CONV_FILTERS[1], kernel_size = 3, strides = 2, padding = "same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(CONV_FILTERS[2], kernel_size = 3, strides = 2, padding = "same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(CONV_FILTERS[3], kernel_size = 3, strides = 2, padding = "same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(CONV_FILTERS[4], kernel_size = 3, strides = 2, padding = "same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        shape_before_flattening = K.int_shape(x)[1:] #needed for the decoder

        x= layers.Flatten()(x)
        z_mean = layers.Dense(LATENT_DIM, name= "z_mean")(x)
        z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name = "encoder")
        return encoder, shape_before_flattening

def decoder(INPUT_SHAPE, CONV_FILTERS, LATENT_DIM, shape_before_flattening):
    N_CHANNELS = INPUT_SHAPE[2]
    decoder_input = layers.Input(shape=(LATENT_DIM,), name="decoder_input")
    x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape(shape_before_flattening)(x)

    x = layers.Conv2DTranspose(CONV_FILTERS[4], kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(CONV_FILTERS[3], kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(CONV_FILTERS[2], kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(CONV_FILTERS[1], kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(CONV_FILTERS[0], kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    decoder_output = layers.Conv2DTranspose(N_CHANNELS, kernel_size=3, strides=1, activation="sigmoid", padding="same")(x)
    return models.Model(decoder_input, decoder_output)


class VAE(models.Model):
    def __init__(self, input_shape, conv_filters, latent_dim ,**kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder, shape_before_flattenting = encoder(input_shape, conv_filters, latent_dim)
        self.decoder = decoder(input_shape, conv_filters, latent_dim, shape_before_flattenting)
        self.total_loss_tracker = metrics.Mean(name = "total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name = "reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name = "kl_loss")


    @property
    def metrics(self):
        return[
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, reconstruction = self(data)
            reconstruction_loss = tf.reduce_mean(BETA*losses.binary_crossentropy(data, reconstruction, axis= (1,2,3)))
            kl_loss = tf.reduce_mean(tf.reduce_sum(-0.5*(1+ z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis =1,))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        """Step run during validation."""
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, reconstruction = self(data)
        reconstruction_loss = tf.reduce_mean(
            BETA
            * losses.binary_crossentropy(data, reconstruction, axis=(1, 2, 3))
        )
        kl_loss = tf.reduce_mean(
            tf.reduce_sum(
                -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
                axis=1,
            )
        )
        total_loss = reconstruction_loss + kl_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }
    