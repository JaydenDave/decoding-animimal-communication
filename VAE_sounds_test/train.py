import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.python.keras import (
    layers,
    models,
    callbacks,
    losses,
    utils,
    metrics,
    optimizers,
)
import tensorflow.python.keras.backend as K
from model import VAE

INPUT_SHAPE = (256,64,1)
CONV_FILTERS = (512, 256, 128, 64, 32)
N_CHANNELS = INPUT_SHAPE[2]
BATCH_SIZE = 4
LATENT_DIM = 200
BETA = 2000
LEARNING_RATE = 0.0005
EPOCHS = 50
VALIDATION_SPLIT = 0.2

def load_spectrograms(spectrogram_dir):
    spectrograms = []
    file_names = []
    for file in os.listdir(spectrogram_dir):
        file_path = os.path.join(spectrogram_dir, file)
        spect = np.load(file_path)
        spectrograms.append(spect)
        file_names.append(file)

    spectrograms = np.array(spectrograms) #(num bins, num layers)
    #only 2 dimensions - no number of channels, need to reshape to add the extra dimension
    #the ... means keeping the existing axis of the array unchangedm np,newaxis adds new dimension with size of 1
    spectrograms = spectrograms[..., np.newaxis] # shape should now be (number of samples, number of bins, number of frames, 1), treating the spectrograms as greyscale images
    return spectrograms, file_names

#SPECTROGRAM_DIR = r"C:\Users\Jayde\Desktop\library\bird_songs\data\bird_songs\spectrograms"
SPECTROGRAM_DIR = "/home/jayden/decoding-animimal-communication/VAE_sounds_test/spectrograms"
train_data, file_names = load_spectrograms(SPECTROGRAM_DIR)
#train_data = tf.data.Dataset.from_tensor_slices(spectrograms)
#train_data = train_data.batch(batch_size = 32, drop_remainder=True)
print("got train data")

vae = VAE(input_shape = (256,128,1),
        conv_filters = (512, 256, 128, 64, 32),
        latent_dim= 128,
        )

optimizer = optimizers.Adam(learning_rate=0.0005)
vae.compile(optimizer=optimizer)

model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath="./checkpoint",
    save_weights_only=False,
    save_freq="epoch",
    monitor="loss",
    mode="min",
    save_best_only=True,
    verbose=0,
)
tensorboard_callback = callbacks.TensorBoard(log_dir="./logs")

vae.fit(
    train_data,
    batch_size = BATCH_SIZE,
    shuffle = True,
    epochs=EPOCHS,
    callbacks=[
        model_checkpoint_callback,
        tensorboard_callback
    ],
)

vae.save("./models/vae")
vae.encoder.save("./models/encoder")
vae.decoder.save("./models/decoder")
print("models saved")
print("complete")
