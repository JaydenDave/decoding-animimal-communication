import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras import (
    layers,
    models,
    callbacks,
    losses,
    utils,
    metrics,
    optimizers,
)
import librosa as lb
from model import WaveGAN

DIM = 64
CHANNELS = 1 #keeping as 1? what for mono or stereo?
PHASE_PARAM = 2
LATENT_DIM = 100
DISCRIMINATOR_STEPS = 5
GP_WEIGHT = 10
LEARNING_RATE = 1e-4
ADAM_BETA_1 = 0.5
ADAM_BETA_2 = 0.9
BATCH_SIZE = 64
EPOCHS = 100

if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")

wavegan = WaveGAN(
    latent_dim = LATENT_DIM,
    discriminator_steps= DISCRIMINATOR_STEPS,
    gp_weight= GP_WEIGHT,
)

wavegan.compile(
    d_optimizer = optimizers.Adam(learning_rate=LEARNING_RATE, beta_1 = ADAM_BETA_1, beta_2 = ADAM_BETA_2),
    g_optimizer = optimizers.Adam(learning_rate=LEARNING_RATE, beta_1 = ADAM_BETA_1, beta_2 = ADAM_BETA_2),
)

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

def load_raw_audio(data_path):
    audio = []
    for folder in os.listdir(data_path):
        path = os.path.join(data_path, folder)
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            signal, sr = lb.load(file_path)
            signal = pad_audio(signal, max = 22050)
            audio.append(signal)
        print(f"Loaded audio from {folder}")
    print(f"Loaded audio from {data_path}")
    return audio

def pad_audio(signal, max):
    if len(signal) < max:
        num_missing_samples = max - len(signal)
        padded_array = np.pad(signal,
                              (0,num_missing_samples),
                              mode = "constant")
        return padded_array
    return signal

path = r"C:\Users\Jayde\Desktop\Datasets\sc09\sc09"
train_data = load_raw_audio(path)

wavegan.fit(
    train_data,
    batch_size = BATCH_SIZE,
    shuffle = True,
    epochs=EPOCHS,
    callbacks=[
        model_checkpoint_callback,
        tensorboard_callback
    ],
)

wavegan.save("./models/vae")
wavegan.generator.save("./models/generator")
wavegan.discriminator.save("./models/discriminator")
print("models saved")
print("complete")

