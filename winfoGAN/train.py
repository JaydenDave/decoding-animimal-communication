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
from model import GAN
from preprocess import load_raw_audio
import datetime
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
EPOCHS = 3000
CHECKPOINT_FREQ = 4
if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")

wavegan = GAN(
    latent_dim = LATENT_DIM,
    discriminator_steps= DISCRIMINATOR_STEPS,
    gp_weight= GP_WEIGHT,
    n_categories= 10
)

wavegan.compile(
    d_optimizer = optimizers.Adam(learning_rate=LEARNING_RATE, beta_1 = ADAM_BETA_1, beta_2 = ADAM_BETA_2),
    g_optimizer = optimizers.Adam(learning_rate=LEARNING_RATE, beta_1 = ADAM_BETA_1, beta_2 = ADAM_BETA_2),
    q_optimizer= optimizers.RMSprop(learning_rate = LEARNING_RATE)
)

time = datetime.datetime.now().strftime("%d%m.%H%M")
model_path = f"/mt/home/jdave/onedrive/models_{time}"
os.mkdir(model_path)


class Generator_Save_Callback(callbacks.Callback):
    def __init__(self, freq):
        self.freq = freq

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.freq == 0:
            wavegan.generator.save(f"{model_path}/generator{epoch}")

save_gen_callback= Generator_Save_Callback(CHECKPOINT_FREQ)

model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=os.path.join(model_path,"checkpoints/checkpoint_epoch-{epoch:04d}"),
    save_weights_only=False,
    save_freq=CHECKPOINT_FREQ,
    monitor="loss",
    mode="min",
    save_best_only=False,
    verbose=1,
)


tensorboard_callback = callbacks.TensorBoard(log_dir=f"{model_path}/logs")



#path = r"C:\Users\Jayde\Desktop\Datasets\sc09\sc09"

#path = "/home/jayden/sc09/train"
path = "/mt/home/jdave/datasets/sc09/sc09/train"
print(f"Loading data from {path}")
train_data = load_raw_audio(path, n_train_data= 640, model_path= model_path)


wavegan.fit(
    train_data,
    batch_size = BATCH_SIZE,
    shuffle = True,
    epochs=EPOCHS,
    callbacks=[
        save_gen_callback,
        tensorboard_callback
    ],
)
#input = tf.TensorSpec(tf.random.normal(shape=(None, LATENT_DIM)))
#wavegan.build((0,100)) #train data or is it looking for the normal distribution?
#tf.saved_model.save(wavegan,"wavegan")
#wavegan.save("./models/vae")

wavegan.generator.save(f"{model_path}/generator")
#wavegan.discriminator.save(f"{model_path}/discriminator")
print("models saved")
print("complete")

