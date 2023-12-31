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
#from model import GAN
from model import GAN
from preprocess import load_raw_audio, load_zebra_finch
import datetime
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
        'num_epochs',
        type=int,
        default=5000,
        help='Epochs'
    )

args = parser.parse_args()

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
#N_TRAIN =1280*2 #from 640
EPOCHS = args.num_epochs
CHECKPOINT_FREQ = 1000
D_OPTIMIZER = optimizers.Adam(learning_rate=LEARNING_RATE, beta_1 = ADAM_BETA_1, beta_2 = ADAM_BETA_2)
G_OPTIMIZER = optimizers.Adam(learning_rate=LEARNING_RATE, beta_1 = ADAM_BETA_1, beta_2 = ADAM_BETA_2)
Q_OPTIMIZER = optimizers.RMSprop(learning_rate = LEARNING_RATE)
N_CATEGORIES = 8
SLICE_LEN = 16384


if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")
  print("Exiting...")
  exit()

def optimizer_type(optimizer):
    return (str(optimizer).split(".")[3]).split(" ")[0]


wavegan = GAN(
    latent_dim = LATENT_DIM,
    discriminator_steps= DISCRIMINATOR_STEPS,
    gp_weight= GP_WEIGHT,
    n_categories= N_CATEGORIES,
    slice_len=SLICE_LEN,
)

#wavegan.compile(
#    d_optimizer = optimizers.Adam(learning_rate=LEARNING_RATE, beta_1 = ADAM_BETA_1, beta_2 = ADAM_BETA_2),
#    g_optimizer = optimizers.Adam(learning_rate=LEARNING_RATE, beta_1 = ADAM_BETA_1, beta_2 = ADAM_BETA_2),
#    q_optimizer= optimizers.RMSprop(learning_rate = LEARNING_RATE)
#)

wavegan.compile(
    d_optimizer = D_OPTIMIZER,
    g_optimizer = G_OPTIMIZER,
    q_optimizer= Q_OPTIMIZER,
)


time = datetime.datetime.now().strftime("%d%m.%H%M")
model_path = f"/mt/home/jdave/onedrive/models_{time}"
os.mkdir(model_path)

#path = "/mt/home/jdave/datasets/sc09/sc09/train"
path = "/mt/home/jdave/onedrive/zebra_finch/"
print(f"Loading data from {path}")
#train_data = load_raw_audio(path, n_train_data= N_TRAIN, model_path= model_path, n_types= N_CATEGORIES)
train_data, N_TRAIN = load_zebra_finch(path, slice_len=SLICE_LEN, model_path= model_path, n_types = N_CATEGORIES)

specs={"Discriminator Steps": DISCRIMINATOR_STEPS,
       "GP Weight": GP_WEIGHT,
       "Latent Dim": LATENT_DIM,
       "N Categories": N_CATEGORIES,
       "Slice Length": SLICE_LEN,
       "Batch Size": BATCH_SIZE,
       "Training Size": int(N_TRAIN),
       "Epochs": EPOCHS,
       "D Optimizer": optimizer_type(D_OPTIMIZER),
       "G Optimizer": optimizer_type(G_OPTIMIZER),
       "Q Optimizer": optimizer_type(Q_OPTIMIZER),
       "Learning Rate": LEARNING_RATE,
       "Phase Parameter": PHASE_PARAM,

       }



spec_path = f"{model_path}/model_specifications.json"
with open(spec_path, 'w') as f:
    json.dump(specs, f)
print("saved model specs")


class Generator_Save_Callback(callbacks.Callback):
    def __init__(self, freq):
        self.freq = freq

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.freq == 0:
            save_path = f"{model_path}/generator{epoch}"
            wavegan.generator.save(save_path)
            print(f"Saved Generator to {save_path}")

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
wavegan.discriminator.save(f"{model_path}/discriminator")
wavegan.auxiliary.save(f"{model_path}/auxiliary")
print("models saved")
print("complete")

