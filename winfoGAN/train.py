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
from preprocess import load_raw_audio, load_zebra_finch, load_macaque_data
import datetime
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
        '--epochs',
        type=int,
        default=5000,
        help='Epochs'
    )

parser.add_argument(
        '--cont',
        type=str,
        default=None,
        help='model number to continue training from'
    )
parser.add_argument(
        '--checkpoints',
        type=int,
        default=100,
        help='model checkpint interval'
    )

parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size'
    )

parser.add_argument(
        '--d_steps',
        type=int,
        default=5,
        help='discriminator update steps'
    )

parser.add_argument(
        '--gp_weight',
        type=int,
        default=10,
        help='weighting of gradient penalty term'
    )

parser.add_argument(
        '--n_cat',
        type=int,
        default=5,
        help='number of latent code categories'
    )

parser.add_argument(
        '--equal',
        type=bool,
        default=True,
        help='make the input data the same amount from each call type?'
    )

args = parser.parse_args()

DIM = 64
CHANNELS = 1 #keeping as 1? what for mono or stereo?
PHASE_PARAM = 2
LATENT_DIM = 100
DISCRIMINATOR_STEPS = args.d_steps
GP_WEIGHT = args.gp_weight
LEARNING_RATE = 1e-4
ADAM_BETA_1 = 0.5
ADAM_BETA_2 = 0.9
BATCH_SIZE = args.batch_size
N_TRAIN =2048
EPOCHS = args.epochs
CHECKPOINT_FREQ = args.checkpoints
D_OPTIMIZER = optimizers.Adam(learning_rate=LEARNING_RATE, beta_1 = ADAM_BETA_1, beta_2 = ADAM_BETA_2)
G_OPTIMIZER = optimizers.Adam(learning_rate=LEARNING_RATE, beta_1 = ADAM_BETA_1, beta_2 = ADAM_BETA_2)
Q_OPTIMIZER = optimizers.RMSprop(learning_rate = LEARNING_RATE)
N_CATEGORIES = args.n_cat
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
if args.cont:
    model_directory = f"/mt/home/jdave/onedrive/models_{args.continue_train_number}"
    wavegan.generator.load_weights(f"{model_directory}/generator")
    wavegan.discriminator.load_weights(f"{model_directory}/discriminator")
    wavegan.auxiliary.load_weights(f"{model_directory}/auxiliary")

    


wavegan.compile(
    d_optimizer = D_OPTIMIZER,
    g_optimizer = G_OPTIMIZER,
    q_optimizer= Q_OPTIMIZER,
)


time = datetime.datetime.now().strftime("%d%m.%H%M")
model_path = f"/mt/home/jdave/onedrive/models_{time}"
os.mkdir(model_path)


#path = "/mt/home/jdave/onedrive/zebra_finch/"
path ="/mt/home/jdave/onedrive/macaque/train/"
#path = "/mt/home/jdave/onedrive/sc09/train/"
print(f"Loading data from {path}")
#train_data = load_raw_audio(path, n_train_data= N_TRAIN, model_path= model_path, n_types= 10)
#train_data, N_TRAIN = load_zebra_finch(path, slice_len=SLICE_LEN, model_path= model_path, n_types = 5, batch_size=BATCH_SIZE, equal=args.equal)
train_data,labels = load_macaque_data(path,slice_len= SLICE_LEN, model_path= model_path, batch_size= BATCH_SIZE)
N_TRAIN =3840

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

