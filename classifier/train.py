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
from sklearn.model_selection import train_test_split
from classifier import CLASSIFICATION_MODEL
from preprocess import load_zebra_finch
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

args = parser.parse_args()

LEARNING_RATE = 1e-4
ADAM_BETA_1 = 0.5
ADAM_BETA_2 = 0.9
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
CHECKPOINT_FREQ = args.checkpoints
OPTIMIZER = optimizers.Adam(learning_rate=LEARNING_RATE, beta_1 = ADAM_BETA_1, beta_2 = ADAM_BETA_2)
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


model = CLASSIFICATION_MODEL(
   n_categories = N_CATEGORIES,
   slice_len = SLICE_LEN,
)

model.compile(optimizer = OPTIMIZER)

time = datetime.datetime.now().strftime("%d%m.%H%M")
model_path = f"/mt/home/jdave/onedrive/classifier_{time}"
os.mkdir(model_path)

path = "/mt/home/jdave/onedrive/zebra_finch/"
print(f"Loading data from {path}")

data, labels = load_zebra_finch(path, slice_len=SLICE_LEN, model_path= model_path, n_types = 5, batch_size=BATCH_SIZE, equal = True)

X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=7)
n_train = X_train.shape[0]

specs={"N Categories": N_CATEGORIES,
       "Slice Length": SLICE_LEN,
       "Batch Size": BATCH_SIZE,
       "Training Size": int(n_train),
       "Epochs": EPOCHS,
       "Optimizer": optimizer_type(OPTIMIZER),
       "Learning Rate": LEARNING_RATE,
       }



spec_path = f"{model_path}/model_specifications.json"
with open(spec_path, 'w') as f:
    json.dump(specs, f)
print("saved model specs")


class Model_Save_Callback(callbacks.Callback):
    def __init__(self, freq):
        self.freq = freq

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.freq == 0:
            save_path = f"{model_path}/classifier{epoch}"
            model.classifier.save(save_path)
            print(f"Saved Generator to {save_path}")

save_gen_callback= Model_Save_Callback(CHECKPOINT_FREQ)

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

model.fit(
    X_train,
    y_train,
    batch_size = BATCH_SIZE,
    shuffle = True,
    epochs=EPOCHS,
    callbacks=[
        save_gen_callback,
        tensorboard_callback
    ],
)

model.classifier.save(f"{model_path}/classifer{EPOCHS}")

print("models saved")
print("complete")