import sys

from winfoGAN.utils import create_inputs, avg_fundamental_freq
#sys.path.append("winfoGAN")
from winfoGAN.model import GAN
import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
#from utils import *

parser = argparse.ArgumentParser()

parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='model directory'
    )

parser.add_argument(
        '--epoch',
        type=str,
        default = "",
        help='generator epoch'
    )

parser.add_argument(
        '--num',
        type=int,
        default = 1000,
        help='number to generate for each dose'
    )

parser.add_argument(
        '--baseline',
        type=int,
        default = 0,
        help='baseline_dose'
    )
args = parser.parse_args()
epoch = args.epoch

NUM = args.num
BASELINE_DOSE = args.baseline
#load in model
model_directory =args.model
spec_path = f"{model_directory}/model_specifications.json"
with open(spec_path, 'r') as f:
  specs = json.load(f)

gan = GAN(
    latent_dim = specs["Latent Dim"],
    discriminator_steps= specs["Discriminator Steps"],
    gp_weight= specs["GP Weight"],
    n_categories= specs["N Categories"],
    slice_len=16384,
)

generator = gan.generator




epochs=[]

sr = 24414

dose_vals =np.arange(-1, 13, 0.5)
#generate z part which will stay the same for all

latent_dim = specs["Latent Dim"]
n_cat = specs["N Categories"]
bits = range(n_cat)
z_dim = latent_dim-n_cat
z= np.random.normal(size=(NUM, latent_dim))
z[:,z_dim:]= BASELINE_DOSE

for epoch in ["300","600","1400"]:
    df = pd.DataFrame()
    all_inputs = []
    all_f0 =[]
    doses = []
    bit_vals = []
    fundamental_freqs = []
    f0_stds = []
    generator.load_weights(f"{model_directory}/generator{epoch}")
    epoch = "5000" if epoch =="" else epoch
    for dose in tqdm(dose_vals):
        
        for bit in bits:
            z[:,z_dim:]= BASELINE_DOSE
            z[:,z_dim+bit]=dose
            input= tf.convert_to_tensor(z, dtype=tf.float32)


            all_inputs += list(input.numpy())
            generated_audio = generator.predict(input)
            generated= np.squeeze(generated_audio)
            f0s = avg_fundamental_freq(generated, sr = sr)
            all_f0 += f0s
            f0_avg = np.average(f0s)
            f0_std = np.std(f0s)
            

            doses.append(dose)
            bit_vals.append(bit)
            fundamental_freqs.append(f0_avg)
            f0_stds.append(f0_std)

    col_names = [f"z_{num}" for num in range(z_dim)] + [f"bit_{num}" for num in range(specs["N Categories"])]
    raw_data = pd.DataFrame(all_inputs, columns=col_names)
    raw_data["F0"] = all_f0
    df['bit'] = bit_vals
    df['dose'] = doses
    df['f0'] = fundamental_freqs
    df['f0 std'] = f0_stds
    df.to_csv(f"{model_directory}/Bits{epoch}.csv", index= False)
    raw_data.to_csv(f"{model_directory}/raw_data{epoch}.csv", index= False)
