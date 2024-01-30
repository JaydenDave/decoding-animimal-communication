import sys

from winfoGAN.utils import create_inputs, avg_fundamental_freq
#sys.path.append("winfoGAN")
from winfoGAN.model import GAN
import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
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
EPOCH = args.epoch

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


df = pd.DataFrame()
doses = []
bits = []
epochs=[]
fundamental_freqs = []
f0_stds = []
sr = 16000
all_inputs = []
all_f0 =[]
dose_vals =np.arange(-1, 13, 0.5)
for epoch in ["1000","2000","3000","4000",""]:
    generator.load_weights(f"{model_directory}/generator{epoch}")
    epoch = "5000" if epoch =="" else epoch
    for dose in tqdm(dose_vals):
        inputs = create_inputs(specs, NUM, bit_value = dose, baseline_dose = BASELINE_DOSE)
        for bit,input in enumerate(inputs):
            all_inputs += list(input.numpy())
            generated_audio = generator.predict(input)
            generated= np.squeeze(generated_audio)
            f0s = avg_fundamental_freq(generated, sr = sr)
            all_f0 += f0s
            f0_avg = np.average(f0s)
            f0_std = np.std(f0s)
            

            doses.append(dose)
            bits.append(bit)
            fundamental_freqs.append(f0_avg)
            f0_stds.append(f0_std)
    z_dim = specs["Latent Dim"] - specs["N Categories"]
    col_names = [f"z_{num}" for num in range(z_dim)] + [f"bit_{num}" for num in range(specs["N Categories"])]
    raw_data = pd.DataFrame(all_inputs, columns=col_names)
    raw_data["F0"] = all_f0
    df['bit'] = bits
    df['dose'] = doses
    df['f0'] = fundamental_freqs
    df['f0 std'] = f0_stds
    df.to_csv(f"{model_directory}/Bits{epoch}.csv", index= False)
    raw_data.to_csv(f"{model_directory}/raw_data{epoch}.csv", index= False)
