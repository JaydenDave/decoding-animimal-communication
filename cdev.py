import sys

from winfoGAN.utils import create_inputs, avg_fundamental_freq
#sys.path.append("winfoGAN")
from winfoGAN.model import GAN
from classifier.classifier import CLASSIFICATION_MODEL
import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from collections import defaultdict, Counter
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
        default = "1000",
        help='generator epochs to loop through'
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

parser.add_argument(
        '--sr',
        type=int,
        default = 24414,
        help='sampling rate of audio'
    )

parser.add_argument(
        '--classifier',
        type=str,
        default = "/mt/home/jdave/onedrive/classifier_0402.1350",
        help='sampling rate of audio'
    )

args = parser.parse_args()
epochs = [args.epoch]
#epochs = ["400", "600", "1000"]
sr = args.sr
NUM = args.num
BASELINE_DOSE = args.baseline

#load in model
model_directory =args.model
spec_path = f"{model_directory}/model_specifications.json"
with open(spec_path, 'r') as f:
  specs = json.load(f)

classifier_directory =args.classifier
spec_path = f"{classifier_directory}/model_specifications.json"
with open(spec_path, 'r') as f:
  classifier_specs = json.load(f)

gan = GAN(
    latent_dim = specs["Latent Dim"],
    discriminator_steps= specs["Discriminator Steps"],
    gp_weight= specs["GP Weight"],
    n_categories= specs["N Categories"],
    slice_len=16384,
)
classifier = CLASSIFICATION_MODEL(
   n_categories = classifier_specs["N Categories"],
   slice_len = classifier_specs["Slice Length"]
)

classifier = classifier.classifier
classifier.load_weights(f"{classifier_directory}/classifier10")


generator = gan.generator

dose_vals =np.arange(-1, 13, 0.5)

#generate z part which will stay the same for all
latent_dim = specs["Latent Dim"]
n_cat = specs["N Categories"]
bits = range(n_cat)
z_dim = latent_dim-n_cat
z= np.random.normal(size=(NUM, latent_dim))

functions=[avg_fundamental_freq,]
function_labels=["F0"]

for epoch in epochs:
    
    all_inputs = []
    all_data=defaultdict(list)
    
    doses = []
    bit_vals = []
    outputs=defaultdict(list)
    classes = defaultdict(list)
    

    generator.load_weights(f"{model_directory}/generator{epoch}")
    epoch = "5000" if epoch =="" else epoch

    #generate baseline for ate
    z[:,z_dim:]= BASELINE_DOSE
    input= tf.convert_to_tensor(z, dtype=tf.float32)
    generated_audio = generator.predict(input)
    generated= np.squeeze(generated_audio)
    baselines=[]

    for function in functions:
        results= function(generated, sr = sr)
        baselines.append(results)

    for dose in tqdm(dose_vals):
        
        for bit in bits:
            z[:,z_dim:]= BASELINE_DOSE
            z[:,z_dim+bit]=dose
            input= tf.convert_to_tensor(z, dtype=tf.float32)

            generated_audio = generator.predict(input)
            generated= np.squeeze(generated_audio)

            doses.append(dose)
            bit_vals.append(bit)
            all_inputs += list(input.numpy())

            #classifier
            pred_labels = np.argmax(classifier.predict(generated), axis=1)
            count =Counter(pred_labels)
            for key in range(4):
                if count[key]:
                    frac= count[key]/NUM
                else:
                    frac =0
                outputs[f"cls_{key}"].append(frac)

            #apply acoustic proprty finding algorithms to generated samples
            for function, name, baseline in zip(functions, function_labels, baselines):
                results= function(generated, sr = sr)
                all_data[name] += results
                ates= [x-y for x,y in zip(results, baseline)]
                
                outputs[f"{name}_avg"].append(np.average(ates))
                outputs[f"{name}_std"].append(np.std(ates))

    col_names = [f"z_{num}" for num in range(z_dim)] + [f"bit_{num}" for num in range(specs["N Categories"])]
    raw_data = pd.DataFrame(all_inputs, columns=col_names)
    raw_data_results = pd.DataFrame(all_data)
    raw_data = pd.concat([raw_data,raw_data_results], axis=1)

    df = pd.DataFrame({"bit":bit_vals, "dose":doses})
    results = pd.DataFrame(outputs)
    df = pd.concat([df,results], axis=1)

    df.to_csv(f"{model_directory}/Bits{epoch}.csv", index= False)
    raw_data.to_csv(f"{model_directory}/raw_data{epoch}.csv", index= False)
