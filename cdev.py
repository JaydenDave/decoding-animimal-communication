import sys

from winfoGAN.utils import create_inputs, avg_fundamental_freq, cut_signal
from winfoGAN.preprocess import bandpass_filter
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
import scipy.signal
from scipy.fft import fft, fftfreq, rfft, rfftfreq
import librosa as lb
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

def durations(signals,sr):
    return [len(signal)/sr for signal in signals]

def ZCRs(signals, sr):
    zcrs=[]
    for signal in signals:
        try:
            zcr = lb.feature.zero_crossing_rate(y=signal,frame_length=32, hop_length=16)[0]
            zcrs.append(np.nanmean(zcr))
        except:
            zcrs.append(zcrs[-1])
    return zcrs

def fundamentals(signals,sr):
    avg_f0s=[]
    start_f0s=[]
    end_f0s=[]
    max_f0s=[]
    range_f0s=[]
    f1s=[]

    for signal in signals:
        try:
            f0, voiced_flag, voiced_probs = lb.pyin(signal,
                                                        fmin=200,
                                                        fmax=800,
                                                        sr =sr,
                                                        frame_length = 1024)

            #times = lb.times_like(f0,sr=sr,hop_length=1024//4)

            #find the 2nd peak in the fft for f1
            amplitudes = np.abs(rfft(signal))
            peaks,y =scipy.signal.find_peaks(amplitudes,distance=100)
            n_samples = len(signal)
            frequencies = rfftfreq(n_samples, 1/sr)
            peak_heights = amplitudes[peaks]
            peak_freq_locations = frequencies[peaks]

            #find index of max
            peak_0_index= np.argmax(peak_heights)

            #getting it to look for 2nd highest peak
            peak_heights = np.delete(peak_heights, peak_0_index)
            #needs the +1 because of the one removed from removing the main peak
            second_highest= np.argmax(peak_heights)+1
            f1s.append(peak_freq_locations[second_highest])
            

            f0_clean = [x for x in f0 if not np.isnan(x)]

            avg_f0s.append(np.mean(f0_clean))
            start_f0s.append(f0_clean[0])
            end_f0s.append(f0_clean[-1])
            max_f0=max(f0_clean)
            max_f0s.append(max_f0)
            min_f0 = min(f0_clean)
            range_f0s.append(max_f0 - min_f0)
        except:
            if len(avg_f0s)>len(start_f0s):
                avg_f0s=avg_f0s[:-1]
            if len(f1s)>len(start_f0s):
                f1s=f1s[:-1]
            f1s.append(np.nan)
            avg_f0s.append(np.nan)
            start_f0s.append(np.nan)
            end_f0s.append(np.nan)
            max_f0s.append(np.nan)
            range_f0s.append(np.nan)

    
    return avg_f0s,start_f0s,end_f0s,max_f0s,range_f0s,f1s



for epoch in epochs:
    
    all_inputs = []
    all_data=defaultdict(list)
    
    doses = []
    bit_vals = []
    outputs=defaultdict(list)
    classes = defaultdict(list)
    

    generator.load_weights(f"{model_directory}/generator{epoch}")
    epoch = "10000" if epoch =="" else epoch

    #generate baseline for ate
    z[:,z_dim:]= BASELINE_DOSE
    input= tf.convert_to_tensor(z, dtype=tf.float32)
    generated_audio = generator.predict(input)
    generated= np.squeeze(generated_audio)

    #bandpass filter
    generated =[bandpass_filter(signal,100,8000,sr = sr) for signal in generated]
    #cut audio
    generated=[cut_signal(signal,sr) for signal in generated]

    avg_f0s,start_f0s,end_f0s,max_f0s,range_f0s,f1s= fundamentals(generated,sr)
    baselines={"zcr":ZCRs(generated,sr),
               "dur":durations(generated,sr),
               "f0":avg_f0s,
               "start":start_f0s,
               "end":end_f0s,
               "max":max_f0s,
               "range":range_f0s,
               "f1":f1s}


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

            generated=[cut_signal(signal,sr) for signal in generated]
            #apply acoustic proprty finding algorithms to generated samples
            
            #durations
            results=durations(generated,sr)
            all_data["dur"] +=results
            #treatment effect
            tes= [x-y for x,y in zip(results, baselines["dur"])]

            outputs["dur"].append(np.nanmean(tes))
            outputs["dur_std"].append(np.nanstd(tes))

            #zcrs
            results=ZCRs(generated,sr)
            all_data["zcr"] +=results
            tes=[x-y for x,y in zip(results, baselines["zcr"])]
            outputs["zcr"].append(np.nanmean(tes))
            outputs["zcr_std"].append(np.nanstd(tes))

            #frequencies
            avg_f0s,start_f0s,end_f0s,max_f0s,range_f0s,f1s= fundamentals(generated,sr)
            all_data["f0"] +=avg_f0s
            all_data["start_f0"] +=start_f0s
            all_data["end_f0"] +=end_f0s
            all_data["max_f0"] += max_f0s
            all_data["range_f0"] +=range_f0s
            all_data["f1"] +=f1s

            f0_tes=[x-y for x,y in zip(avg_f0s, baselines["f0"])]
            start_tes=[x-y for x,y in zip(start_f0s, baselines["start"])]
            end_tes=[x-y for x,y in zip(end_f0s, baselines["end"])]
            max_tes=[x-y for x,y in zip(max_f0s, baselines["max"])]
            range_tes=[x-y for x,y in zip(range_f0s, baselines["range"])]
            f1_tes=[x-y for x,y in zip(f1s, baselines["f1"])]

            outputs["f0"].append(np.nanmean(f0_tes))
            outputs["f0_std"].append(np.nanstd(f0_tes))
            outputs["start"].append(np.nanmean(start_tes))
            outputs["start_std"].append(np.nanstd(start_tes))
            outputs["end"].append(np.nanmean(end_tes))
            outputs["end_std"].append(np.nanstd(end_tes))
            outputs["max"].append(np.nanmean(max_tes))
            outputs["max_std"].append(np.nanstd(max_tes))
            outputs["range"].append(np.nanmean(range_tes))
            outputs["range_std"].append(np.nanstd(range_tes))
            outputs["f1"].append(np.nanmean(f1_tes))
            outputs["f1_std"].append(np.nanstd(f1_tes))
            


    col_names = [f"z_{num}" for num in range(z_dim)] + [f"bit_{num}" for num in range(specs["N Categories"])]
    raw_data = pd.DataFrame(all_inputs, columns=col_names)
    for i in all_data.values():
        print(len(i))

    raw_data_results = pd.DataFrame(all_data)
    raw_data = pd.concat([raw_data,raw_data_results], axis=1)

    df = pd.DataFrame({"bit":bit_vals, "dose":doses})
    results = pd.DataFrame(outputs)
    df = pd.concat([df,results], axis=1)

    df.to_csv(f"{model_directory}/Bits{epoch}_{str(BASELINE_DOSE)}.csv", index= False)
    raw_data.to_csv(f"{model_directory}/raw_data{epoch}_{str(BASELINE_DOSE)}.csv", index= False)
