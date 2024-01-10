import numpy as np
import os
import librosa as lb
import pickle
import random
import pandas as pd
from scipy.signal import butter, lfilter


def set_duration(signal, max):
    if len(signal) < max:
        num_missing_samples = max - len(signal)
        padded_array = np.pad(signal,
                              (0,num_missing_samples),
                              mode = "constant")
        return padded_array
    else:
        signal = signal[:max]
    return signal

def centre_and_pad(signal, slice_len= 16384):
    nsamples = len(signal)
    l_pad = (slice_len - nsamples)//2
    r_pad = slice_len - (nsamples + l_pad)
    if len(signal) < slice_len:
        padded_array = np.pad(signal,
                              (l_pad,r_pad),
                              mode = "constant")
        return padded_array
    else:
        signal = signal[:slice_len]
    return signal

def denormalise(samples, normaliser_file_path):
    with open(normaliser_file_path, 'rb') as f:
        norm_vals = pickle.load(f)
    mean = norm_vals["mean"]
    std_dev = norm_vals["std_dev"]
    print(f"DENORMALISING- MEAN:{mean}, STD_DEV:{std_dev}")
    samples = (samples * std_dev) + mean
    print("DENORMALISED")
    return samples


def z_score_normalise(array, save_path = "."):
    flattened_array = array.flatten()

    # Calculate the mean and standard deviation of the entire set
    mean_value = np.mean(flattened_array)
    std_deviation = np.std(flattened_array)

    # Z-score normalize each audio clip individually
    normalized_samples = (array - mean_value) / std_deviation
    norm_vals = {"mean": mean_value, "std_dev": std_deviation}
    with open(f"{save_path}/normaliser_values", "wb") as f:
            pickle.dump(norm_vals, f)
    return normalized_samples

def load_raw_audio(data_path, n_train_data, model_path, n_types,folders = False):
    audio = []
    types = []
    if folders:
        for folder in os.listdir(data_path):
            path = os.path.join(data_path, folder)
            for file in os.listdir(path):  
                file_path = os.path.join(path, file)
                signal, sr = lb.load(file_path, sr= 16000) #loading in at 16KHz sampling rate
                #22050
                signal = set_duration(signal, max = 16384)
                audio.append(signal)
            print(f"Loaded audio from {folder}")
    else:
        for file in os.listdir(data_path):
                type = file.split("_")[0]
                #print(type)
                if type not in types:
                     if len(types) == n_types:
                          continue
                     else:
                          types.append(type)
                if type in types:   
                    file_path = os.path.join(data_path, file)
                    signal, sr = lb.load(file_path, sr= 16000) #loading in at 16KHz sampling rate
                    #22050
                    signal = set_duration(signal, max = 16384)
                    audio.append(signal)
                #file_path = os.path.join(data_path, file)
                #signal, sr = lb.load(file_path)
                #signal = set_duration(signal, max = 16384)
                #audio.append(signal)
    print(f"Loaded audio from {data_path}")

    audio = np.array(audio)
    print(f"obtained {len(audio)} samples")
    print(f"got audio for {types}")
    random.shuffle(audio)
    audio = audio[:n_train_data]
    print(f"reduced to {n_train_data} training samlples")

    audio = z_score_normalise(audio, model_path)
    print("normalised")

    
    audio = np.expand_dims(audio, axis=-1)
    return audio

def load_zebra_finch(data_dir,slice_len, model_path, n_types, n_train_data=None, batch_size= 64):
    recs = []
    #file_names= []
    call_types = []
    #names = []


    for file_name in os.listdir(data_dir):
        recording, sr = lb.load(f'{data_dir}{file_name}', sr = None)
        #print(file)
        try:
            name, file= file_name.split("_")
            date, call_type, rendition= file.split("-")
            call_type = call_type[:2]
            #print(name, date, call_type)
            recs.append(recording)
            #names.append(name)
            #file_names.append(file_name)
            call_types.append(call_type)
        except:
            print(f"skipped {file}")
            continue
    df = pd.DataFrame()
    df["call_type"]= call_types
    df["rec"] = recs
    durations = [x.shape[0] / sr for x in recs]
    df["duration"] = durations
    print(f"obtained {len(os.listdir(data_dir))} samples")
    dur = slice_len/sr
    df= df[df["duration"] <= dur]

    call_type_counts = df["call_type"].value_counts()
    top_n = call_type_counts.head(n_types).index.tolist()
    print(f"top {n_types} call types for duration <= {dur}: {top_n}")
    df_top = df[df["call_type"].isin(top_n)]
    if not n_train_data:
        dur = slice_len /sr
        count = (df_top["duration"]<= dur).sum()
        n_train_data = (count//batch_size) *batch_size
    
    audio = [centre_and_pad(signal, slice_len) for signal in audio]
    audio = [bandpass_filter(signal,250,12000, sr) for signal in df_top["rec"]]
    audio = np.array(audio)
    audio = z_score_normalise(audio, model_path)
    #print("normalised")

    
    audio = np.array(audio)
    
    random.shuffle(audio)
    audio = audio[:n_train_data]
    print(f"reduced to {n_train_data} training samlples")

    

    
    audio = np.expand_dims(audio, axis=-1)
    return audio, n_train_data

def bandpass_filter(data, low, high, sr, order=5):
    nyquist = sr/2
    low = low/nyquist
    high = high/nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = np.array(lfilter(b, a, data))
    return filtered