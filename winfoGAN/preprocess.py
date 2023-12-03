import numpy as np
import os
import librosa as lb
import pickle
import random


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
                print(type)
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



