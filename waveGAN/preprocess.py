import numpy as np
import os
import librosa as lb
import pickle


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

def z_score_normalise(array):
    flattened_array = array.flatten()

    # Calculate the mean and standard deviation of the entire set
    mean_value = np.mean(flattened_array)
    std_deviation = np.std(flattened_array)

    # Z-score normalize each audio clip individually
    normalized_samples = (array - mean_value) / std_deviation
    norm_vals = {"mean": mean_value, "std_dev": std_deviation}
    with open("./normaliser_values", "wb") as f:
            pickle.dump(norm_vals, f)
    return normalized_samples

def load_raw_audio(data_path, folders = False):
    audio = []
    if folders:
        for folder in os.listdir(data_path):
            path = os.path.join(data_path, folder)
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                signal, sr = lb.load(file_path)
                #22050
                signal = set_duration(signal, max = 16384)
                audio.append(signal)
            print(f"Loaded audio from {folder}")
    else:
        for file in os.listdir(data_path):
                file_path = os.path.join(data_path, file)
                signal, sr = lb.load(file_path)
                signal = set_duration(signal, max = 16384)
                audio.append(signal)
    print(f"Loaded audio from {data_path}")

    audio = np.array(audio)
    audio = z_score_normalise(audio)
    print("normalised")
    audio = np.expand_dims(audio, axis=-1)
    return audio



