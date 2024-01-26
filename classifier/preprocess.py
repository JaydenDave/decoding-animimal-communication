import numpy as np
import os
import librosa as lb
import pickle
import random
import pandas as pd
from scipy.signal import butter, lfilter

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


def load_zebra_finch(data_dir,slice_len, model_path, n_types, n_train_data=None, batch_size= 64, equal=False):
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
            recording /= np.max(np.abs(recording))
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
    if equal:
        min_count = min(call_type_counts.head(n_types).values.tolist())

        def sample_from_group(group):
            return group.sample(min_count, replace=True)
        
        # Apply the sampling function to each group
        df_top = df_top.groupby('call_type', group_keys=False).apply(sample_from_group)

        # Reset the index of the resulting DataFrame
        df_top.reset_index(drop=True, inplace=True)
        call_type_counts = df_top["call_type"].value_counts()
        print(call_type_counts)
    
    audio = [centre_and_pad(signal, slice_len) for signal in df_top["rec"]]
    audio = [bandpass_filter(signal,250,12000, sr) for signal in audio]
    audio = np.array(audio)
    n_train_data = len(audio)
    print(f"reduced to {n_train_data} training samlples")
    labels = pd.get_dummies(df_top['call_type']).values

    audio = np.expand_dims(audio, axis=-1)
    return audio, labels

def bandpass_filter(data, low, high, sr, order=5):
    nyquist = sr/2
    low = low/nyquist
    high = high/nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = np.array(lfilter(b, a, data))
    return filtered



