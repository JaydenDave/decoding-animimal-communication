import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import librosa as lb
import IPython
import pandas as pd
import json

import scipy.signal
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy.signal import spectrogram
from preprocess import denormalise


def create_inputs(specs, num, bit_value= 1, baseline_dose = 0, random = False):
    latent_dim = specs["Latent Dim"]
    n_cat = specs["N Categories"]
    bits = range(n_cat)
    z_dim = latent_dim-n_cat

    if random:
        print("Random c")
        z= tf.random.normal(shape=(num, z_dim))
        c = tf.random.normal(shape=(num, n_cat))
        #thresholding values to 0 and 1
        c = tf.cast(c < 0, dtype=tf.float32)
        inputs = [tf.concat([z,c], axis=1)]

    else:
        inputs = []
        for bit in bits:
            z= np.random.normal(size=(num, latent_dim))
            z[:,z_dim:]= baseline_dose
            z[:,z_dim+bit]=bit_value
            z= tf.convert_to_tensor(z, dtype=tf.float32)
            inputs.append(z)


    return inputs

def create_plots(signal, sr, window_length = 1024, save_dir = None, lw=0.6, fmax=6e3, font_size = 13, amp_envelope= True, plot_rms = True):
    fmax = sr/2 #nyquist frequency 
    N = len(signal)
    delta_t = 1 / sr
    times = np.arange(0, N) / sr
    signalf = fft(signal)
    freqs = np.linspace(0.0, 1.0/(2.0*delta_t), N//2)
    
    fig, axs = plt.subplots(1,3, figsize=(20,5))
    axs[0].plot(times, signal, linewidth=lw)
    #axs[0].librosa.display.waveshow(code_1[1], sr=sr)
    axs[0].set_xlabel('Time (s)', fontsize= font_size)
    axs[0].set_ylabel('Amplitude', fontsize = font_size)
    t_ae, ae = amplitude_envelope(signal,frame_size = 128, hop_length= 64, sr= sr)
    t_rms, rms = RMS_energy(signal,frame_size = 128, hop_length= 64, sr= sr)
    axs[0].plot(t_ae,ae, color = "r") if amp_envelope else None
    axs[0].plot(t_rms,rms, color = "r") if plot_rms else None
    axs[0].set_xlim(0,max(times))
    #axs[0].set_title('Time Domain Representation')
    #axs[0].xticks(fontsize = font_size)
    #axs[0].yticks(fontsize= font_size)
    
    axs[1].plot(freqs, 2.0/N * np.abs(signalf[0:N//2]), linewidth=0.4)
    axs[1].set_xlabel('Frequency (Hz)', fontsize= font_size)
    axs[1].set_ylabel('Amplitude', fontsize= font_size)
    #axs[1].set_title('Frequency Domain Representation')
    axs[1].set_xlim([0, fmax])
    #axs[1].xticks(fontsize = font_size)
    #axs[1].yticks(fontsize= font_size)


    f_bins, t_bins, Sxx = spectrogram(signal, fs=sr,
                                 window='hann', nperseg=window_length,
                                 noverlap=window_length-64, detrend=False,
                                 scaling='spectrum')
    im = axs[2].pcolormesh(t_bins, f_bins, 20*np.log10(Sxx+1e-100), cmap='magma')

    axs[2].set_xlabel('Time (s)', fontsize= font_size)
    axs[2].set_ylabel('Frequency (Hz)', fontsize= font_size)
    #axs[2].set_xlim(0,1)
    #axs[2].xticks(fontsize = font_size)
    #axs[2].yticks(fontsize= font_size)
    #axs[2].set_title(f'STFT with nperseg={window_length}')

    # Add colorbar
    cbar = fig.colorbar(im, ax=axs[2])
    cbar.set_label('Amplitude (dB)',fontsize= font_size)
    for ax in axs:
        ax.tick_params(axis='both', labelsize=font_size-2)
    
    if save_dir:
        plt.savefig(save_dir, dpi= 300, bbox_inches = "tight", transparent = True)
    plt.show()

def window_lengths(signal,sr):
    npersegs = [64, 256,512, 1024,768, 4096, 16384]
    fig, axs = plt.subplots(1, len(npersegs), sharey=True, figsize=(20,5))
    axs[0].set_ylabel('Frequency (Hz)')

    for i, window_length in enumerate(npersegs):
        f_bins, t_bins, Sxx = spectrogram(signal, fs=sr,
                                    window='hann', nperseg=window_length,
                                    noverlap=window_length-64, detrend=False,
                                    scaling='spectrum')
        axs[i].pcolormesh(t_bins, f_bins, 20*np.log10(Sxx+1e-100), cmap='magma')
        axs[i].set_xlabel('Time(s)')
        axs[i].set_title(f'STFT with nperseg={window_length}')
    plt.show()

def create_sprectrogram(signal, sr, n_overlap= None, window_length=1024):
    if n_overlap ==None:
        n_overlap = window_length-64
    f_bins, t_bins, Sxx = spectrogram(signal, fs=sr,
                                 window='hann', nperseg=window_length,
                                 noverlap=n_overlap, detrend=False,
                                 scaling='spectrum')
    plt.pcolormesh(t_bins, f_bins, 20*np.log10(Sxx+1e-100), cmap='magma')
    plt.colorbar()
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    #plt.ylim([0, 7500])
    plt.show()

def create_fft(signal,sr, font_size = 13):
    n_samples = len(signal)
    nyquist_freq = sr/2 #max freq
    #using rfft as it only computes the useful half of the fft to speed up processing
    amplitudes = np.abs(rfft(signal))
    frequencies = rfftfreq(n_samples, 1/sr)
    fundamental_freq_index = np.argmax(amplitudes)
    fundamental_freq= frequencies[fundamental_freq_index]
    print(f"Fundamental Frequency- {fundamental_freq} Hz")
    plt.plot(frequencies, amplitudes, linewidth=0.4)
    #plot vertical line
    plt.axvline(x = fundamental_freq, color = 'red', linestyle= '--', label = "Fundamental Frequency", linewidth = 0.1)
    plt.xlabel('Frequency (Hz)', fontsize= font_size)
    plt.ylabel('Amplitude', fontsize= font_size)
    plt.xlim(0,nyquist_freq)
    plt.legend()
    plt.show()


def generate_samples(generator, input, model_directory):
    generated_audio = generator.predict(input) 
    generated_audio= np.squeeze(generated_audio)
    audio = denormalise(generated_audio,f"{model_directory}/normaliser_values")
    return audio

def avg_fundamental_freq(inputs, sr):
    freqs = []
    for signal in inputs:
        n_samples = len(signal)
        amplitudes = np.abs(rfft(signal))
        frequencies = rfftfreq(n_samples, 1/sr)
        fundamental_freq_index = np.argmax(amplitudes)
        fundamental_freq= frequencies[fundamental_freq_index]
        freqs.append(fundamental_freq)
    avg= np.average(freqs)
    std = np.std(freqs)
    return avg,std



def amplitude_envelope(signal, frame_size, hop_length, sr):
    #get max amplitude value for each frame,sliding with hop_length
    ae = np.array([max(signal[i:i+frame_size]) for i in range(0, signal.size, hop_length)])
    frames = range(0,ae.size)
    t = lb.frames_to_time(frames, hop_length= hop_length, sr = sr)
    return t,ae

def RMS_energy(signal, frame_size, hop_length, sr):
    rms = np.array([np.sqrt(sum(k**2 for k in signal[i:i+frame_size]/frame_size)) for i in range(0, signal.size, hop_length)])
    frames = range(0,rms.size)
    t = lb.frames_to_time(frames, hop_length= hop_length, sr = sr)
    return t,rms