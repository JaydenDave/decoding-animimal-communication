import librosa
import pickle
import numpy as np
import os

class Loader:
    def __init__(self,sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono
    
    def load(self,file_path):
        signal = librosa.load(file_path,
                              sr= self.sample_rate,
                              duration = self.duration,
                              mono = self.mono)[0]
        return signal

class Padder:
    def __init__(self,mode = "constant"):
        self.mode = mode
    
    def left_pad(self, array, num_missing_items):
        padded_array = np.pad(array,
                              (num_missing_items,0),
                              mode = self.mode)
        return padded_array
    
    def right_pad(self, array, num_missing_items):
        padded_array = np.pad(array,
                              (0,num_missing_items),
                              mode = self.mode)
        return padded_array
    
class LogSpectrogramExtractor:
    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self,signal):
        stft = librosa.stft(signal,
                            n_fft= self.frame_size,
                            hop_length=self.hop_length)[:-1]
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram

class MinMaxNormaliser:
    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val
    
    def normalise(self,array):
        norm_array = (array- array.min()) / (array.max()-array.min())
        norm_array =  norm_array*(self.max-self.min) + self.min
        return norm_array
    
    def denormalise(self, norm_array, origional_min, origional_max):
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (origional_max - origional_min) + origional_min
        return array

class Saver:
    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir
    
    def save_feature(self,feature, file_path):
        save_path = self._generate_save_path(file_path)
        np.save(save_path, feature)
    
    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        return save_path
    
    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir, "min_max_values.pkl")
        #save_path = f"{self.min_max_values_save_dir}/min_max_values.pkl"
        #self._save(min_max_values, save_path)
        with open(save_path, "wb") as f:
            pickle.dump(min_max_values, f)
        

    @staticmethod
    def _save(self, data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)


class PreprocessingPipeline:
    def __init__(self):
        self._loader= None
        self.padder = None
        self.extractor = None
        self.normaliser = None
        self.saver = None
        #need to store min max vals for all the log spectrograms (needed for denormalising)
        self.min_max_values = {}
        self._num_expected_samples = None
    
    @property
    def loader(self):
        return self._loader
    
    @loader.setter
    def loader(self,loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)
    
    def process(self, audio_files_dir):
        for file in os.listdir(audio_files_dir):
            file_path = os.path.join(audio_files_dir, file)
            print(file_path)
            self._process_file(file_path)
            print(f"Processed File {file_path}")
        self.saver.save_min_max_values(self.min_max_values)

    def _process_file(self, file_path):
        signal = self.loader.load(file_path)
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        
        feature = self.extractor.extract(signal)
        norm_feature = self.normaliser.normalise(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        self._store_min_max_value(save_path, feature.min(), feature.max())
    
    def _is_padding_necessary(self, signal):
        if len(signal) < self._num_expected_samples:
            return True
        return False
    
    def _apply_padding(self,signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal

    def _store_min_max_value(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = {"min": min_val,
                                          "max": max_val}
    
if __name__ == "__main__":
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 0.74
    SAMPLE_RATE = 44100
    MONO = True

    FILES_DIR = r"C:\Users\Jayde\Desktop\library\bird_songs\data\bird_songs\audio"
    SPECTROGRAM_DIR = r"C:\Users\Jayde\Desktop\library\bird_songs\data\bird_songs\spectrograms"
    MIN_MAX_DIR = r"C:\Users\Jayde\Desktop\library\bird_songs\data\bird_songs"

    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    min_max_normaliser = MinMaxNormaliser(0,1)
    saver = Saver(SPECTROGRAM_DIR, MIN_MAX_DIR)

    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.extractor = log_spectrogram_extractor
    preprocessing_pipeline.normaliser =  min_max_normaliser
    preprocessing_pipeline.saver = saver

    preprocessing_pipeline.process(FILES_DIR)




