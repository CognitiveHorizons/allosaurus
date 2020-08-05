
import json
import numpy as np
import random
import scipy.signal
import soundfile
import pickle
import os
import torch

def array_from_wave(file_name):
    audio, samp_rate = soundfile.read(file_name, dtype='int16')
    return audio, samp_rate

def wav_duration(file_name):
    audio, samp_rate = soundfile.read(file_name, dtype='int16')
    nframes = audio.shape[0]
    duration = nframes / samp_rate
    return duration


def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(window_size * sample_rate / 1e3)
    noverlap = int(step_size * sample_rate / 1e3)
    _, _, spec = scipy.signal.spectrogram(audio,
                    fs=sample_rate,
                    window='hann',
                    nperseg=nperseg,
                    noverlap=noverlap,
                    detrend=False)
    return np.log(spec.T.astype(np.float32) + eps)



def log_specgram_from_file(audio_file):
    audio, sr = array_from_wave(audio_file)
    return log_specgram(audio, sr)

def compute_mean_std(audio_files):
    samples = [log_specgram_from_file(af)
               for af in audio_files]
    samples = np.vstack(samples)
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    return mean, std

class Preprocessor():
    """ This will change if dataset is in a  different format and not in a json """
    END = "</s>"
    START = "<s>"

    def __init__(self, data_json, max_samples=100, start_and_end=True):
        """
        Builds a preprocessor from a dataset.
        Arguments:
            data_json (string): A file containing a json representation
                of each example per line.
            max_samples (int): The maximum number of examples to be used
                in computing summary statistics.
            start_and_end (bool): Include start and end tokens in labels.
        """
        data = self.read_data_json(data_json)

        # Compute data mean, std from sample
        audio_files = [d['audio'] for d in data]
        random.shuffle(audio_files)
        self.mean, self.std = compute_mean_std(audio_files[:max_samples])
        self._input_dim = self.mean.shape[0]

        # Make char map
        chars = list(set(t for d in data for t in d['text']))
        if start_and_end:
            # START must be last so it can easily be
            # excluded in the output classes of a model.
            chars.extend([self.END, self.START])
        self.start_and_end = start_and_end
        self.int_to_char = dict(enumerate(chars))
        self.char_to_int = {v : k for k, v in self.int_to_char.items()}

    def read_data_json(self,data_json):
        with open(data_json) as fid:
            return [json.loads(l) for l in fid]

    def read_pickle_data(self, pickle_file):
        data = pickle.load(open(pickle_file, "rb"))
        return data

    def encode(self, text):
        text = list(text)
        if self.start_and_end:
            text = [self.START] + text + [self.END]
        return [self.char_to_int[t] for t in text]

    def decode(self, seq):
        text = [self.int_to_char[s] for s in seq]
        if not self.start_and_end:
            return text

        s = text[0] == self.START
        e = len(text)
        if text[-1] == self.END:
            e = text.index(self.END)
        return text[s:e]

    def preprocess(self, wave_file, text):
        inputs = log_specgram_from_file(wave_file)
        inputs = (inputs - self.mean) / self.std
        targets = self.encode(text)
        return inputs, targets

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def vocab_size(self):
        return len(self.int_to_char)

class SpeechToIntentPreprocessor(Preprocessor):

    def __init__(self, dataset_dir, left=0, right=0, concatenate_avg=False, use_raw=False):
       
        print ("Reading train.pkl ...") 
        data = self.read_pickle_data(os.path.join(dataset_dir, "train.pkl"))
        print ("Completed reading train.pkl to compute statistics")
        self.left = left
        self.right = right
        self.dataset_dir = dataset_dir
        self.use_raw = use_raw
        self.concatenate_avg = concatenate_avg

        if self.concatenate_avg:
            #store mean and std from the training set.
            file_name = "statistics_concatenate_avg.pkl"
            file_name = os.path.join(self.dataset_dir, file_name)
            if os.path.isfile(file_name):
                self.mean, self.std = pickle.load(open(file_name, "rb"))
            else:
                self.mean, self.std = self.concatenate_avg_statistics(data)
                pickle.dump([self.mean, self.std], open(file_name, "wb"))
        elif self.use_raw:
            #Nothing to be done here.
            self.mean = None
            self.std = None
        else:
            #store mean and std from the training set.
            file_name = "statistics_" + str(left) + "_" + str(right) + ".pkl"
            file_name = os.path.join(self.dataset_dir, file_name)
            if os.path.isfile(file_name):
                self.mean, self.std = pickle.load(open(file_name, "rb"))
            else:
                self.mean, self.std = self.statistics(data)
                pickle.dump([self.mean, self.std], open(file_name, "wb"))

        if not self.use_raw:
            self._input_dim = self.mean.shape[0]
        self._num_classes = len(set(d['class_label'] for d in data))

    def normalize(self, audio_feature):
        #If using raw audio, nothing needs to be done.
        if self.use_raw:
            return audio_feature
        
        #If using pase features, normalize them
        normalized_feature = (audio_feature - self.mean)/self.std
        return normalized_feature

    def concatenate_avg_statistics(self, data):
        print("Computing essential statistics")
        feature_list = []
        for row in data:
            features = row['audio_features']
            features = torch.from_numpy(features).float()
            avg_vect = features.mean(0).repeat(features.shape[0],1)
            features = torch.cat([features,avg_vect],1)
            feature_list.append(features)

        feature_concetenation = np.concatenate(feature_list)

        # feature normalization
        mean = np.mean(feature_concetenation, axis=0)
        std = np.std(feature_concetenation, axis=0)

        print("Finshed computing statistics")
        return mean, std

    def statistics(self, data):
        print("Computing essential statistics")
        feature_list = []
        for row in data:
            feature = torch.from_numpy(self.context_window(row['audio_features'])).float()
            feature_list.append(feature)

        feature_concetenation = np.concatenate(feature_list)
        #feature_concetenation = self.context_window(feature_concetenation)

        # feature normalization
        mean = np.mean(feature_concetenation,axis=0)
        std = np.std(feature_concetenation,axis=0)

        print("Finshed computing statistics")
        return mean, std

    def context_window_item(self, audio_features, idx):

        N_elem = audio_features.shape[0]
        N_features = audio_features.shape[1]
        left = self.left
        right = self.right

        if idx < left:
            pad = left - idx
            input_slice = np.zeros((left + right + 1, N_features))
            input_slice[pad: ] = audio_features[0: idx + right + 1]
        elif N_elem - idx - 1 < right:
            pad = right - N_elem + idx + 1
            input_slice = np.zeros((left + right + 1, N_features))
            input_slice[0:left + right + 1 - pad] = audio_features[idx-left: N_elem]
        else:
            input_slice = audio_features[idx-left: idx+right+1]
            
        audio_output = self.context_window(input_slice)[0]

        return audio_output

    def context_window(self, audio_features):

        N_elem = audio_features.shape[0]
        N_features = audio_features.shape[1]
    
        audio_features_context = np.empty([N_elem, N_features*(self.left + self.right + 1)])
        
        index_feature =0
        for lag in range(-self.left, self.right+1):
            audio_features_context[:,index_feature:index_feature+N_features] = np.roll(audio_features, lag, axis=0)
            index_feature += N_features
        
        audio_features_context = audio_features_context[self.left:audio_features_context.shape[0]-self.right]
    
        return audio_features_context

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def num_classes(self):
        return self._num_classes
