
import json
import numpy as np
import random
import torch
import torch.autograd as autograd
import torch.utils.data as data
import pickle
import os
import soundfile as sf
import os.path
from tqdm import tqdm
import sys

def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds

def segment_signal(signal):
    wlen = 3200
    wshift = 80
    en_th = 0.3
    smooth_window = 40
    smooth_th_low = 0.25
    smooth_th_high = 0.6
    avoid_sentences_less_that = 24000
    
    beg_fr=[0]
    end_fr=[wlen]
    count_fr=0
    en_fr=[]
    
    while end_fr[count_fr] < signal.shape[0]:
        #print(beg_fr[count_fr])
        #print(end_fr[count_fr])
        signal_seg = signal[beg_fr[count_fr]:end_fr[count_fr]]
        en_fr.append(np.mean(np.abs(signal_seg) ** 1))
        beg_fr.append(beg_fr[count_fr]+wshift)
        end_fr.append(beg_fr[count_fr]+wlen+wshift)
        count_fr = count_fr + 1
    
    en_arr=np.asarray(en_fr)
    mean_en=np.mean(en_arr)
    en_bin=(en_arr > mean_en * en_th).astype(int)
    en_bin_smooth=np.zeros(en_bin.shape)
    
    # smooting the window
    for i in range(count_fr):
        if i + smooth_window > count_fr - 1:
            wlen_smooth = count_fr
        else:
            wlen_smooth = i + smooth_window
            
        en_bin_smooth[i] = np.mean(en_bin[i:wlen_smooth])
      
    en_bin_smooth_new = np.zeros(en_bin.shape)
    
    vad = False
    beg_arr_vad=[]
    end_arr_vad=[]
    
    for i in range(count_fr):
        if vad==False:
            
            if en_bin_smooth[i]>smooth_th_high:
                if i<count_fr-1:
                    vad=True
                    en_bin_smooth_new[i]=1
                    beg_arr_vad.append((beg_fr[i])+wlen)
            else:
                en_bin_smooth_new[i]=0                    

        else:
            if i==count_fr-1:
                end_arr_vad.append(end_fr[i]) 
                break
            if en_bin_smooth[i]<smooth_th_low:
                vad=False
                en_bin_smooth_new[i]=0
                end_arr_vad.append((beg_fr[i])+wlen)
            else:
                en_bin_smooth_new[i]=1 
       
    if len(beg_arr_vad) != len(end_arr_vad):
        print('error')
        sys.exit(0)

    # Writing on buffer
    out_buffer = []
    for i in range(len(beg_arr_vad)):
        #count_seg_tot=count_seg_tot+1
        if end_arr_vad[i] - beg_arr_vad[i] > avoid_sentences_less_that:
            seg = (beg_arr_vad[i], end_arr_vad[i])
            out_buffer.append(seg)
        #else:
        #    count_short = count_short + 1
    return out_buffer


class PairWaveDataset(data.Dataset):
    """ Return paired wavs, one is current wav and the other one is a randomly
        chosen one. This is needed for MI Chunking.
    """

    def __init__(self, dataset_dir, 
                    phase="train", 
                    max_duration=4, 
                    sampling_rate=16000,
                    transform=None,
                    distortion_transforms=None,
                    distortion_probability=0.4,
                    validate_data=False, 
                    seg_split=False):

        self.phase = phase
        self.name = "fluent_intent_pase"

        self.validate_data = validate_data

        #input transformation configuration
        self.transform = transform
        self.distortion_transforms = distortion_transforms
        self.distortion_probability = distortion_probability

        #Audio specific configuration. Change per dataset.
        self.max_duration = max_duration
        self.sampling_rate = sampling_rate
        self.seg_split = seg_split

        self.max_pase_frames = int(self.max_duration * 100)
        self.max_raw_audio_feature_length = int(self.max_duration * self.sampling_rate)

        if self.phase=='train':
            print("Reading train.pkl file")
            # During training phase we only load the training phase images
            # of the training categories (aka base categories).
            self.data, self.labels = self.load_data(0, os.path.join(dataset_dir, 'train.pkl'))
            # print ("WARNING: swapped test for train")

            #print (self.labels)
            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)
            self.labelIds_base = self.labelIds
            self.num_cats_base = len(self.labelIds_base)
            #print (self.data.shape)
        elif self.phase=='val' or self.phase=='test':
            if self.phase=='test':
                # load data that will be used for evaluating the recognition
                # accuracy of the base categories.
                print("Reading test.pkl file")
                data_base = self.load_data(0, os.path.join(dataset_dir, 'train.pkl'))
                # load data that will be use for evaluating the few-shot recogniton
                # accuracy on the novel categories.
                start = len(np.unique(data_base[1]))
                data_novel = self.load_data(start, os.path.join(dataset_dir, 'test.pkl'))
            else: # phase=='val'
                # load data that will be used for evaluating the recognition
                # accuracy of the base categories.
                data_base = self.load_data(0, os.path.join(dataset_dir, 'train.pkl'))
                # load data that will be use for evaluating the few-shot recogniton
                # accuracy on the novel categories.
                data_novel = None
                start = len(np.unique(data_base[1]))
                if os.path.exists(os.path.join(dataset_dir, 'val.pkl')):
                    print("Reading val.pkl file")
                    data_novel = self.load_data(start, os.path.join(dataset_dir, 'val.pkl'))
                else:
                    print("val.pkl file not present, so making test.pkl the validation set")
                    data_novel = self.load_data(start, os.path.join(dataset_dir, 'test.pkl'))

            self.data = np.concatenate([data_base[0], data_novel[0]], axis=0)
            self.labels = data_base[1] + data_novel[1]

            self.label2ind = buildLabelIndex(self.labels)
            self.labelIds = sorted(self.label2ind.keys())
            self.num_cats = len(self.labelIds)

            self.labelIds_base = buildLabelIndex(data_base[1]).keys()
            self.num_cats_base = len(self.labelIds_base)
            self.labelIds_novel = buildLabelIndex(data_novel[1]).keys()
            self.num_cats_novel = len(self.labelIds_novel)
            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert(len(intersection) == 0)
        else:
            raise ValueError('Not valid phase {0}'.format(self.phase))

    def load_data(self, start, pickle_file):

        data = pickle.load(open(pickle_file, "rb"))

        label_dict = {}
        label_dict_rev = {}
        label_no = start + 1
        for d in data:
            label = d['class_label']
            if not label in label_dict:
                label_dict[label] = label_no
                label_dict_rev[label_no] = label
                label_no += 1
        data_lst = []
        labels_lst = []

        if self.phase == "train":
            #If in training phase, randomly shuffle the data when using raw signals.
            random.shuffle(data)  

        for d in data:
            features = d['audio_file']
            data_lst.append(features)
            label_no = label_dict[d['class_label']]
            labels_lst.append(label_no)

        if self.validate_data:
            self.validate(data_lst, labels_lst)

        return  data_lst, labels_lst
        
    def __len__(self):
        return len(self.data)

    def validate(self, data_lst, labels_lst):

        print("Starting Validation")
        for audio_file in tqdm(data_lst):
            _, audio = self.get_raw_audio_features(audio_file)

            pkg = {'raw': audio, 'raw_rand': audio}

            try:
                if self.transform is not None:
                    pkg = self.transform(pkg)
            except:
                print("{} has errors.".format(audio_file))
                continue

            #TODO: @debug this.
            pkg['cchunk'] = pkg['chunk'].squeeze(0)
            # initialize overlap label
            if 'dec_resolution' in pkg:
                pkg['overlap'] = torch.zeros(len(pkg['chunk']) // pkg['dec_resolution']).float()
            else:
                pkg['overlap'] = torch.zeros(len(pkg['chunk'])).float()

            if self.distortion_transforms:
                pkg = self.distortion_transforms(pkg)

            isNan = []
            for key in pkg.keys():
                if type(pkg[key]) == torch.Tensor:
                    if torch.isnan(pkg[key]).any():
                        isNan.append(key)
    
            if len(isNan) != 0:
                print("{} has nan workers: {}".format(audio_file, isNan))


    def get_raw_audio_features(self, audio_file):
        
        audio, samp_rate = sf.read(audio_file)
        nframes = audio.shape[0]
        duration = nframes / samp_rate
        #process audio
        audio = audio.astype(np.float32)
        #audio = audio / np.max(np.abs(audio))

        return duration, audio

    def __getitem__(self, idx):
        
        audio_file = self.data[idx]
        duration, audio = self.get_raw_audio_features(audio_file)

        if self.seg_split:
            splits = segment_signal(audio)
            if len(splits) > 1:
                print("Segmenting audio {} {} {}".format(audio_file, len(splits), duration))
                split = random.choice(splits)
                audio = audio[split[0]:split[1]]

        # create candidate indices for random other wavs without current index
        indices = list(range(self.__len__()))
        indices.remove(idx)
        rindex = random.choice(indices)
        _, r_audio = self.get_raw_audio_features(self.data[rindex])

        pkg = {'raw': audio, 'raw_rand': r_audio}

        try:
            if self.transform is not None:
                pkg = self.transform(pkg)
        except:
            print("Trying again to find good segment {}".format(audio_file))
            return self.__getitem__(idx)

        #TODO: @debug this.
        pkg['cchunk'] = pkg['chunk'].squeeze(0)
        # initialize overlap label
        if 'dec_resolution' in pkg:
            pkg['overlap'] = torch.zeros(len(pkg['chunk']) // pkg['dec_resolution']).float()
        else:
            pkg['overlap'] = torch.zeros(len(pkg['chunk'])).float()

        if self.distortion_transforms:
            pkg = self.distortion_transforms(pkg)

        class_label = self.labels[idx]

        return pkg, class_label
