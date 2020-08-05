
import json
import numpy as np
import random
import torch
import torch.autograd as autograd
import torch.utils.data as data
import Preprocessor as preprocess

class SpeechDataLoader(data.Dataset):

    def __init__(self, data_json, preproc, batch_size):

        data = preproc.read_data_json(data_json)
        self.preproc = preproc

        # a standard trick 
        bucket_diff = 4  
        max_len = max(len(x['text']) for x in data)
        num_buckets = max_len // bucket_diff
        buckets = [[] for _ in range(num_buckets)]
        for d in data:
            bid = min(len(d['text']) // bucket_diff, num_buckets - 1)
            buckets[bid].append(d)

        # sorting of data
        sort_fn = lambda x : (round(x['duration'], 1),
                              len(x['text']))
        for b in buckets:
            b.sort(key=sort_fn)
        data = [d for b in buckets for d in b]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        datum = self.preproc.preprocess(datum["audio"],
                                        datum["text"])
        return datum

def make_loader(dataset_json, preproc,
                batch_size, num_workers=4):
#TODO: define a sampler 
    dataset = SpeechDataLoader(dataset_json, preproc,
                           batch_size)
    loader = data.DataLoader(dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=lambda batch : zip(*batch),
                drop_last=True)
    return loader        