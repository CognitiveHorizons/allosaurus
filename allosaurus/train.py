import warnings
warnings.filterwarnings('ignore')

# from dataloader import AllosaurusSpeechDataset
from torch.utils.data import DataLoader
import torch
import pickle
import torch.nn as nn
import numpy as np
import argparse
import os
import json
import random
torch.backends.cudnn.benchmark = True
import torchaudio
import trainer

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def str2None(v):
    if v.lower() in ('none'):
        return None
    return v

def build_dataset_providers(opts):

    # Build Dataset(s) and DataLoader(s)
    # dataset = getattr(pase.dataset, opts.dataset[idx])
    # print ('Dataset name {} and opts {}'.format(dataset, opts.dataset[idx]))
    
    # train_set = AllosaurusSpeechDataset(opts.data_root, 'train')
    # val_set = AllosaurusSpeechDataset(opts.data_root,'valid')
    yesno_data = torchaudio.datasets.YESNO('.', download=True)
    data_loader = torch.utils.data.DataLoader(yesno_data,
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=2)
    return data_loader,data_loader
    # return train_set, val_set

def train(opts):
    CUDA = True if torch.cuda.is_available() and not opts.no_cuda else False
    device = 'cuda' if CUDA else 'cpu'
    num_devices = 1
    np.random.seed(opts.seed)
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if CUDA:
        torch.cuda.manual_seed_all(opts.seed)
        num_devices = torch.cuda.device_count()
        print('[*] Using CUDA {} devices'.format(num_devices))
    else:
        print('[!] Using CPU')
    print('Seeds initialized to {}'.format(opts.seed))

    #Get dataset provider
    train_set, val_set = build_dataset_providers(opts)

    # dloader = DataLoader(train_set, batch_size=opts.batch_size,
    #                      shuffle=True,
    #                      num_workers=opts.num_workers,
    #                      drop_last=True,
    #                      pin_memory=CUDA)
    # testing with deafult dataloade
   
    # # Compute estimation of batches per epoch (bpe). 
    bpe = train_set.len // opts.batch_size
    opts.bpe = bpe
    # if opts.do_eval:
    #     assert val_set is not None, (
    #         "Asked to do validation, but failed to build validation set"
    #     )
    #     va_dloader = DataLoader(val_set, batch_size=opts.batch_size,
    #                             shuffle=True,
    #                             num_workers=opts.num_workers,
    #                             drop_last=True,
    #                             pin_memory=CUDA)
    #     va_bpe = val_set.len // opts.batch_size
    #     opts.va_bpe = va_bpe
    # else:
    #     va_dloader = None

    # ---------------------
    # Build Model
    # load config file for attention blocks

    #Borrow code from recognize.py

    if opts.model_cfg is not None:
        with open(opts.model_cfg, 'r') as model_cfg_f:
            print(model_cfg_f)
            model_cfg = json.load(model_cfg_f)
            print(model_cfg)
    else:
        model_cfg = None

    print(str2bool(opts.tensorboard))
    Trainer = trainer(model_cfg=model_cfg,
                      cfg=vars(opts),
                      backprop_mode=opts.backprop_mode,
                      lr_mode=opts.lr_mode,
                      tensorboard=str2bool(opts.tensorboard),
                      device=device)
    print(Trainer.model)
    print('Model params: ', Trainer.model.describe_params())

    Trainer.model.to(device)

    Trainer.train_(dloader, device=device, valid_dataloader=va_dloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #Dataset specific configuration
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--data_cfg', type=str, default=None) 
    parser.add_argument('--dataset', type=str, default='AllosaurusSpeechDataset')

    #model configuration TODO: Derive from allosaurus code.
    parser.add_argument('--model_cfg', type=str, default=None, help='model-end config')
    
    #model save/checkpoint configurations
    parser.add_argument('--net_ckpt', type=str, default=None,
                        help='Ckpt to initialize the full network '
                             '(Def: None).')
    parser.add_argument('--pretrained_ckpt', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='ckpt')
    parser.add_argument('--max_ckpts', type=int, default=5)

    #general configurations
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--log_freq', type=int, default=100)

    #optimization specific configuration
    parser.add_argument('--opt', type=str, default='Adam')
    parser.add_argument('--lrdec_step', type=int, default=30,
                        help='Number of epochs to scale lr (Def: 30).')
    parser.add_argument('--lrdecay', type=float, default=0,
                        help='Learning rate decay factor with '
                             'cross validation. After patience '
                             'epochs, lr decays this amount in '
                             'all optimizers. ' 
                             'If zero, no decay is applied (Def: 0).')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_mode', type=str, default='step', help='learning rate scheduler mode')

    parser.add_argument('--tensorboard', type=str, default='True', help='use tensorboard for logging')
    parser.add_argument('--no_continue', type=str, default='True', help='use tensorboard for logging')
    parser.add_argument('--backprop_mode', type=str, default='base',help='backprop policy can be choose from: [base, select_one, select_half]')

    opts = parser.parse_args()
    # enforce evaluation for now, no option to disable
    opts.do_eval = True
    opts.ckpt_continue = not str2bool(opts.no_continue)

    #save directory
    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)

    #dump the training parameters onto a file.
    with open(os.path.join(opts.save_path, 'train.opts'), 'w') as opts_f:
        opts_f.write(json.dumps(vars(opts), indent=2))
    train(opts)