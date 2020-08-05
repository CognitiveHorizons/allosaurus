from ..Minions.minions import *
from ..Minions.cls_minions import *
from .encoder import encoder
from .lr_scheduler import LR_Scheduler
from ..pase import pase, pase_attention, pase_chunking
from .worker_scheduler import backprop_scheduler
from ...utils import AuxiliarSuperviser, get_grad_norms
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import numpy as np
import random
import os
import pickle
from tqdm import tqdm, trange
try:
    from tensorboardX import SummaryWriter
    use_tb = True
except ImportError:
    print('cannot import Tensorboard, use pickle for logging')
    use_tb = False

def get_grad_norms(model, keys=[]):
    grads = {}
    for i, (k, param) in enumerate(dict(model.named_parameters()).items()):
        accept = False
        for key in keys:
            # match substring in collection of model keys
            if key in k:
                accept = True
                break
        if not accept:
            continue
        if param.grad is None:
            print('WARNING getting grads: {} param grad is None'.format(k))
            continue
        grads[k] = torch.norm(param.grad).cpu().item()
    return grads

class trainer(object):
    def __init__(self,
                 model_cfg=None,
                 cfg=None,
                 pretrained_ckpt=None,
                 tensorboard=None,
                 backprop_mode=None,
                 lr_mode = 'step',
                 name='Allosaurus',
                 device=None):


        self.model = AllosaurusTorchModel(model_config)

        # initialize parameter
        self.epoch = cfg['epoch']
        self.bsize = cfg['batch_size']
        self.save_path = cfg['save_path']
        self.log_freq = cfg['log_freq']
        self.bpe = cfg['bpe']
        self.va_bpe = cfg['va_bpe']
        self.savers = []
        self.model_cfg = model_cfg
        self.cfg = cfg

        self.model_optim = getattr(optim, cfg['opt'])(self.model.parameters(), lr=cfg['lr'])

        self.model_scheduler = LR_Scheduler(lr_mode, lr_step=cfg['lrdec_step'], optim_name="model", base_lr=cfg['lr'],
                                    num_epochs=self.epoch,
                                    iters_per_epoch=self.bpe)

        self.savers.append(Saver(self.model, 
                            self.save_path,
                            max_ckpts=cfg['max_ckpts'],
                            optimizer=self.model_optim, 
                            prefix='Allosaurus-')
                            )

        self.epoch_beg = 0

        # init tensorboard writer
        print("Use tensorboard: {}".format(tensorboard))
        self.tensorboard = tensorboard and use_tb
        if tensorboard and use_tb:
            self.writer = SummaryWriter(self.save_path)
        else:
            self.writer = None
            self.train_losses = {}
            self.valid_losses = {}

    #@profile
    def train_(self, dataloader, valid_dataloader, device):

        print('=' * 50)
        print('Beginning training...')
        print('Batches per epoch: ', self.bpe)
        print('Loss schedule policy: {}'.format(self.backprop.mode))

        if self.cfg["ckpt_continue"]:
            self.resume_training(device)
        else:
            self.epoch_beg = 0

        for e in range(self.epoch_beg, self.epoch):

            self.model.train()

            iterator = iter(dataloader)
            
            with trange(1, self.bpe + 1) as pbar:
                for bidx in pbar:
                    pbar.set_description("Epoch {}/{}".format(e, self.epoch))
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(dataloader)
                        batch = next(iterator)

                    # inference
                    preds = self.model.forward(batch, device)

                    self.model_optim.zero_grad()
                    #TODO: Add proper loss function
                    loss = CTC_loss(preds, labels) + regularization

                    loss.backward()
                    self.model_optim.step()

                    if bidx % self.log_freq == 0 or bidx >= self.bpe:
                        # decrease learning rate
                        lr_model = self.model_scheduler(self.model_optim, bidx, e, loss.item())

                        # print out info
                        self.train_logger(preds, labels, loss, e, bidx, lr_model, pbar)

            self._eval(valid_dataloader, epoch=e, device=device)

            model_path = os.path.join(self.save_path,
                                   'model_e{}.ckpt'.format(e))
            torch.save(self.model.state_dict(), model_path)

            for saver in self.savers:
                saver.save(saver.prefix[:-1], e * self.bpe + bidx)



    def _eval(self, dataloader, epoch=0, device='cpu'):

        self.model.eval()
        with torch.no_grad():
            print('=' * 50)
            print('Beginning evaluation...')
            running_loss = {}
            iterator = iter(dataloader)

            with trange(1, self.va_bpe + 1) as pbar:
                for bidx in pbar:
                    pbar.set_description("Eval: {}/{}".format(bidx, self.va_bpe+1))
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(dataloader)
                        batch = next(iterator)

                    # inference
                    preds = self.model.forward(batch, device)

                    # calculate losses
                    loss = CTC_loss(preds, labels) + regularization
                    
                    if 'total' not in running_loss:
                        running_loss["total"] = [loss.item()]
                    else:
                        running_loss["total"].append(losss.item())

                    if bidx % self.log_freq == 0 or bidx >= self.bpe:
                        pbar.write('-' * 50)
                        pbar.write('EVAL Batch {}/{} (Epoch {}):'.format(bidx,
                                                                    self.va_bpe,
                                                                    epoch))
                        pbar.write('loss: {:.3f}'.format(loss.item()))
                           

            self.eval_logger(running_loss, epoch, pbar)

    def resume_training(self, device):
        giters = 0
        for saver in self.savers:
            # try loading all savers last state if not forbidden is active
            try:
                state = saver.read_latest_checkpoint()
                giter_ = saver.load_ckpt_step(state)
                print('giter_ found: ', giter_)
                # assert all ckpts happened at last same step
                if giters == 0:
                    giters = giter_
                else:
                    assert giters == giter_, giter_
                saver.load_pretrained_ckpt(os.path.join(self.save_path,
                                                        'weights_' + state),
                                           load_last=True)
            except TypeError:
                break

            global_step = giters
            # redefine num epochs depending on where we left it
            self.epoch_beg = int(global_step / self.bpe)

        # self.load_checkpoints(self.save_path)
        self.model.to(device)

    def load_checkpoints(self, load_path):

        # now load each ckpt found
        giters = 0
        for saver in self.savers:
            # try loading all savers last state if not forbidden is active
            try:
                state = saver.read_latest_checkpoint()
                giter_ = saver.load_ckpt_step(state)
                print('giter_ found: ', giter_)
                # assert all ckpts happened at last same step
                if giters == 0:
                    giters = giter_
                else:
                    assert giters == giter_, giter_
                saver.load_pretrained_ckpt(os.path.join(load_path,
                                                        'weights_' + state), 
                                           load_last=True)
            except TypeError:
                break


    def train_logger(self, preds, labels, running_loss, epoch, bidx, lrs, pbar):
        step = epoch * self.bpe + bidx
        pbar.write("=" * 50)
        pbar.write('Batch {}/{} (Epoch {}) step: {}:'.format(bidx, self.bpe, epoch, step))

        pbar.write('%s, learning rate = %.8f, loss = %.4f' % ("total", lr['model'], loss))

        if self.writer:
            self.writer.add_scalar('train/{}_loss'.format("model"), loss.item(), global_step=step)

        grads = get_grad_norms(self.model)
        for kgrad, vgrad in grads.items():
            writer.add_scalar('train/GRAD/{}'.format(kgrad),
                              vgrad, global_step)

        if not self.tensorboard:
            for name, _ in preds.items():
                    preds[name] = preds[name].data
                    labels[name] = labels[name].data

            self.train_losses['itr'] = step
            self.train_losses['loss'] = loss
            self.train_losses['dist'] = preds
            self.train_losses['dist_gt'] = labels

            with open(os.path.join(self.save_path, 'train_losses.pkl'), "wb") as f:
                pbar.write("saved log to {}".format(os.path.join(self.save_path, 'train_losses.pkl')))
                pickle.dump(self.train_losses, f, protocol=pickle.HIGHEST_PROTOCOL)

    def eval_logger(self, running_loss, epoch, pbar):
        pbar.write("=" * 50)
        if self.writer:
            loss = np.mean(running_loss)
            pbar.write("avg loss: {}".format(loss))

            self.writer.add_scalar('eval/{}_loss'.format("model"),
                                    loss,
                                    global_step=epoch)
        else:
            self.valid_losses['epoch'] = epoch
            self.valid_losses['losses'] = running_loss

            with open(os.path.join(self.save_path, 'valid_losses.pkl'), "wb") as f:
                pbar.write("saved log to {}".format(os.path.join(self.save_path, 'valid_losses.pkl')))
                pickle.dump(self.valid_losses, f, protocol=pickle.HIGHEST_PROTOCOL)



class Saver(object):

    def __init__(self, model, save_path, max_ckpts=5, optimizer=None, prefix=''):
        self.model = model
        self.save_path = save_path
        self.ckpt_path = os.path.join(save_path, '{}checkpoints'.format(prefix)) 
        self.max_ckpts = max_ckpts
        self.optimizer = optimizer
        self.prefix = prefix

    def save(self, model_name, step, best_val=False):
        save_path = self.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        ckpt_path = self.ckpt_path
        if os.path.exists(ckpt_path):
            with open(ckpt_path, 'r') as ckpt_f:
                # read latest checkpoints
                ckpts = json.load(ckpt_f)
        else:
            ckpts = {'latest':[], 'current':[]}

        model_path = '{}-{}.ckpt'.format(model_name, step)
        if best_val: 
            model_path = 'best_' + model_path
        model_path = '{}{}'.format(self.prefix, model_path)
        
        # get rid of oldest ckpt, with is the frst one in list
        latest = ckpts['latest']
        if len(latest) > 0:
            todel = latest[0]
            if self.max_ckpts is not None:
                if len(latest) > self.max_ckpts:
                    try:
                        print('Removing old ckpt {}'.format(os.path.join(save_path, 
                                                            'weights_' + todel)))
                        os.remove(os.path.join(save_path, 'weights_' + todel))
                        latest = latest[1:] 
                    except FileNotFoundError:
                        print('ERROR: ckpt is not there?')

        latest += [model_path]

        ckpts['latest'] = latest
        ckpts['current'] = model_path

        with open(ckpt_path, 'w') as ckpt_f:
            ckpt_f.write(json.dumps(ckpts, indent=2))

        st_dict = {'step':step,
                   'state_dict':self.model.state_dict()}

        if self.optimizer is not None: 
            st_dict['optimizer'] = self.optimizer.state_dict()
        # now actually save the model and its weights
        #torch.save(self.model, os.path.join(save_path, model_path))
        torch.save(st_dict, os.path.join(save_path, 
                                          'weights_' + \
                                           model_path))

    def read_latest_checkpoint(self):
        ckpt_path = self.ckpt_path
        print('Reading latest checkpoint from {}...'.format(ckpt_path))
        if not os.path.exists(ckpt_path):
            print('[!] No checkpoint found in {}'.format(self.save_path))
            return None
        else:
            with open(ckpt_path, 'r') as ckpt_f:
                ckpts = json.load(ckpt_f)
            curr_ckpt = ckpts['current'] 
            return curr_ckpt

    def load_weights(self):
        save_path = self.save_path
        curr_ckpt = self.read_latest_checkpoint()
        if curr_ckpt is None:
            print('[!] No weights to be loaded')
            return False
        else:
            st_dict = torch.load(os.path.join(save_path,
                                              'weights_' + \
                                              curr_ckpt))
            if 'state_dict' in st_dict:
                # new saving mode
                model_state = st_dict['state_dict']
                self.model.load_state_dict(model_state)
                if self.optimizer is not None and 'optimizer' in st_dict:
                    self.optimizer.load_state_dict(st_dict['optimizer'])
            else:
                # legacy mode, only model was saved
                self.model.load_state_dict(st_dict)
            print('[*] Loaded weights')
            return True

    def load_ckpt_step(self, curr_ckpt):
        ckpt = torch.load(os.path.join(self.save_path,
                                       'weights_' + \
                                       curr_ckpt),
                          map_location='cpu')
        step = ckpt['step']
        return step

    def load_pretrained_ckpt(self, ckpt_file, load_last=False, load_opt=True,
                             verbose=True):
        model_dict = self.model.state_dict() 
        st_dict = torch.load(ckpt_file, 
                             map_location=lambda storage, loc: storage)
        if 'state_dict' in st_dict:
            pt_dict = st_dict['state_dict']
        else:
            # legacy mode
            pt_dict = st_dict
        all_pt_keys = list(pt_dict.keys())
        if not load_last:
            # Get rid of last layer params (fc output in D)
            allowed_keys = all_pt_keys[:-2]
        else:
            allowed_keys = all_pt_keys[:]
        # Filter unnecessary keys from loaded ones and those not existing
        pt_dict = {k: v for k, v in pt_dict.items() if k in model_dict and \
                   k in allowed_keys and v.size() == model_dict[k].size()}
        if verbose:
            print('Current Model keys: ', len(list(model_dict.keys())))
        if len(pt_dict.keys()) != len(model_dict.keys()):
            raise ValueError('WARNING: LOADING DIFFERENT NUM OF KEYS')
        # overwrite entries in existing dict
        model_dict.update(pt_dict)
        # load the new state dict
        self.model.load_state_dict(model_dict)
        for k in model_dict.keys():
            if k not in allowed_keys:
                print('WARNING: {} weights not loaded from pt ckpt'.format(k))
