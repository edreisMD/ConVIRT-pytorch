import torch
from models.resnet_clr import ResNetSimCLR
from models.bert_clr import BERTSimCLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from loss.nt_xent import NTXentLoss
import os
import shutil
import sys
from tqdm import tqdm
from transformers import AdamW
import ast
from transformers import BertTokenizer, AutoTokenizer
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys, os

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

import numpy as np

torch.manual_seed(0)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


class SimCLR(object):

    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter()
        self.dataset = dataset
        self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])
        self.truncation = config['model_bert']['truncation']
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_bert']['base_model'], do_lower_case=config['model_bert']['do_lower_case'])

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _step(self, model_res, model_bert, xis, xls, n_iter):

        # get the representations and the projections
        ris, zis = model_res(xis)  # [N,C]

        # get the representations and the projections
        zls = model_bert(xls)  # [N,C]
        # zls = xls
        # normalize projection feature vectors

        loss = self.nt_xent_criterion(zis, zls)
        return loss

    def process_BERT_input():
        return 

    def train(self):
        #Dataloaders
        train_loader, valid_loader = self.dataset.get_data_loaders()

        #Model Resnet Initialize
        model_res = ResNetSimCLR(**self.config["model_res"]).to(self.device)
        model_res = self._load_pre_trained_weights(model_res, 'res')

        optimizer_res = torch.optim.Adam(model_res.parameters(), 
                                        eval(self.config['learning_rate']), 
                                        weight_decay=eval(self.config['weight_decay']))

        scheduler_res = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_res, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)

        # scheduler_res = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_res, 'min', factor=0.5,
        #                                                             patience=5)


        if apex_support and self.config['fp16_precision']:
            model_res, optimizer = amp.initialize(model_res, optimizer_res,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)

        #Model BERT Initialize
        model_bert = BERTSimCLR(**self.config["model_bert"]).to(self.device)
        model_bert = self._load_pre_trained_weights(model_bert, 'bert')

        optimizer_bert = AdamW(model_bert.parameters(),
                      eval(self.config['learning_rate']),    # Default learning rate
                      weight_decay=eval(self.config['weight_decay'])    # Default epsilon value
                      )

        scheduler_bert = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_bert, T_max=len(train_loader), eta_min=0,
                                                                    last_epoch=-1)

        # scheduler_res = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_bert, 'min', factor=0.5,
        #                                                             patience=5)

        if apex_support and self.config['fp16_precision']:
            model_bert, optimizer_bert = amp.initialize(model_bert, optimizer_bert,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)

        #Checkpoint folder
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        print(f'Training...')

        for epoch_counter in range(self.config['epochs']):
            print(f'Epoch {epoch_counter}')
            for xis, xls in tqdm(train_loader):

                optimizer_res.zero_grad()
                optimizer_bert.zero_grad()

                xls = self.tokenizer(list(xls), return_tensors="pt", padding=True, truncation=self.truncation)

                xis = xis.to(self.device)
                xls = xls.to(self.device)

                loss = self._step(model_res, model_bert, xis, xls, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer_res.step()
                optimizer_bert.step()
                n_iter += 1
                
            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model_res, model_bert, valid_loader, n_iter)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model_res.state_dict(), os.path.join(model_checkpoints_folder, 'model_res.pth'))
                    torch.save(model_bert.state_dict(), os.path.join(model_checkpoints_folder, 'model_bert.pth'))
                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler_res.step()
                scheduler_bert.step()
            self.writer.add_scalar('cosine_lr_decay_res', scheduler_res.get_lr()[0], global_step=n_iter)
            self.writer.add_scalar('cosine_lr_decay_bert', scheduler_bert.get_lr()[0], global_step=n_iter)

    def _load_pre_trained_weights(self, model, which_model):
        try:
            checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model_'+which_model+'.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model_res, model_bert, valid_loader, n_iter):

        # validation steps
        with torch.no_grad():
            model_res.eval()
            model_bert.eval()
            valid_loss = 0.0
            counter = 0
            print(f'Validation step')
            for xis, xls in tqdm(valid_loader):

                xls = self.tokenizer(list(xls), return_tensors="pt", padding=True, truncation=self.truncation)

                xis = xis.to(self.device)
                xls = xls.to(self.device)

                loss = self._step(model_res, model_bert, xis, xls, n_iter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model_res.train()
        model_bert.train()
        return valid_loss