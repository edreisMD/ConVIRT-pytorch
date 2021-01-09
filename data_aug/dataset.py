from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
from sentence_transformers import SentenceTransformer, util
import pickle
import os
import time


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# plt.ion()   # interactive mode

class ClrDataset(Dataset):
    """Contrastive Learning Representations Dataset."""

    def __init__(self, csv_file, img_root_dir, emb_root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.clr_frame = pd.read_csv(csv_file)
        self.img_root_dir = img_root_dir
        self.emb_root_dir = emb_root_dir
        self.transform = transform

    def __len__(self):
        return len(self.clr_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_root_dir,
                                self.clr_frame.iloc[idx, 0])
        image = Image.open(img_name)
        image = image.resize((512, 512), Image.ANTIALIAS)
        image = image.convert('RGB')
        
        laudo = self.clr_frame.iloc[idx, 4]
        ls_laudo = laudo.split('.')
        phrase = random.choice(ls_laudo)
        
        sample = {'image': image, 'phrase': phrase}

        if self.transform:
            sample = self.transform(sample)

        return sample