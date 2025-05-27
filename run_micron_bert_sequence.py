from meb import core, datasets, utils
from meb.models import me_networks
from meb.datasets.sampling import UniformTemporalSubsample
from meb.utils.traditional_methods import tim
from meb.utils.metrics import MultiClassCELoss, MultiClassF1Score

from timm import models

from micron_bert import get_model, load_micron_bert_model


from functools import partial
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

from skimage.transform import resize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision.models.video import MC3_18_Weights, mc3_18
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
# from torchviz import make_dot

from typing import List

action_units = utils.dataset_aus['cross']

emote_dict = {
    'disgust': 0,
    'happiness': 1,
    'repression': 2,
    'surprise': 3,
    'others': 4

}

class Onset_Apex_Dataset(Dataset):
    def __init__(self, df, transform):
        super().__init__()
        # data_root_path = '/home/hq/Documents/data/CASME2/Cropped_original/'
        data_root_path = '/home/hq/Documents/data/CASME2/CASME2_RAW_selected/'
        subject_identifier = 'sub'
        # image_identifier = 'reg_img'
        image_identifier = 'img'
        

        # remove small classes
        df = df.loc[df['emotion'].isin(list(emote_dict.keys()))]

        self.onset_name = data_root_path + subject_identifier + df['subject'].astype(str) + '/' + df['material'] + '/' + image_identifier + df['onset'].astype(str) + '.jpg'
        self.apex_name = data_root_path + subject_identifier + df['subject'].astype(str) + '/' + df['material'] + '/' + image_identifier + df['apex'].astype(str) + '.jpg'


        self.onset = self.onset_name.tolist()
        self.apex = self.apex_name.tolist()

        self.labels = df['emotion'].tolist()
        self.labels = [emote_dict[x] for x in self.labels]
        
        

        self.transform = transform

    
    def __len__(self):
        return len(self.onset)
    
    def __getitem__(self, index):

        x1 = Image.open(self.onset[index]).convert("RGB")
        x2 = Image.open(self.apex[index]).convert("RGB")

        x1_filename = self.onset[index]
        x2_filename = self.apex[index]

        if self.transform:
            x1 = self.transform(x1).cuda()
            x2 = self.transform(x2).cuda()

        labels = self.labels[index]

        return x1, x2, x1_filename, x2_filename, labels

def visualizing_diagonal_attention_map(dataloader):
    attention_path = 'Weights/micron_bert_attention_map_with_crossAttention_ONLY/'


    for x1, x2, x1_filename, x2_filename, labels in dataloader:
        output, feat, diag_attentions = model(x1, x2)
        for i in range(len(diag_attentions)):
            diag_img = diag_attentions[i, :, :, :].cpu()
            diag_img = np.transpose(diag_img, (1, 2, 0))

            filename = attention_path + '/'.join(x1_filename[i].split('/')[-3:])
            if os.path.exists('/'.join(filename.split('/')[:-1])) == False:
                os.makedirs('/'.join(filename.split('/')[:-1]))
            plt.imshow(diag_img)
            plt.colorbar()
            plt.savefig(filename)
            plt.close()


def read_onset_apex(df):
    data_root_path = '/home/hq/Documents/data/CASME2/Cropped_original/'
    subject_identifier = 'sub'
    image_identifier = 'reg_img'

    # # TODO build fold information into this array: output (fold number, number of images, 224, 224, 3)
    # remove small samples labels
    emotes, emotes_count = np.unique(df['emotion'].values, return_counts=True)
    emotes = emotes[emotes_count > 10]
    df = df.loc[df['emotion'].isin(emotes)]

    onset = data_root_path + subject_identifier + df['subject'].astype(str) + '/' + df['material'] + '/' + image_identifier + df['onset'].astype(str) + '.jpg'
    apex = data_root_path + subject_identifier + df['subject'].astype(str) + '/' + df['material'] + '/' + image_identifier + df['apex'].astype(str) + '.jpg'
    labels = df['emotion'].tolist()

    onset_frames = onset.tolist()
    apex_frames = apex.tolist()

    # subject fold for loso
    subject_list = df['subject'].astype(int).values.tolist()

    return onset_frames, apex_frames
    # return onset_frames, apex_frames, labels, subject_list

def read_frame(filepaths: List[str]):
    frames = []
    for path in filepaths:
        image = Image.open(path)
        image = image.resize((224, 224))
        image = np.array(image)
        image = np.expand_dims(image, 0)
        frames += [image]
    frames = np.vstack(frames)

    return frames

class bert_model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = load_micron_bert_model()
        for param in self.model.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(512, num_classes)

        # self.head = nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, x):
        x = self.model.extract_features(x)
        # x = x.transpose(1, 2)
        # x = self.gap(x).squeeze(-1)

        x = x[:, 0, :]
        x = self.fc(x)
        return x

class bert_model_with_attention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = load_micron_bert_model()
        self.model = self.model.cuda()
        self.fc = nn.Linear(512, num_classes).cuda()
    
    def forward(self, x_t, x_t_1):
        x, _, _ = self.model.forward_encoder(x_t, x_t_1)
        x = x[:, 0, :]
        x = self.fc(x)

        # for visualization
        feat, diag_attentions = self.model.get_attmap(x_t, x_t_1)

        return x, feat, diag_attentions 

# CD6ME
# cross_dataset_seq = datasets.CrossDataset(resize=224, optical_flow=False, cropped=True, color=True, magnify=False, num_samples=3)
# cross_dataset_seq = datasets.CrossDataset(resize=224, optical_flow=True, cropped=True, color=False, magnify=False, num_samples=None) # optcal flow, apex only
# cross_dataset_seq = datasets.CrossDataset(resize=224, optical_flow=False, cropped=True, color=True, magnify=False, num_samples=None) # rgb cropped frames
single_dataset = datasets.Casme2(resize=224, optical_flow=False, cropped=False, color=True)

sampler = UniformTemporalSubsample(num_samples=1, temporal_dim=0)

model = bert_model_with_attention(num_classes=5)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet mean and std
])


### visualizing feature map ###
train_dataset = Onset_Apex_Dataset(df=single_dataset.data_frame, transform=transform)
train_data_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
visualizing_diagonal_attention_map(train_data_loader)
###############################


