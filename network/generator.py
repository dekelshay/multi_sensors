import torch
from torch.utils import data
import numpy as np
import torch.nn.functional as F
from torch.nn import Conv2d, Linear,MaxPool2d, BatchNorm2d, Dropout, init
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage.transform import resize,rotate
import copy
from network.nets import Model
import cv2
import math
import albumentations as A
from torchvision.transforms import transforms
#from network import transforms
from network.positional_encodings  import PositionalEncoding2D
from network.spp_layer import spatial_pyramid_pool
from PIL import Image


def NormalizeImages(x):
    #Result = (x/255.0-0.5)/0.5
    Result = x / (255.0/2)
    return Result



def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class DatasetPairwiseTriplets(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, Data, Labels,batch_size, Augmentation, Mode,NegativeMode='Random'):
        'Initialization'
        self.PositiveIdx = np.squeeze(np.asarray(np.where(Labels == 1)));
        self.NegativeIdx = np.squeeze(np.asarray(np.where(Labels == 0)));

        self.PositiveIdxNo = len(self.PositiveIdx)
        self.NegativeIdxNo = len(self.NegativeIdx)

        self.Data   = Data
        self.Labels = Labels

        self.batch_size = batch_size
        self.Augmentation = Augmentation

        self.Mode = Mode
        self.NegativeMode = NegativeMode

        self.ChannelMean1 = Data[:, :, :, 0].mean()
        self.ChannelMean2 = Data[:, :, :, 1].mean()

        self.RowsNo = Data.shape[1]
        self.ColsNo = Data.shape[2]

        self.transform = A.Compose([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(self.RowsNo, self.ColsNo, scale=(0.8, 1.0), ratio=(0.8, 1),interpolation=cv2.INTER_CUBIC,p=0.5),
            A.RandomRotate90(p=1),
            #A.RandomGamma(gamma_limit=136, always_apply=False, p=0.5),
            #A.transforms.RandomBrightnessContrast (brightness_limit=0.5, contrast_limit=0.5, brightness_by_max=True, always_apply=False, p=0.75),
            #A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=0.5),
            #A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.10, rotate_limit=5,always_apply=True, p=1.0),
        ],additional_targets={'image0': 'image'})

    def __len__(self):
        'Denotes the total number of samples'
        return int(self.Data.shape[0]/self.batch_size)

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select pos2 pairs
        if (self.Mode == 'Pairwise') or (self.Mode == 'DomainAdaptation'):
            PosIdx = np.random.randint(self.PositiveIdxNo, size=self.batch_size)
            #print(repr(PosIdx[1:5]))

        if self.Mode == 'Test':
            PosIdx = index


        PosIdx    = self.PositiveIdx[PosIdx]
        PosImages = self.Data[PosIdx, :, :, :].astype(np.float32)/255.0

        # imshow(torchvision.utils.make_grid(PosImages[0,:,:,0]))
        # plt.imshow(np.squeeze(PosImages[2040, :, :, :]));  # plt.show()

        pos1 = PosImages[:, :, :, 0]
        pos2 = PosImages[:, :, :, 1]

        if self.Mode == 'DomainAdaptation':
            if (np.random.uniform(0, 1) > 0.5):
                pos2 =pos1
            else:
                pos1 = pos2



        for i in range(0, PosImages.shape[0]):

            # Flip LR
            if (np.random.uniform(0, 1) > 0.5) & self.Augmentation["HorizontalFlip"]:
                pos1[i,] = np.fliplr(pos1[i,])
                pos2[i,] = np.fliplr(pos2[i,])


            #flip UD
            if (np.random.uniform(0, 1) > 0.5) & self.Augmentation["VerticalFlip"]:
                pos1[i,] = np.flipud(pos1[i,])
                pos2[i,] = np.flipud(pos2[i,])


            # rotate:0, 90, 180,270,
            if self.Augmentation["Rotate90"]:
                idx = np.random.randint(low=0, high=4, size=1)[0]  # choose rotation
                pos1[i,] = np.rot90(pos1[i,], idx)
                pos2[i,] = np.rot90(pos2[i,], idx)



                # random crop
                #if (np.random.uniform(0, 1) > 0.5) & self.Augmentation["RandomCrop"]['Do']:
            if self.Augmentation["RandomCrop"]['Do']:
                dx = np.random.uniform(self.Augmentation["RandomCrop"]['MinDx'],self.Augmentation["RandomCrop"]['MaxDx'])
                dy = np.random.uniform(self.Augmentation["RandomCrop"]['MinDy'],self.Augmentation["RandomCrop"]['MaxDy'])

                dx = dy

                x0 = int(dx * self.ColsNo)
                y0 = int(dy * self.RowsNo)

                # ShowRowImages(pos1[0:1,:,:])
                # plt.imshow(pos1[i,:,:],cmap='gray');plt.show();
                # aa = pos1[i,y0:,x0:]

                pos1[i,] = resize(pos1[i, y0:, x0:], (self.RowsNo, self.ColsNo))

                # ShowRowImages(pos1[0:1, :, :])
                if self.Mode != 'DomainAdaptation':
                    pos2[i,] = resize(pos2[i, y0:, x0:], (self.RowsNo, self.ColsNo))
                else:
                    dx = np.random.uniform(self.Augmentation["RandomCrop"]['MinDx'],self.Augmentation["RandomCrop"]['MaxDx'])
                    dy = np.random.uniform(self.Augmentation["RandomCrop"]['MinDy'],self.Augmentation["RandomCrop"]['MaxDy'])

                    dx = dy

                    x0 = int(dx * self.ColsNo)
                    y0 = int(dy * self.RowsNo)
                    pos2[i,] = resize(pos2[i, y0:, x0:], (self.RowsNo, self.ColsNo))


            #test
            if self.Augmentation["albumentations"]:

                #plt.imshow(pos1[i,:,:],cmap='gray');plt.show();
                if self.Mode != 'DomainAdaptation':

                    #symmetric transform
                    # For np.float32 input, Albumentations expects that value will lie in the range between 0.0 and 1.0.

                    transformed = self.transform(image=pos1[i, :, :], image0=pos2[i, :, :])
                    pos1[i,] = transformed['image']
                    pos2[i,] = transformed['image0']
                else:#'DomainAdaptation'
                    # For np.float32 input, Albumentations expects that value will lie in the range between 0.0 and 1.0.
                    pos1[i,] = self.transform(image=pos1[i,]/255.0)['image']*255.0
                    pos2[i,] = self.transform(image=pos2[i,]/255.0)['image']*255.0


            # plt.imshow(pos1[i,:,:],cmap='gray');plt.show();
            # plt.imshow(pos2[i,:,:],cmap='gray');plt.show();

        pos1*=255
        pos2*=255

        Result = dict()
        Result['pos1']   = NormalizeImages(pos1-self.ChannelMean1)
        Result['pos2']   = NormalizeImages(pos2-self.ChannelMean2)

        return Result


def CreateCorruptedBatch(Size,CorruptionRatio):
    idx1 = np.arange(0, Size)
    idx  = np.random.randint(low=0, high=Size, size=math.floor(CorruptionRatio*Size))[0]
    idx1[idx] = np.random.randint(low=0, high=Size, size=math.floor(CorruptionRatio*Size))[0]


    idx2 = np.arange(0,Size)

    return idx1,idx2



