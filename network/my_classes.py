import torch
from torch.utils import data
import numpy as np
import torch.nn.functional as F
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
from network.positional_encodings  import PositionalEncoding2D
from network.spp_layer import spatial_pyramid_pool
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from network.losses import AngularPenaltySMLoss
from copy import deepcopy
from collections import OrderedDict


def NormalizeImages(x):
    #Result = (x/255.0-0.5)/0.5
    Result = x / (255.0/2)
    return Result




def FPR95Accuracy(Dist,Labels,FPR = 0.95):
    PosIdx = np.squeeze(np.asarray(np.where(Labels == 1)))
    NegIdx = np.squeeze(np.asarray(np.where(Labels == 0)))

    NegDist = Dist[NegIdx]
    PosDist = np.sort(Dist[PosIdx])

    Val = PosDist[int(FPR * PosDist.shape[0])]

    FalsePos = sum(NegDist < Val);

    FPR95Accuracy = FalsePos / float(NegDist.shape[0])

    return FPR95Accuracy,Val



def FPR95Threshold(PosDist):

    PosDist = PosDist.sort(dim=-1, descending=False)[0]
    Val = PosDist[int(0.95 * PosDist.shape[0])]

    return Val





def ShowRowImages(image_data):
    fig = plt.figure(figsize=(1,image_data.shape[0]))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(1,image_data.shape[0]),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    #for ax, im in zip(grid, image_data):
    for ax, im in zip(grid, image_data):
        # Iterating over the grid returns the Axes.
        ax.imshow(im,cmap='gray')
    plt.show()





def ShowTwoRowImages(image_data1,image_data2):
    fig = plt.figure(figsize=(2, image_data1.shape[0]))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(2,image_data1.shape[0]),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    #for ax, im in zip(grid, image_data):
    for ax, im in zip(grid, image_data1):
        # Iterating over the grid returns the Axes.
        if im.shape[0]==1:
            ax.imshow(im,cmap='gray')
        if im.shape[0]==3:
            ax.imshow(im)

    for i in range(image_data2.shape[0]):
        # Iterating over the grid returns the Axes.
        if im.shape[0] == 1:
            grid[i+image_data1.shape[0]].imshow(image_data2[i],cmap='gray')
        if im.shape[0] == 3:
            grid[i + image_data1.shape[0]].imshow(image_data2[i])
    plt.show()





def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x




def Prepare2DPosEncoding1(PosEncodingX, PosEncodingY, SkipY, SkipX,RowNo,ColNo):

    #PosEncodingX = PosEncodingX[0:ColNo].unsqueeze(0)#x=[1,..,20]
    PosEncodingX = PosEncodingX[0:ColNo:SkipX, :].unsqueeze(1)  # x=[1,..,20]

    for i in range(0,RowNo,SkipY):

        CurrentY = PosEncodingY[i, :].unsqueeze(0).unsqueeze(0).repeat(PosEncodingX.shape[0], 1, 1)

        if i == 0:
            PosEncoding2D = torch.cat((PosEncodingX, CurrentY), 2)
        else:
            CurrentPosEncoding2D = torch.cat((PosEncodingX, CurrentY), 2)

            PosEncoding2D = torch.cat((PosEncoding2D, CurrentPosEncoding2D), 0)

    return PosEncoding2D



def Prepare2DPosEncoding(PosEncodingX, PosEncodingY, RowNo, ColNo):

    #PosEncodingX = PosEncodingX[0:ColNo].unsqueeze(0)#x=[1,..,20]
    PosEncodingX = PosEncodingX[0:ColNo, :].unsqueeze(1)  # x=[1,..,20]

    for i in range(RowNo):

        CurrentY = PosEncodingY[i, :].unsqueeze(0).unsqueeze(0).repeat(ColNo, 1, 1)

        if i == 0:
            PosEncoding2D = torch.cat((PosEncodingX, CurrentY), 2)
        else:
            CurrentPosEncoding2D = torch.cat((PosEncodingX, CurrentY), 2)

            PosEncoding2D = torch.cat((PosEncoding2D, CurrentPosEncoding2D), 0)

    return PosEncoding2D





class MetricLearningCnn(nn.Module):
    #def __init__(self):
     #   super(SiameseTripletCnn, self).__init__()


    def __init__(self,Mode,DropoutP=0):
        super(MetricLearningCnn, self).__init__()

        self.Mode   = Mode

        K =128

        self.AngularLoss = AngularPenaltySMLoss(in_features=K, out_features=2,
                                                 loss_type='cosface', s=0.7, m=1.5)  # loss_type in ['arcface', 'sphereface', 'cosface']

        if (self.Mode =='AsymmetricEncoder') or (self.Mode == 'SymmetricEncoder') \
           or (self.Mode == 'DualEncoder') or (self.Mode == 'DualEncoderHard'):
            self.AttenAS1 = AttentionEmbeddingCNN(K,DropoutP, EmbeddingMaxDim=8)


        if (self.Mode == 'AsymmetricEncoder'):
            self.AttenAS2 = AttentionEmbeddingCNN(K,DropoutP, EmbeddingMaxDim=8)


        if (self.Mode =='AsymmetricEncoder') or (self.Mode == 'SymmetricEncoder') \
            or (self.Mode == 'PairwiseSymmetric') or (self.Mode == 'DualEncoder') \
            or (self.Mode == 'DualEncoderHard')or (self.Mode == 'PairwiseAsymmetric'):
            self.netAS1 = Model(DropoutP)
            #self.netAS2 = Model(DropoutP)

            #self.netAS1G = Model(DropoutP)
            #self.netAS2G = Model(DropoutP)

        if (self.Mode =='AsymmetricEncoder') or (self.Mode == 'AsymmetricEncoder') or (self.Mode == 'PairwiseAsymmetric'):
            self.netAS2 = Model(DropoutP)


        if (Mode == 'Hybrid'):
            self.fc1 = Linear(2*K, K)
            self.fc2 = Linear(2*K, K)
            self.fc2 = copy.deepcopy(self.fc1)

        if (Mode == 'PairwiseSymmetric') or (Mode == 'PairwiseAsymmetric'):
            a=4
            self.fcMeta1A = nn.Linear(8192, K)
            #self.fcMeta1AG = nn.Linear(K, K)
            #self.fcMeta2AG = nn.Linear(K, K)

            self.act1 = nn.GELU()
            self.act2 = nn.GELU()

        if (Mode == 'Hybrid') or (Mode == 'PairwiseAsymmetric'):
            self.fcMeta2A = nn.Linear(8192,K)



            self.fc1A = nn.Linear(K, K)
            self.fc2A = copy.deepcopy(self.fc1A)

            self.act3 = nn.GELU()
            self.act4 = nn.GELU()


        self.Dropout1 = nn.Dropout(DropoutP)
        self.Dropout2 = nn.Dropout(DropoutP)

        self.BCEloss = nn.BCEWithLogitsLoss(reduction='none')

    def FreezeCnns(self, OnOff):
        self.netAS1.FreezeCnn(OnOff)
        try:
            self.netAS2.FreezeCnn(OnOff)
        except:
            aa= 0


    #output CNN
    #@torch.cuda.amp.autocast()
    def forward(self,S1A, S1B=0,Mode = -1,margin=0,UseSE=False):

        if (S1A.nelement() == 0):
            return 0

        if Mode == -1:
            Mode = self.Mode


        if Mode == 'SymmetricEncoder':
            # compute CNNs
            _, ActivMap1 = self.netAS1(S1A, ActivMode=True,UseSE=UseSE)
            _, ActivMap2 = self.netAS1(S1B, ActivMode=True,UseSE=UseSE)

            #ActivMap1 = self.Dropout1(ActivMap1)
            #ActivMap2 = self.Dropout2(ActivMap2)

            # apply different attentions
            output1 = self.AttenAS1(ActivMap1,'SymmetricEncoder')
            output2 = self.AttenAS1(ActivMap2,'SymmetricEncoder')

            Result = dict()
            Result['Emb1'] = output1
            Result['Emb2'] = output2

            return Result

        if Mode == 'AsymmetricEncoder':

            #compute CNNs
            _, ActivMap1 = self.netAS1(S1A, ActivMode=True,UseSE=UseSE)
            _, ActivMap2 = self.netAS1(S1B, ActivMode=True,UseSE=UseSE)


            #apply different attentions
            output1 = self.AttenAS1(ActivMap1,'AsymmetricEncoder')
            output2 = self.AttenAS2(ActivMap2,'AsymmetricEncoder')

            Result = dict()
            Result['Emb1'] = output1
            Result['Emb2'] = output2

            return Result



        if Mode == 'DualEncoderHard':

            Loss = self.AngularLoss

            # compute CNNs
            _, ActivMap1 = self.netAS1(S1A, ActivMode=True)
            _, ActivMap2 = self.netAS1(S1B, ActivMode=True)

            #positive samples
            with torch.no_grad():
                PosScore = self.AttenAS1(ActivMap1, Mode, ActivMap2).squeeze()
                PosScore = self.AngularLoss(PosScore, Mode='logits')[:,1]
                pos_idx = torch.where(PosScore < margin)[0]


            if pos_idx.size()[0] > 0:
                    Pos1  = self.AttenAS1(ActivMap1[pos_idx,], Mode, ActivMap2[pos_idx,]).squeeze()
                    if pos_idx.size()[0]>1:
                        losses1 = Loss(Pos1,torch.ones(Pos1.shape[0],dtype=torch.long,device=ActivMap1.device))[1]#.mean()
                    else:
                        Pos1    = Pos1.view(1, Pos1.shape[0]).repeat(2, 1)
                        losses1 = Loss(Pos1,torch.ones(2,dtype=torch.long,device=ActivMap1.device).squeeze())[1]#.mean()
                        losses1 = losses1[1].view(1)
            else:
                losses1 = torch.empty(0,dtype=torch.float,device=ActivMap1.device)


            #negative1
            with torch.no_grad():
                NegScore = -1e5*torch.ones(ActivMap2.shape[0],device=ActivMap1.device)
                NegIdx   = -torch.ones(ActivMap2.shape[0],dtype=torch.long,device=ActivMap1.device)
                Idx = torch.arange(0,ActivMap1.shape[0],device=ActivMap1.device)
                for i in range(1,ActivMap1.shape[0]):

                    #create shifted idx
                    RolledIdx = torch.roll(Idx, shifts=i, dims=0)
                    CurrNegScore   = self.AttenAS1(ActivMap1, Mode, ActivMap2[RolledIdx,]).squeeze()

                    #get POSITIVE score
                    CurrNegScore   = self.AngularLoss(CurrNegScore, Mode='logits')[:, 1]

                    idx           = torch.where(CurrNegScore>NegScore)[0]
                    NegScore[idx] = CurrNegScore[idx]

                    #store best negative idx
                    NegIdx[idx]   = RolledIdx[idx]

                #losses1 = F.relu(Neg1 - Pos + margin)
                neg_idx = torch.where(NegScore > -margin)[0]

            if neg_idx.size()[0] > 0:
                Neg = self.AttenAS1(ActivMap1[neg_idx,], Mode, ActivMap2[NegIdx[neg_idx],]).squeeze()
                if neg_idx.size()[0] > 1:
                    Score,losses2 = Loss(Neg, torch.zeros(Neg.shape[0],dtype=torch.long, device=ActivMap1.device))#.mean()
                else:
                    Neg = Neg.view(1, Neg.shape[0]).repeat(2, 1)
                    Score,losses2 = Loss(Neg, torch.zeros(2,dtype=torch.long, device=ActivMap1.device).squeeze())#.mean()
                    losses2 = losses2[1].view(1)
            else:
                losses2 = torch.empty(0,dtype=torch.float, device=ActivMap1.device)



            #negative2
            with torch.no_grad():
                NegScore = -1e5 * torch.ones(ActivMap2.shape[0], device=ActivMap1.device)
                NegIdx = -torch.ones(ActivMap2.shape[0], dtype=torch.long, device=ActivMap1.device)
                Idx = torch.arange(0, ActivMap1.shape[0], device=ActivMap1.device)
                for i in range(1, ActivMap1.shape[0]):

                    # create shifted idx
                    RolledIdx = torch.roll(Idx, shifts=i, dims=0)
                    CurrNegScore = self.AttenAS1(ActivMap1[RolledIdx,], Mode, ActivMap2).squeeze()

                    # get POSITIVE score
                    CurrNegScore = self.AngularLoss(CurrNegScore, Mode='logits')[:, 1]

                    idx = torch.where(CurrNegScore > NegScore)[0]
                    NegScore[idx] = CurrNegScore[idx]

                    # store best negative idx
                    NegIdx[idx] = RolledIdx[idx]

                neg_idx = torch.where(NegScore > -margin)[0]


            if neg_idx.size()[0] > 0:
                Neg      = self.AttenAS1(ActivMap1[NegIdx[neg_idx],], Mode, ActivMap2[neg_idx,]).squeeze()

                if neg_idx.size()[0] > 1:
                    Score,losses3 = Loss(Neg, torch.zeros(Neg.shape[0],dtype=torch.long, device=ActivMap1.device))#.mean()
                else:
                    Neg = Neg.view(1, Neg.shape[0]).repeat(2, 1)
                    Score,losses3 = Loss(Neg, torch.zeros(2,dtype=torch.long, device=ActivMap1.device).squeeze())#.mean()
                    losses3 = losses3[1].view(1)
            else:
                losses3 = torch.empty(0,dtype=torch.float, device=ActivMap1.device)



            Result = dict()

            try:
                Result['Emb1'] = torch.cat((losses1,losses2,losses3))
                #Result['Emb1'] = losses1.mean() + (losses2.mean() + losses3.mean())/2
            except:
                print('Error')
                aa=9
            #Result['Emb1'] = losses1.mean()+ (losses2.mean() + losses3.mean())/2

            return Result


        if Mode == 'DualEncoder':
            # compute CNNs
            _, ActivMap1 = self.netAS1(S1A, ActivMode=True)
            _, ActivMap2 = self.netAS1(S1B, ActivMode=True)

            #ActivMap1 = self.Dropout1(ActivMap1)
            #ActivMap2 = self.Dropout2(ActivMap2)

            # apply different attentions
            PosScore = self.AttenAS1(ActivMap1,Mode,ActivMap2)

            Result = dict()
            Result['Emb1'] = PosScore

            return Result



        if Mode == 'PairwiseSymmetric':
            output1 = self.netAS1(S1A, Mode='Normalized',UseSE=UseSE)
            output2 = self.netAS1(S1B, Mode='Normalized',UseSE=UseSE)

            #output1 = self.netAS1(S1A, Mode='NoFC')
            #output2 = self.netAS1(S1B, Mode='NoFC')

            #output1 = self.fcMeta1A(output1)
            #output2 = self.fcMeta1A(output2)

            #output1 = self.act1(output1)
            #output2 = self.act2(output2)

            #output1 = self.fcMeta1AG(output1)
            #output2 = self.fcMeta2AG(output2)

            #output1 = F.normalize(output1, dim=1, p=2)
            #output2 = F.normalize(output2, dim=1, p=2)



            Result = dict()
            Result['Emb1'] = output1
            Result['Emb2'] = output2

            return Result




        if Mode == 'PairwiseAsymmetric':
            # source#1: vis
            output1 = self.netAS1(S1A,Mode='NoFC',UseSE=UseSE)
            output2 = self.netAS2(S1B,Mode='NoFC',UseSE=UseSE)

            #output1 = self.act1(output1)
            #output2 = self.act2(output2)

            #output1 = self.Dropout1(output1)
            #output2 = self.Dropout2(output2)

            output1 = self.fcMeta1A(output1)
            output2 = self.fcMeta2A(output2)


            #output1G = self.fcMeta1AG(output1)
            #output2G = self.fcMeta2AG(output2)

            #output1G = nn.functional.softmax(output1G)
            #output2G = nn.functional.softmax(output2G)

            #output1 = output1*output1G
            #output2 = output2*output2G


            #output1 = F.relu(output1)
            #output2 = F.relu(output2)

            #output1 = self.fc1A(output1)
            #output2 = self.fc2A(output2)


            output1 = F.normalize(output1, dim=1, p=2)
            output2 = F.normalize(output2, dim=1, p=2)


            Result = dict()
            Result['Emb1'] = output1
            Result['Emb2'] = output2
            return Result




        if (Mode == 'Hybrid') :

            # p: probability of an element to be zeroed.Default: 0.5
            DropoutP1 = 0

            # source#1: vis
            # channel1
            EmbSym1  = self.netAS1(S1A,'Normalized')
            EmbAsym1 = self.netAS1(S1A,'Normalized')

            # concat embeddings and apply relu: K+K=256
            #Hybrid1 = torch.cat((EmbSym1, EmbAsym1), 1)

            Hybrid1 = EmbSym1+self.Gain1*EmbAsym1
            #Hybrid1 = F.normalize(Hybrid1, dim=1, p=2)


            Hybrid1 = F.relu(Hybrid1)
            #Hybrid1 = Dropout(DropoutP)(Hybrid1)  # 20% probabilit

            # prepare output
            #Hybrid1 = self.fc1(Hybrid1)
            Hybrid1 = self.fc1A(Hybrid1)
            #Hybrid1 = self.fc1B(self.fc1A(Hybrid1))

            Hybrid1 = F.normalize(Hybrid1, dim=1, p=2)



            # channel2
            EmbSym2  = self.netAS1(S1B,'Normalized')
            EmbAsym2 = self.netAS2(S1B,'Normalized')

            # concat embeddings and apply relu: K+K=256
            #Hybrid2 = torch.cat((EmbSym2, EmbAsym2), 1)

            Hybrid2 = EmbSym2+self.Gain2*EmbAsym2
            #Hybrid2 = F.normalize(Hybrid2, dim=1, p=2)


            Hybrid2 = F.relu(Hybrid2)


            # prepare output
            #Hybrid2 = self.fc2(Hybrid2)
            Hybrid2 = self.fc2A(Hybrid2)

            Hybrid2 = F.normalize(Hybrid2, dim=1, p=2)

            if torch.any(torch.isnan(Hybrid1)) or torch.any(torch.isnan(Hybrid2)):
                print('Nan found')

            Result = dict()
            Result['Hybrid1']  = Hybrid1
            Result['Hybrid2']  = Hybrid2
            Result['EmbSym1']  = EmbSym1
            Result['EmbSym2']  = EmbSym2
            Result['EmbAsym1'] = EmbAsym1
            Result['EmbAsym2'] = EmbAsym2
            return Result



















class AttentionEmbeddingCNN(nn.Module):


    def __init__(self,K,DropoutP,EmbeddingMaxDim=20):
        super(AttentionEmbeddingCNN, self).__init__()

        #self.net = Model()

        self.output_num = [8, 4, 2,1]
        NumScales = len(self.output_num)

        # self.output_num = [8, 4]
        # self.output_num = [4,8]
        # self.output_num = [8]

        self.Query = nn.Parameter(F.normalize(torch.randn(1,1, K),dim=2))
        self.QueryPosEncode = nn.Parameter(F.normalize(torch.randn(1,1, K),dim=2))

        self.SEP = nn.Parameter(torch.randn(1, 1, K))
        self.SepPosEncode = nn.Parameter(torch.randn(1, 1, K))

        EmbeddingMaxDim      = 8
        self.PosEncodingX    = nn.Parameter(F.normalize(torch.randn(EmbeddingMaxDim, int(K / 2)),dim=1))
        self.PosEncodingY    = nn.Parameter(F.normalize(torch.randn(EmbeddingMaxDim, int(K / 2)),dim=1))
        self.PosEncoding1D   = nn.Parameter(F.normalize(torch.randn(NumScales,(EmbeddingMaxDim**2)*2+2,1,K),dim=1))

        self.SegmentEncoding1 = nn.Parameter(torch.randn(1,1,K))
        self.SegmentEncoding2 = nn.Parameter(torch.randn(1,1,K))

        self.SegmentEncodingSEP = nn.Parameter(torch.randn(1, 1, K))


        EncoderLayersNo  = 4
        EncoderHeadsNo   = 4
        DetrEncoderLayer = TransformerEncoderLayer(d_model=K, nhead=EncoderHeadsNo, dim_feedforward=int(K),
                                                        dropout=0.1, activation="relu")
        self.DetrEncoder = TransformerEncoder(encoder_layer=DetrEncoderLayer, num_layers=EncoderLayersNo)


        self.Gain = torch.nn.Parameter(torch.ones(len(self.output_num)))

        self.Dropout  = nn.Dropout(DropoutP)
        self.Dropout1 = nn.Dropout(0.25)

        #self.SPFC = nn.Linear(10880, K)
        self.SPFC = nn.Linear(8704, K)
        self.SPFC1 = nn.Linear(K, K)
        self.Relu  = nn.ReLU()
        #self.SPFC = nn.Linear(640, K)

        #self.BFC  = nn.Linear(16896, K)
        self.BFC = nn.Linear(512, K)
        # self.SPFC = nn.Linear(10240, K)
        # self.SPFC = nn.Linear(8192, K)

        self.LayerNorm = torch.nn.LayerNorm([128,29,29])


    def DualEncoder(self, Conv1,Conv2, conv_size):

        num_sample = Conv1.shape[0]
        for i in range(len(self.output_num)):

            # Pooling support
            h_wid = int(math.ceil(conv_size[0] / self.output_num[i]))
            w_wid = int(math.ceil(conv_size[1] / self.output_num[i]))

            # Padding to retain orgonal dimensions
            h_pad = int((h_wid * self.output_num[i] - conv_size[0] + 1) / 2)
            w_pad = int((w_wid * self.output_num[i] - conv_size[1] + 1) / 2)

            # apply pooling
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))

            y1 = maxpool(Conv1)
            y2 = maxpool(Conv2)



            #else:

            Skip = 1#int(self.output_num[0] / self.output_num[i])
            PosEncoding2D = Prepare2DPosEncoding(self.PosEncodingX,
                                                 self.PosEncodingY,
                                                 #Skip, Skip,
                                                 self.output_num[i],self.output_num[i])

            #PosEncoding2D = self.PosEncoding1D[0, 0:(y1.shape[2] * y1.shape[3]), :, :]
            PosEncoding2D = torch.cat((PosEncoding2D,
                                       self.SepPosEncode,
                                       PosEncoding2D,
                                       self.SepPosEncode), 0)


            SegmentEncoding = torch.cat((self.SegmentEncoding1.repeat(y1.shape[2]*y1.shape[3],1,1),
                                         self.SegmentEncodingSEP,
                                         self.SegmentEncoding2.repeat(y1.shape[2]*y1.shape[3],1,1),
                                         self.SegmentEncodingSEP)
                                        ,0)
            PosEncoding2D += SegmentEncoding

            PosEncoding = torch.cat((self.QueryPosEncode, PosEncoding2D), 0)

            x1 = y1.reshape((y1.shape[0], y1.shape[1], y1.shape[2] * y1.shape[3]))
            x1 = x1.permute(2, 0, 1)

            x2 = y2.reshape((y2.shape[0], y2.shape[1], y2.shape[2] * y2.shape[3]))
            x2 = x2.permute(2, 0, 1)

            Query = self.Query.repeat(1, x1.shape[1], 1)
            SEP   = self.SEP.repeat(1, x1.shape[1], 1)

            x = torch.cat((Query,x1,SEP,x2,SEP), 0)

            #x = self.DetrEncoder(src=x, pos=PosEncoding)
            x = self.DetrEncoder(x+PosEncoding)

            x = x[0,]

            if (i == 0):
                if False:
                    spp1 = y1.reshape(num_sample, -1)
                    spp2 = y2.reshape(num_sample, -1)
                    spp  = torch.cat((spp1, spp2), 1)
                else:
                    spp = x.reshape(num_sample, -1)
                    continue

            spp = torch.cat((spp, x.reshape(num_sample, -1)), 1)
            #spp = self.Gain[i]*x
        return spp






    def spatial_pyramid_pool_2D(self, Conv1, conv_size):

        #Conv1 = self.LayerNorm(Conv1)

        num_sample = Conv1.shape[0]
        for i in range(len(self.output_num)):

            # Pooling support
            h_wid = int(math.ceil(conv_size[0] / self.output_num[i]))
            w_wid = int(math.ceil(conv_size[1] / self.output_num[i]))

            # Padding to retain original dimensions
            h_pad = int((h_wid * self.output_num[i] - conv_size[0] + 1) / 2)
            w_pad = int((w_wid * self.output_num[i] - conv_size[1] + 1) / 2)

            # apply pooling
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))

            y = maxpool(Conv1)

            if (i == 0):
                spp = y.reshape(num_sample, -1)

            if self.output_num[i] == 1:
                x   = F.normalize(y.reshape(num_sample, -1), dim=1, p=2)
                spp = torch.cat((spp, x), 1)
                continue

            StepX = int(self.PosEncodingX.shape[0]/self.output_num[i])
            StepY = int(self.PosEncodingY.shape[0]/self.output_num[i])
            PosEncoding2D = Prepare2DPosEncoding(self.PosEncodingX[0:Conv1.shape[0]:StepX,:],
                                                 self.PosEncodingY[0:Conv1.shape[1]:StepY,:],
                                                 y.shape[2], y.shape[3])/math.sqrt(2)

            #PosEncoding2D = self.Dropout(PosEncoding2D)

            #PosEncoding2D = self.PosEncoding1D[0,0:(y.shape[2]*y.shape[3]),:,:]

            PosEncoding = torch.cat((self.QueryPosEncode, PosEncoding2D), 0)

            x = y.reshape((y.shape[0], y.shape[1], y.shape[2] * y.shape[3]))
            x = x.permute(2, 0, 1)
            x = x / math.sqrt(((x ** 2).sum(2).median()))

            Query = self.Query.repeat(1, x.shape[1], 1)
            x = torch.cat((Query, x), 0)
            #x = self.DetrEncoder(src=x, pos=PosEncoding)

            mask = torch.ones((x.shape[1], x.shape[0])).cuda()
            mask = (self.Dropout(mask) == 0)
            mask[:,0] = False

            #x = self.DetrEncoder(src =x+PosEncoding,src_key_padding_mask = mask)
            x = self.DetrEncoder(src=x + PosEncoding)

            x = x[0,]

            x = F.normalize(x, dim=1, p=2)

            #if (i == 0):
             #   spp = x.reshape(num_sample, -1))
            #else:
            spp = torch.cat((spp, x.reshape(num_sample, -1)), 1)

        return spp



    def forward(self, ActivMap1,Mode,ActivMap2=None,Embedd=None):

        if (Mode == 'SymmetricEncoder') or (Mode == 'AsymmetricEncoder'):
            spp_a = self.spatial_pyramid_pool_2D(ActivMap1, [int(ActivMap1.size(2)), int(ActivMap1.size(3))])
            spp_a = self.Dropout1(spp_a)
            Result = self.SPFC(spp_a)
            Result = F.normalize(Result, dim=1, p=2)

        if (Mode == 'DualEncoder') or (Mode == 'DualEncoderHard'):
            spp_a = self.DualEncoder(ActivMap1,ActivMap2,[int(ActivMap1.size(2)), int(ActivMap1.size(3))])
            #spp_a = F.normalize(spp_a, dim=1, p=2)
            Result = self.BFC(spp_a)
            Result = F.normalize(Result, dim=1, p=2)



        return Result


class EMA(nn.Module):
    """ Model Exponential Moving Average V2 from timm"""

    def __init__(self, model, decay=0.9999,update_after_step=100,update_every=10):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

        self.step              = 0
        self.update_every      = update_every
        self.update_after_step = update_after_step

    def copy_params_from_model_to_ema(self,model):

        model_params = OrderedDict(model.named_parameters())
        shadow_params = OrderedDict(self.module.named_parameters())

        assert model_params.keys() == shadow_params.keys()


        with torch.no_grad():
            try:
                for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                    ema_v.copy_(model_v)
            except:
                for ema_v, model_v in zip(self.module.module.state_dict().values(), model.state_dict().values()):
                    ema_v.copy_(model_v)

    def _update(self, model, update_fn):
        with torch.no_grad():
            try:
                for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                    ema_v.copy_(update_fn(ema_v, model_v))
            except:
                for ema_v, model_v in zip(self.module.module.state_dict().values(), model.state_dict().values()):
                    ema_v.copy_(update_fn(ema_v, model_v))
    def update(self, model):
        step = self.step
        self.step += 1

        if (step % self.update_every) != 0:
            return

        if step <= self.update_after_step:
            self.copy_params_from_model_to_ema(model)
            return

        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)


    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)




