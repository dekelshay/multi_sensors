import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Optimizer
from colorama import init,Fore, Back, Style


class OnlineHardNegativeMiningTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin,Mode,MarginRatio=1,PosRatio=1,NegPow=1,PosPow=1,device=None):
        super(OnlineHardNegativeMiningTripletLoss, self).__init__()
        self.margin    = margin
        self.Mode      = Mode
        self.MarginRatio = MarginRatio
        self.PosRatio  = PosRatio
        self.PosPow    = PosPow
        self.NegPow    = NegPow
        self.device    = device


        # self.VICRegLoss = VICRegLoss()  #SHAY CHANGE


    def forward(self, emb1, emb2):

        if self.Mode == 'Random':
            #with torch.no_grad():
            NegIdx = torch.randint(high=emb1.shape[0], size=(emb1.shape[0],),device=self.device)

            ap_distances = (emb1 - emb2).pow(2).sum(1)  # .pow(.5)
            an_distances = (emb1 - emb2[NegIdx, :]).pow(2).sum(1)  # .pow(.5)
            margin = ap_distances - an_distances


        if (self.Mode == 'Hardest')  | (self.Mode == 'HardPos'):
            #Dist  = sim_matrix(emb1,emb2).cpu().detach()


            with torch.no_grad():
                #emb = torch.cat((emb1,emb2), 0)
                emb = emb2
                Similarity = torch.mm(emb1, emb.transpose(0, 1))

                #Similarity = torch.mm(emb1, emb2.transpose(0, 1))
                Diagonal = torch.eye(n=emb1.shape[0], m=emb1.shape[0], device=self.device)
                #Diagonal = torch.cat((Diagonal, Diagonal), 1)

                Similarity -= 1000000000*Diagonal
                NegIdx = torch.argmax(Similarity, axis=1) #find negative with highest similarity

            emb    = emb[NegIdx, :]



        if (self.Mode == 'Hardest'):

            #ap_distances = (emb1 - emb2[0:emb1.shape[0],:]).pow(2).sum(1)  # .pow(.5)
            ap_distances = (emb1 - emb2).pow(2).sum(1)  # .pow(.5)
            an_distances = (emb1 - emb).pow(2).sum(1)

            margin = ap_distances - an_distances



        if (self.Mode == 'HardPos'):

            #ap_distances = (emb1 - emb2[0:emb1.shape[0],:,]).pow(2).sum(1)  # .pow(.5)
            ap_distances = (emb1 - emb2).pow(2).sum(1)  # .pow(.5)
            an_distances = (emb1 - emb2[NegIdx, :]).pow(2).sum(1)

            #get LARGEST positive distances
            with torch.no_grad():
                PosIdx = ap_distances.argsort(dim=-1, descending=True)#sort positive distances
                PosIdx = PosIdx[0:int(self.PosRatio * PosIdx.shape[0])]#retain only self.PosRatio of the positives

                NegIdx=NegIdx[PosIdx]

            margin = ap_distances[PosIdx] - an_distances[PosIdx]

            # hard examples first: sort margin
            with torch.no_grad():
                Idx = margin.argsort(dim=-1, descending=True)

                # retain a subset of hard examples
                Idx = Idx[0:int(self.MarginRatio * Idx.shape[0])]#retain some of the examples

            margin = margin[Idx]



        losses = F.relu(margin + self.margin)
        #return losses
        idx = torch.where(losses>0)[0]

        if idx.size()[0]>0:
            losses = losses[idx].mean()

            if torch.isnan(losses):
                print('Found nan in loss ')
        else:
            losses = 0
            print('\n' + Fore.BLACK + Back.MAGENTA + 'No margin samples'+ Style.RESET_ALL)
            print('\n' + Fore.BLACK + Back.MAGENTA + 'New margin = ' +repr(self.margin)[0:4] + Style.RESET_ALL)

        return losses



class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels=None,Mode = 'Regular',device = None):
        '''
        input shape (N, in_features)
        '''

        try:
            if Mode != 'logits':
                assert len(x) == len(labels)
                assert torch.min(labels) >= 0
                assert torch.max(labels) < self.out_features
        except:
            aa=9

        #normalize L2 columns of FC
        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        # normalize L2 columns of x
        x = F.normalize(x, p=2, dim=1)

        logits = self.fc(x)

        if Mode == 'logits':
            return logits

        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(logits.transpose(0, 1)[labels]) - self.m)
        
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(
                torch.clamp(torch.diagonal(logits.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
        
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(
                torch.clamp(torch.diagonal(logits.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

        excl = torch.cat([torch.cat((logits[i, :y], logits[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)

        Loss = -L

        return logits,Loss



    


