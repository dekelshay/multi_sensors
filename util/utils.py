import numpy as np
import torch
import glob
import os

from torch.optim.lr_scheduler import ReduceLROnPlateau
import pathlib
import shutil
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree
import faiss #conda install -c pytorch faiss-cpu conda install faiss-gpu cudatoolkit=10.2 -c pytorch -c conda-forge
from math import floor
import yaml
from tqdm import tqdm
import sys

from network.my_classes import NormalizeImages,FPR95Accuracy,MetricLearningCnn
from util.read_matlab_imdb import read_matlab_imdb

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def ClearPytorchCache():
    for p in pathlib.Path('.').rglob('__pycache__'):
        try:
            #print(p)
            shutil.rmtree(p)
        except:
            aa = 9

def LoadNetworkModel(ModelName):

    checkpoint = torch.load(ModelName)
    net = MetricLearningCnn(checkpoint['Mode'], DropoutP=0)

    net_dict = net.state_dict()
    state_dict= {k: v for k, v in checkpoint['state_dict'].items() if
                                (k in net_dict) and (net_dict[k].shape == checkpoint['state_dict'][k].shape)}

    aa = net.load_state_dict(state_dict, strict=True)

    return net,aa


def LoadTestFiles(TestDir):

    FileList = glob.glob(TestDir + "*.hdf5")
    TestData = dict()
    for File in FileList:
        path, DatasetName = os.path.split(File)
        DatasetName = os.path.splitext(DatasetName)[0]

        Data = read_matlab_imdb(File)

        x = Data['Data'].astype(np.float32)
        TestLabels = torch.from_numpy(np.squeeze(Data['Labels']))
        del Data

        x[:, :, :, :, 0] -= x[:, :, :, :, 0].mean()
        x[:, :, :, :, 1] -= x[:, :, :, :, 1].mean()

        x = NormalizeImages(x)
        x = torch.from_numpy(x)

        TestData[DatasetName] = dict()
        TestData[DatasetName]['Data'] = x
        TestData[DatasetName]['Labels'] = TestLabels

    return TestData




def LoadModel(net,StartBestModel,ModelsDirName,BestFileName,UseBestScore,device):

    scheduler = None
    optimizer = None

    LowestError = 1e5
    NegativeMiningMode = 'Random'

    if StartBestModel:
        FileList = glob.glob(ModelsDirName + BestFileName + '.pth')
    else:
        FileList = glob.glob(ModelsDirName + "visnir_sym*.pth")

    if FileList:
        FileList.sort(key=os.path.getmtime)

        ModelFileStr = FileList[-1] + ' loded'
        print(ModelFileStr)

        checkpoint = torch.load(FileList[-1])

        if ('LowestError' in checkpoint.keys()) and UseBestScore:
            LowestError = checkpoint['LowestError']

        if 'NegativeMiningMode' in checkpoint.keys():
            NegativeMiningMode = checkpoint['NegativeMiningMode']

        net_dict = net.state_dict()
        state_dict = {k: v for k, v in checkpoint['state_dict'].items() if
                                    (k in net_dict) and (net_dict[k].shape == checkpoint['state_dict'][k].shape)}

        net.load_state_dict(state_dict, strict=False)

        if 'optimizer_name' in checkpoint.keys():
            if checkpoint['optimizer_name'] == 'Lookahead':
                optimizer = RangerLars(net.parameters())

            if checkpoint['optimizer_name'] == 'Adam':
                    optimizer = torch.optim.Adam(net.parameters())

            if checkpoint['optimizer_name'] == 'Lamb':
                    optimizer = torch.optim.Lamb(net.parameters())

            try:
                #optimizer.load_state_dict(checkpoint['optimizer'])
                optimizer = checkpoint['optimizer']
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda(device)
            except Exception as e:
                print(e)
                print('Optimizer loading error')




        if ('scheduler_name' in checkpoint.keys()) and (optimizer != None):

            try:
                if checkpoint['scheduler_name'] == 'ReduceLROnPlateau':
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, verbose=True)

                if checkpoint['scheduler_name'] == 'StepLR':
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=10)

                #scheduler.load_state_dict(checkpoint['scheduler'])
                scheduler= checkpoint['scheduler']
            except Exception as e:
                print(e)
                print('Optimizer loading error')

        StartEpoch = checkpoint['epoch'] + 1
    else:
        ModelFileStr = 'Weights file NOT loaded!!'
        print(ModelFileStr)
        optimizer  = None
        StartEpoch = 0

    FileList = glob.glob(ModelsDirName + BestFileName + '.pth')
    # noinspection PyInterpreter

    print('LowestError: ' + repr(LowestError)[0:6])

    return net,optimizer,LowestError,StartEpoch,scheduler,NegativeMiningMode,ModelFileStr





class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch

class MyGradScaler:
    def __init__(self):
        pass
    def scale(self, loss):
        return loss
    def unscale_(self, optimizer):
        pass
    def step(self, optimizer):
        optimizer.step()
    def update(self):
        pass


def EvaluateDualNets(net,Data1, Data2,CnnMode=None,device=None,StepSize=100,TqdmDisable=True,UseSE=False):

    with torch.no_grad():
        train_tqdm_gen = tqdm(range(0, Data1.shape[0], StepSize), desc="Evaluating", leave=True, position=0, file=sys.stderr,disable=TqdmDisable)
        for k in train_tqdm_gen:
        #for k in range(0, Data1.shape[0], StepSize):

            # ShowTwoRowImages(x[0:3, :, :, 0], x[0:3, :, :, 1])
            a = Data1[k:(k + StepSize), :, :, :]  # - my_training_Dataset.VisAvg
            b = Data2[k:(k + StepSize), :, :, :]  # - my_training_Dataset.VisAvg

            # ShowTwoRowImages(a[0:10,0,:,:], b[0:10,0,:,:])
            a,b = a.to(device),b.to(device)
            x = net(a,b,CnnMode,UseSE=UseSE)

            if k == 0:
                keys = list(x.keys())
                Emb = dict()
                for key in keys:
                    Emb[key] = np.zeros((Data1.shape[0], x[key].shape[1]), dtype=np.float32)

            for key in keys:
                Emb[key][k:(k + StepSize)] = x[key].cpu()

    return Emb


def CreatePseudoLables2to1(net,Data,CnnMode,device,ValidayionStepSize,PseudoLablesPercent):

    x = Data.astype(np.float32)

    x[:, :, :, 0] -= x[:, :, :, 0].mean()
    x[:, :, :, 1] -= x[:, :, :, 1].mean()

    x = NormalizeImages(x)
    x = torch.from_numpy(x)

    #compute embedding
    if x.ndim == 4:
        x =  x.unsqueeze(1)

    with torch.no_grad():
        Emb = EvaluateDualNets(net, x[:, :, :, :,0], x[:, :, :, :,1], CnnMode,device,ValidayionStepSize);

    Emb1 = Emb['Emb1']
    Emb2 = Emb['Emb2']

    faiss_index = faiss.IndexFlatL2(Emb1.shape[1])
    faiss_index.add(Emb1)
    dist,idx = faiss_index.search(Emb2,k=1)
    dist = np.stack(dist, axis=1).squeeze()
    idx  = np.stack(idx, axis=1).squeeze()

    #sort the distances Emb2->Emb1
    Emb2Idx = np.argsort(dist)
    Emb2Idx = Emb2Idx[0:floor(Emb2Idx.shape[0]*PseudoLablesPercent)]

    #dist = dist[Emb2Idx]
    Emb1Idx  = idx[Emb2Idx]

    #dd = dist[Emb1Idx]

    return Emb1Idx,Emb2Idx


def EvaluateNet(net,data,device,StepSize):

    if (torch.cuda.device_count() > 1):
        net = nn.DataParallel(net)

    with torch.no_grad():

        for k in range(0, data.shape[0], StepSize):

            # ShowTwoRowImages(x[0:3, :, :, 0], x[0:3, :, :, 1])
            a = data[k:(k + StepSize), :, :, :]  # - my_training_Dataset.VisAvg

            # ShowTwoRowImages(a[0:10,0,:,:], b[0:10,0,:,:])
            a = a.to(device)
            a = net(a)

            if k==0:
                EmbA = np.zeros((data.shape[0], a.shape[1]),dtype=np.float32)

            EmbA[k:(k + StepSize)] = a.cpu()

    return EmbA




def ComputeTestError(TestData,net,TestDecimation,FPR,CnnMode,device,ValidayionStepSize,UseSE=False):
    NoSamples = 0
    TotalTestError = 0
    TotalFPR95 = 0

    train_tqdm_gen = tqdm(TestData, desc="Test Sets ", leave=True, position=0, file=sys.stderr,disable=False)
    for DataName in enumerate(train_tqdm_gen):
    #for DataName in TestData:
        DataName=DataName[1]
        EmbTest = EvaluateDualNets(net, TestData[DataName]['Data'][0::TestDecimation, :, :, :, 0],
                                   TestData[DataName]['Data'][0::TestDecimation, :, :, :, 1], CnnMode,
                                   device,
                                   ValidayionStepSize,
                                   TqdmDisable=True,
                                   UseSE=UseSE)

        if CnnMode == 'DualEncoder':
            # Dist = -EmbTest['Emb1'].squeeze()
            x = torch.from_numpy(EmbTest['Emb1'])

            if NumGpus == 1:
                net.AngularLoss.to("cpu")
                logits = net.AngularLoss(x, Mode='logits')
                net.AngularLoss.to(device)
            else:
                net.module.AngularLoss.to("cpu")
                logits = net.module.AngularLoss(x, Mode='logits')
                net.module.AngularLoss.to(device)

            Dist = logits[:, 0].numpy()
        else:
            Dist = np.power(EmbTest['Emb1'] - EmbTest['Emb2'], 2).sum(1)

        TestData[DataName]['TestError'], TestFPR95 = \
            FPR95Accuracy(Dist, TestData[DataName]['Labels'][0::TestDecimation], FPR=FPR)
        # number of errorneous samples
        TotalTestError += TestData[DataName]['TestError'] * TestData[DataName]['Data'].shape[0] * 100
        NoSamples += TestData[DataName]['Data'].shape[0]
        TotalFPR95 = max(TotalFPR95, TestFPR95)
    TotalTestError /= NoSamples

    return TotalTestError,TotalFPR95


def SaveDict2Yaml(param_dict,fname):
    yaml_string = yaml.dump(param_dict)
    print(yaml_string)
    file = open(fname, "w")
    yaml.dump(param_dict, file);
    file.close()