import torch
import torchvision
import matplotlib.pyplot as plt
from colorama import init,Fore, Back, Style
from termcolor import colored
import numpy as np
import glob
import os
import copy
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,CosineAnnealingWarmRestarts
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import GPUtil
import math
from util.warmup_scheduler import GradualWarmupSchedulerV2
import gc
from torchviz import make_dot



# my classes
from network.my_classes import imshow, ShowRowImages, ShowTwoRowImages,FPR95Accuracy
from network.my_classes import MetricLearningCnn, NormalizeImages,EMA
from network.generator import DatasetPairwiseTriplets,worker_init_fn,CreateCorruptedBatch
from network.losses import OnlineHardNegativeMiningTripletLoss   #,FocalLoss # SHAY CHANGE

from util.read_matlab_imdb import read_matlab_imdb
from util.utils import LoadModel,MultiEpochsDataLoader,MyGradScaler,ClearPytorchCache,EvaluateDualNets,\
    CreatePseudoLables2to1,LoadTestFiles,ComputeTestError,LoadNetworkModel,SaveDict2Yaml

import warnings
warnings.filterwarnings("ignore", message="UserWarning: albumentations.augmentations.transforms.RandomResizedCrop")

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


if __name__ == '__main__':
    np.random.seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NumGpus = torch.cuda.device_count()
    torch.cuda.empty_cache()
    gc.collect()
    GPUtil.showUtilization()

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    name = torch.cuda.get_device_name(0)

    ModelsDirName = './models2/'
    LogsDirName = './logs2/'
    Description = 'Symmetric CNN with Triplet loss, no HM'
    BestFileName = 'visnir_best'
    FileName = 'visnir_sym_triplet'

    TestDir = r'F:\Data\multisensor\test\\'
    TrainFile = r'F:\Data\multisensor\train\Vis-Nir_Train.hdf5'

    # TestDir = 'F:\\multisensor\\Vis-Nir_grid\\'
    # TrainFile = 'F:\\multisensor\\Vis-Nir_grid\\Vis-Nir_grid_Train.hdf5'

    TestDecimation = 10
    FPR95 = 0.8

    ClearPytorchCache()

    writer = SummaryWriter(LogsDirName)
    LowestError = 1e10

    # ----------------------------     configuration   ---------------------------
    Augmentation = {}
    Augmentation["HorizontalFlip"] = False
    Augmentation["VerticalFlip"] = False
    Augmentation["Rotate90"] = True
    Augmentation["Test"] = {'Do': True}
    Augmentation["RandomCrop"] = {'Do': False, 'MinDx': 0, 'MaxDx': 0.2, 'MinDy': 0, 'MaxDy': 0.2}



    AssymetricInitializationPhase = False

    TestMode = False

    torch.manual_seed(0)
    np.random.seed(0)
    #torch.set_deterministic(True)

    BCEloss     = nn.BCEWithLogitsLoss(reduction='none')
    # Focal       = FocalLoss(alpha=2, gamma=3)
    FPR         = 0.95

    if True:
        # GeneratorMode = 'Pairwise'
        GeneratorMode = 'DomainAdaptation'

        CnnMode = 'PairwiseSymmetric'
        #CnnMode = 'SymmetricEncoder'
        #CnnMode = 'DualEncoder'
        #CnnMode = 'DualEncoderHard'
        NegativeMiningMode = 'Random'
        #NegativeMiningMode = 'Hardest'
        #NegativeMiningMode = 'HardPos'
        LossMargin = 1
        criterion = OnlineHardNegativeMiningTripletLoss(margin=LossMargin, Mode=NegativeMiningMode,device=device)
        #criterion         = OnlineHaOnlineHardNegativeMiningTripletLossrdNegativeMiningTripletLoss(margin=1, Mode='HardPos', MarginRatio=0.5)
        #criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='HardPos', MarginRatio=1.0/2, PosRatio=1. / 2)
        Description = 'PairwiseSymmetric Hardest'

        CreatePseudoLabels  = False
        PseudoLablesPercent = 1.0

        #visnir
        PseudoLabelsModel   = r'D:\Dropbox\MutiSensors\Results\unsupervised\vis-nir\phase0\visnir_sym_triplet16.pth'
        PseudoLabelsModel   = r'D:\Dropbox\MutiSensors\Results\unsupervised\vis-nir\phase1\visnir_sym_triplet49.pth'
        PseudoLabelsModel   = r'D:\Dropbox\MutiSensors\Results\unsupervised\vis-nir\visnir_sym_triplet70.pth'
        PseudoLabelsModel   = r'D:\Dropbox\MutiSensors\Results\unsupervised\vis-nir\visnir_sym_triplet83.pth'
        PseudoLabelsModel   = r'D:\Dropbox\MutiSensors\Results\unsupervised\vis-nir\visnir_sym_triplet340.pth'
        PseudoLabelsModel   = r'D:\Dropbox\MutiSensors\Results\unsupervised\vis-nir\visnir_sym_triplet565.pth'

        # #Vis-Nir_grid
        # PseudoLabelsModel = 'results\\unsupervised\\Vis-Nir_grid\\phase0\\visnir_sym_triplet1.pth'
        # PseudoLabelsModel = 'results\\unsupervised\\Vis-Nir_grid\\phase1\\visnir_sym_triplet51.pth'
        # PseudoLabelsModel = 'results\\unsupervised\\Vis-Nir_grid\\phase2\\visnir_sym_triplet995.pth'
        # PseudoLabelsModel = 'results\\unsupervised\\Vis-Nir_grid\\phase3\\visnir_sym_triplet170.pth'
        # PseudoLabelsModel = 'results\\unsupervised\\Vis-Nir_grid\\phase4\\visnir_sym_triplet419.pth'
        # PseudoLabelsModel = 'results\\unsupervised\\Vis-Nir_grid\\phase5\\visnir_sym_triplet440.pth'

        InitializeOptimizer = True
        UseWarmUp           = True

        StartBestModel      = False
        UseBestScore        = False

        #EMA
        UseEMA              = False
        EmaDecay = 1.0 - 1e-3

        FPR = 0.95

        UseCorruptedBatch = False
        BatchCorruptionRatio = 0

        LearningRate = 1e-3
        Random2HardLR = LearningRate/1000

        patience = 3
        weight_decay = 0
        DropoutP = 0.0
        WarmUpEpochs = 4

        OuterBatchSize = 12#4*12
        InnerBatchSize = 12#4*12
        ValidayionStepSize = 2048*4
        Augmentation["Test"] = {'Do': True}
        Augmentation["HorizontalFlip"] = True
        Augmentation["VerticalFlip"] = True
        Augmentation["Rotate90"] = True
        Augmentation["RandomCrop"] = {'Do': True, 'MinDx': 0, 'MaxDx': 0.2, 'MinDy': 0, 'MaxDy': 0.2}
        Augmentation["albumentations"] = True ## SHAY CHANGE

        FreezeCnns = False

    # if False:
    #     GeneratorMode = 'Pairwise'
    #     CnnMode = 'PairwiseAsymmetric'
    #     CnnMode = 'AsymmetricEncoder'
    #
    #     NegativeMiningMode = 'Random'
    #     #NegativeMiningMode = 'Hardest'
    #     criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode=NegativeMiningMode,device=device)
    #     # criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='HardPos', MarginRatio=1.0/2, PosRatio=1. / 2)
    #
    #     InitializeOptimizer = True
    #     UseWarmUp           = True
    #
    #     StartBestModel      = False
    #     UseBestScore        = False
    #
    #     LearningRate = 1e-0
    #     OuterBatchSize = 4 * 12
    #     InnerBatchSize = 2 * 12
    #
    #     patience     = 3
    #     weight_decay = 0
    #     DropoutP = 0.0
    #
    #     WarmUpEpochs = 4
    #
    #     Augmentation["Test"] = {'Do': False}
    #     Augmentation["HorizontalFlip"] = True
    #     Augmentation["VerticalFlip"] = False
    #     Augmentation["Rotate90"] = True
    #     Augmentation["RandomCrop"] = {'Do': False, 'MinDx': 0, 'MaxDx': 0.2, 'MinDy': 0, 'MaxDy': 0.2}
    #
    #     #AssymetricInitializationPhase = True
    #     Description = 'PairwiseAsymmetric'
    #
    #     FreezeCnns = False

    # if False:
    #     GeneratorMode = 'Pairwise'
    #     # CnnMode            = 'HybridRot'
    #     CnnMode = 'Hybrid'
    #     CnnMode = 'AttenHybrid'
    #
    #     NegativeMiningMode = 'Random'
    #     #NegativeMiningMode = 'Hardest'
    #
    #     criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode=NegativeMiningMode,device=device)
    #     #criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode="Hardest",device=device)
    #     #criterion        = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='HardPos', MarginRatio=1.0/4, PosRatio=1./4)
    #     #HardestCriterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='Hardest')
    #
    #     #criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode='HardPos', MarginRatio=1.0 / 2, PosRatio=1. / 2,device=device)
    #
    #     PairwiseLoss      = PairwiseLoss()
    #
    #     InitializeOptimizer = True
    #     UseWarmUp           = True
    #
    #     StartBestModel = False
    #     UseBestScore = False
    #
    #     LearningRate = 1e-1
    #     OuterBatchSize = 2*12
    #     InnerBatchSize = 2*12
    #
    #
    #     DropoutP = 0.5
    #     weight_decay= 0#1e-5
    #
    #     TestMode = False
    #     TestDecimation = 10
    #
    #
    #     AssymetricInitializationPhase = False
    #
    #     Augmentation["Test"] = {'Do': False}
    #     Augmentation["HorizontalFlip"] = True
    #     Augmentation["VerticalFlip"] = True
    #     Augmentation["Rotate90"] = True
    #     Augmentation["RandomCrop"] = {'Do': True, 'MinDx': 0, 'MaxDx': 0.2, 'MinDy': 0, 'MaxDy': 0.2}


    # ----------------------------- read data----------------------------------------------
    Data = read_matlab_imdb(TrainFile)
    TrainingSetData = Data['Data']
    TrainingSetLabels = np.squeeze(Data['Labels'])
    TrainingSetSet = np.squeeze(Data['Set'])
    del Data

    TrainIdx = np.squeeze(np.asarray(np.where(TrainingSetSet == 1)))
    ValIdx = np.squeeze(np.asarray(np.where(TrainingSetSet == 3)))

    # VALIDATION data
    ValSetLabels = torch.from_numpy(TrainingSetLabels[ValIdx])

    ValSetData = TrainingSetData[ValIdx, :, :, :].astype(np.float32)
    ValSetData[:, :, :, :, 0] -= ValSetData[:, :, :, :, 0].mean()
    ValSetData[:, :, :, :, 1] -= ValSetData[:, :, :, :, 1].mean()
    ValSetData = torch.from_numpy(NormalizeImages(ValSetData));





    # TRAINING data
    TrainingSetData = np.squeeze(TrainingSetData[TrainIdx,])
    TrainingSetLabels = TrainingSetLabels[TrainIdx]

    if CreatePseudoLabels:
        PseudoLabelsNet,msg = LoadNetworkModel(PseudoLabelsModel)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            PseudoLabelsNet = nn.DataParallel(PseudoLabelsNet)
        PseudoLabelsNet.eval()
        PseudoLabelsNet.to(device)
        idx1,idx2 = CreatePseudoLables2to1(PseudoLabelsNet, TrainingSetData, CnnMode, device, ValidayionStepSize,PseudoLablesPercent)
        TrainingSetData = np.concatenate((np.expand_dims(TrainingSetData[idx1, :, :, 0], 3),
                                          np.expand_dims(TrainingSetData[idx2, :, :, 1], 3)), axis=3)
        TrainingSetLabels=TrainingSetLabels[idx2]
        del PseudoLabelsNet


    # define generators
    Training_Dataset = DatasetPairwiseTriplets(TrainingSetData, TrainingSetLabels, InnerBatchSize, Augmentation, GeneratorMode)
    # Training_DataLoader = data.DataLoader(Training_Dataset, batch_size=OuterBatchSize, shuffle=True,num_workers=8,pin_memory=True)
    Training_DataLoader = MultiEpochsDataLoader(Training_Dataset, batch_size=OuterBatchSize, shuffle=True,
                                                num_workers=8, pin_memory=True,worker_init_fn=worker_init_fn)


    # Load all TEST datasets
    TestData = LoadTestFiles(TestDir)


    # -------------------------    loading previous results   ------------------------
    net = MetricLearningCnn(CnnMode,DropoutP)
    optimizer = torch.optim.Adam(net.parameters(), lr=LearningRate)

    if False:
        net.to(device)
        net.eval()
        TotalTestError, TotalFPR95 = ComputeTestError(TestData, net, TestDecimation, FPR, CnnMode, device,
                                                      ValidayionStepSize)
        net1,msg = LoadNetworkModel(PseudoLabelsModel)
        net1.to(device)
        net1.eval()
        TotalTestError, TotalFPR95 = ComputeTestError(TestData, net1, TestDecimation, FPR, CnnMode, device,
                                                      ValidayionStepSize)

    StartEpoch = 0

    net,optimizer,LowestError,StartEpoch,scheduler,LodedNegativeMiningMode,ModelFileStr =  \
        LoadModel(net, StartBestModel, ModelsDirName, BestFileName, UseBestScore, device)
    print('LodedNegativeMiningMode: ' + LodedNegativeMiningMode)


    if not InitializeOptimizer:
        criterion.Mode = LodedNegativeMiningMode


    net.FreezeCnns(FreezeCnns)

    if InitializeOptimizer:
        optimizer = torch.optim.Adam(
            [{'params': filter(lambda p: p.requires_grad == True, net.parameters()),'lr': LearningRate, 'weight_decay': weight_decay},
             {'params': filter(lambda p: p.requires_grad == False, net.parameters()),'lr': 0, 'weight_decay': 0}],
            betas=(0.9, 0.999),lr=0, weight_decay=0.00)



    # ------------------------------------------------------------------------------------------


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net.to(device)

    #TotalTestError, TotalFPR95 = ComputeTestError(TestData, net, TestDecimation, FPR, CnnMode, device,ValidayionStepSize)

    if UseEMA:
        print('EMA loaded')
        #if torch.cuda.device_count() > 1:
        EMA = EMA(net, decay=EmaDecay,update_after_step=10,update_every=10)
        #else:
         #   EMA = EMA(net, decay=1.0 - 1e-3)
        #EMA.module.to(device)


    ########################################################################
    # Train the network
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True)
    #LRscheduler =  StepLR(optimizer, step_size=10, gamma=0.1)

    if UseWarmUp:
        scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=1, total_epoch=WarmUpEpochs, after_scheduler= StepLR(optimizer, step_size=10, gamma=0.1))

    CeLoss = nn.CrossEntropyLoss()

    param_dict = {
        'Mode': {'GeneratorMode': GeneratorMode, 'CnnMode': CnnMode, 'NegativeMiningMode': NegativeMiningMode,
                 'LossMargin': LossMargin},
        'PseudoLabels': {'CreatePseudoLabels': CreatePseudoLabels, 'PseudoLablesPercent': PseudoLablesPercent,
                         'PseudoLabelsModel': PseudoLabelsModel},
        'CorruptedBatch': {'UseCorruptedBatch': UseCorruptedBatch, 'BatchCorruptionRatio': BatchCorruptionRatio},
        'Optimization': {'InitializeOptimizer': InitializeOptimizer, 'UseWarmUp': UseWarmUp,
                         'StartBestModel': StartBestModel, 'UseBestScore': UseBestScore, 'UseEMA': UseEMA,
                         'EmaDecay': EmaDecay},
        'Training': {'LearningRate': LearningRate, 'Random2HardLR': Random2HardLR, 'patience': patience,
                     'weight_decay': weight_decay,
                     'DropoutP': DropoutP, 'WarmUpEpochs': WarmUpEpochs,
                     'InnerBatchSize': InnerBatchSize, 'OuterBatchSize': OuterBatchSize,
                     'ValidayionStepSize': ValidayionStepSize,'ModelFileStr': ModelFileStr},
        'Augmentations': {'Augmentation["Test"]': Augmentation["Test"],
                          'Augmentation["HorizontalFlip"]': Augmentation["HorizontalFlip"],
                          'Augmentation["VerticalFlip"]': Augmentation["VerticalFlip"],
                          'Augmentation["Rotate90"]': Augmentation["Rotate90"],
                          'Augmentation["RandomCrop"]': Augmentation["RandomCrop"]}}
    SaveDict2Yaml(param_dict, ModelsDirName + FileName + '.yaml')

    print(CnnMode + ' training\n')


    # writer.add_graph(net, images)
    for epoch in range(StartEpoch, 1000):  # loop over the dataset multiple times

        running_loss_pos = 0
        running_loss_neg = 0
        optimizer.zero_grad()

        #warmup
        if  (epoch - StartEpoch < WarmUpEpochs) and UseWarmUp:
            #print(colored('\n Warmup step #' + repr(epoch - StartEpoch), 'green', attrs=['reverse', 'blink']))
            print('\n'+Fore.BLACK + Back.RED + ' Warmup step #' + repr(epoch - StartEpoch) + Style.RESET_ALL)
            scheduler_warmup.step()
        else:
            if epoch > StartEpoch:
                print('CurrentError=' + repr(ValError)[0:8])

                if (type(scheduler).__name__ == 'StepLR') or (type(scheduler).__name__ == 'CosineAnnealingWarmRestarts'):
                    scheduler.step()

                if type(scheduler).__name__ == 'ReduceLROnPlateau':
                    scheduler.step(ValError)

        running_loss = 0

        str = 'LR: '
        for param_group in optimizer.param_groups:
            str += repr(param_group['lr']) + ' '
        #print(colored(str, 'blue', attrs=['reverse', 'blink']))
        print('\n'+Fore.BLACK + Back.YELLOW + str + Style.RESET_ALL)

        print('Mode  = ' + CnnMode+ '\n')
        print('NegativeMiningMode='+criterion.Mode)

        if (CnnMode == 'DualEncoder') or (CnnMode == 'PairwiseAsymmetric') or (CnnMode == 'PairwiseSymmetric') \
                or (CnnMode == 'SymmetricEncoder'):
            Case1 = (optimizer.param_groups[0]['lr'] <= (Random2HardLR*1.01)) and (epoch - StartEpoch > WarmUpEpochs)
        else:
            Case1 = (criterion.Mode == 'Random') and (optimizer.param_groups[0]['lr'] <= (Random2HardLR*1.01)) \
                    and (epoch-StartEpoch>WarmUpEpochs)

        if Case1:
            print(Fore.BLACK + Back.GREEN + 'Switching Random->Hardest' + Style.RESET_ALL)
            criterion = OnlineHardNegativeMiningTripletLoss(margin=1, Mode = 'Hardest',device=device)

            if True:
                if False:
                    optimizer = torch.optim.Adam(param_dicts,lr=0, weight_decay=0.0)
                else:
                    optimizer = torch.optim.Adam(
                        [{'params': filter(lambda p: p.requires_grad == True, net.parameters()), 'lr': LearningRate,
                          'weight_decay': 1e-4},
                         {'params': filter(lambda p: p.requires_grad == False, net.parameters()), 'lr': 0, 'weight_decay': 0}],
                        lr=0, weight_decay=0.00)

                UseWarmUp = True
                if UseWarmUp:
                    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=1, total_epoch=WarmUpEpochs)
                    scheduler_warmup.step()

                if (CnnMode == 'DualEncoder') and False:
                    Training_DataLoader = MultiEpochsDataLoader(Training_Dataset, batch_size=2*OuterBatchSize,
                                                                shuffle=True,num_workers=8, pin_memory=True)

                StartEpoch = epoch

            if CnnMode == 'DualEncoder':
                CnnMode = 'DualEncoderHard'

            if type(scheduler).__name__ == 'StepLR':
                scheduler =  StepLR(optimizer, step_size=10, gamma=0.1)

            if type(scheduler).__name__ == 'ReduceLROnPlateau':
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True)

        NumElements = 0
        bar = tqdm(Training_DataLoader, 0, leave=False)
        for i, Data in enumerate(bar):

            net = net.train()

            # get the inputs
            pos1 = Data['pos1']
            pos2 = Data['pos2']

            pos1 = np.reshape(pos1, (pos1.shape[0] * pos1.shape[1], 1, pos1.shape[2], pos1.shape[3]), order='F')
            pos2 = np.reshape(pos2, (pos2.shape[0] * pos2.shape[1], 1, pos2.shape[2], pos2.shape[3]), order='F')

            if UseCorruptedBatch:
                idx1,idx2 = CreateCorruptedBatch(pos1.shape[0], BatchCorruptionRatio)
                pos1 = pos1[idx1,]
                pos2 = pos2[idx2,]


            if (CnnMode == 'PairwiseAsymmetric') or (CnnMode == 'PairwiseSymmetric') or (CnnMode == 'SymmetricEncoder') \
                or (CnnMode == 'AsymmetricEncoder'):

                pos1, pos2 = pos1.to(device), pos2.to(device)

                #with torch.cuda.amp.autocast():
                Embed = net(pos1, pos2)
                loss           = criterion(Embed['Emb1'], Embed['Emb2']) + criterion(Embed['Emb2'], Embed['Emb1'])


            if (CnnMode == 'DualEncoder'):
                pos1, pos2 = pos1.to(device), pos2.to(device)

                Embed1 = net(pos1, pos2,CnnMode)

                idx   = torch.randperm(pos2.shape[0])
                Embed2 = net(pos1, pos2[idx,])

                Label = torch.cat((torch.ones(Embed1['Emb1'].shape[0],dtype=torch.float32),
                                  torch.zeros(Embed2['Emb1'].shape[0],dtype=torch.float32)),0).to(device)

                Embed = torch.cat((Embed1['Emb1'],Embed2['Emb1']),0)

                #loss = BCEloss(Embed.squeeze(),Label).mean()
                if NumGpus == 1:
                    logits,loss = net.AngularLoss(Embed,Label.long())
                else:
                    logits,loss = net.module.AngularLoss(Embed,Label.long())

                loss = loss.mean()
                #loss = Focal(nn.Sigmoid()(Embed.squeeze()), Label)


            if (CnnMode == 'DualEncoderHard'):
                pos1, pos2 = pos1.to(device), pos2.to(device)

                #Focal
                Loss = BCEloss
                Embed1 = net(pos1, pos2,CnnMode,Loss,margin=5)

                #loss = BCEloss(Embed1['Emb1'].squeeze(),Label)
                #loss = (loss*Label).mean()
                #loss = Focal(nn.Sigmoid()(Embed1['Emb1'].squeeze()), Label)
                loss = Embed1['Emb1'].mean()

                NumElements += Embed1['Emb1'].shape[0]



            if (CnnMode == 'Hybrid') or (CnnMode == 'AttenHybrid'):
                pos1, pos2 = pos1.to(device), pos2.to(device)

                # GPUtil.showUtilization()
                #with torch.cuda.amp.autocast():
                Embed = net(pos1, pos2)
                loss = criterion(Embed['Hybrid1'], Embed['Hybrid2']) + criterion(Embed['Hybrid2'],Embed['Hybrid1'])
                loss += criterion(Embed['EmbSym1'], Embed['EmbSym2']) + criterion(Embed['EmbSym2'], Embed['EmbSym1'])
                #loss +=criterion(Embed['EmbAsym1'], Embed['EmbAsym2'])+criterion(Embed['EmbAsym2'], Embed['EmbAsym1'])

            # backward + optimize
            loss.backward()

            clipping_value = 1
            #torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_value)

            optimizer.step()  # Now we can do an optimizer step

            if UseEMA:
                #EMA.update(net.module)
                EMA.update(net)

            # zero the parameter gradients
            optimizer.zero_grad()

            running_loss     += loss.item()


            SchedularUpadteInterval = 100
            if (i % SchedularUpadteInterval == 0) &(i>0):
                print('running_loss: '+repr(running_loss/i)[0:8])
                if CnnMode == 'DualEncoderHard':
                    print('NumElements: ' + repr(NumElements / i)[0:8])



            PrintStep = 1000
            if (((i % PrintStep == 0) or (i * InnerBatchSize >= len(Training_DataLoader) - 1)) and (i > 0)) or TestMode:

                with torch.no_grad():

                    if i > 0:
                        running_loss     /=i
                        running_loss_neg /= i
                        running_loss_pos /= i

                    # val accuracy
                    if CnnMode == 'DualEncoderHard':
                        CnnMode1 = 'DualEncoder'
                    else:
                        CnnMode1 = CnnMode

                    net.eval()

                    Emb = EvaluateDualNets(net, ValSetData[:, :, :, :, 0], ValSetData[:, :, :, :, 1], CnnMode1,device,ValidayionStepSize)

                    if CnnMode1 == 'DualEncoder':
                        x = torch.from_numpy(Emb['Emb1'])

                        if NumGpus == 1:
                            net1.AngularLoss.to("cpu")
                            logits = net1.AngularLoss(x,Mode = 'logits')
                            net1.AngularLoss.to(device)
                        else:
                            net1.module.AngularLoss.to("cpu")
                            logits= net1.module.AngularLoss(x,Mode = 'logits')
                            net1.module.AngularLoss.to(device)

                        Dist = logits[:,0].numpy()
                    else:
                        Dist = np.power(Emb['Emb1'] - Emb['Emb2'], 2).sum(1)

                    ValError,CurrentFPR95 = FPR95Accuracy(Dist, ValSetLabels,FPR = FPR)
                    del Emb
                    ValError *=100

                    # estimate fpr95 threshold
                    if i > 0:
                        print('FPR95: ' + repr(CurrentFPR95)[0:6] + ' Loss= ' + repr(running_loss)[0:6])

                    print('FPR95 changed: ' + repr(FPR95)[0:5])

                    # compute stats
                    if i >= len(Training_DataLoader):
                        TestDecimation1 = 1
                    else:
                        TestDecimation1 = TestDecimation;

                    # test accuracy
                    if CnnMode == 'DualEncoderHard':
                        CnnMode1 = 'DualEncoder'
                    else:
                        CnnMode1 = CnnMode

                    TotalTestError,TotalFPR95 = ComputeTestError(TestData, net, TestDecimation1,FPR,CnnMode1, device, ValidayionStepSize)
                    if UseEMA:
                        EmaTotalTestError, _ = ComputeTestError(TestData, EMA.module, TestDecimation1, FPR, CnnMode1,
                                                                      device, ValidayionStepSize)

                    state = {'epoch': epoch,
                             'state_dict': net.state_dict() if NumGpus == 1 else net.module.state_dict(),
                             #'state_dict': net1.state_dict(),
                             'optimizer_name': type(optimizer).__name__,
                             #'optimizer': optimizer.state_dict(),
                             'optimizer': optimizer,
                             'scheduler_name': type(scheduler).__name__,
                             #'scheduler': scheduler.state_dict(),
                             'scheduler': scheduler,
                             'Description': Description,
                             'LowestError': LowestError,
                             'OuterBatchSize': OuterBatchSize,
                             'InnerBatchSize': InnerBatchSize,
                             'Mode': net.Mode,  ### SHAY CHNAGE net.module.Mode
                             'CnnMode': CnnMode,
                             'NegativeMiningMode': criterion.Mode,
                             'GeneratorMode': GeneratorMode,
                             'Loss': criterion.Mode,
                             'FPR95': FPR95}

                    #if (TotalTestError < LowestError):
                    if (ValError < LowestError):
                        #LowestError = TotalTestError
                        LowestError = ValError


                        print(Back.GREEN +'Best error found and saved: ' + repr(TotalTestError)[0:5]+Style.RESET_ALL)
                        filepath = ModelsDirName + BestFileName + '.pth'
                        torch.save(state, filepath)

                        if False:
                            net2 = MetricLearningCnn(CnnMode, DropoutP)
                            net2,msg = LoadNetworkModel(filepath)
                            net2.to(device)
                            net2.eval()
                            TotalTestError, TotalFPR95 = ComputeTestError(TestData, net2, TestDecimation, FPR, CnnMode,device,ValidayionStepSize)

                    str = '[%d, %5d] loss: %.3f' % (epoch, i, 100 * running_loss) + ' Val Error: ' + repr(ValError)[0:6] + '%'


                    # for DataName in TestData:
                    #   str +=' ' + DataName + ': ' + repr(TestData[DataName]['TestError'])[0:6]
                    str += ' Test FPR95 = ' + repr(TotalFPR95)[0:20] + ' Test Mean: ' + repr(TotalTestError)[0:6] + '%'
                    if UseEMA:
                        str +=  ' EMS Test Mean: ' + repr(EmaTotalTestError)[0:6] + '%'
                    print(str)

                    # save epoch
                    filepath = ModelsDirName + FileName + repr(epoch) + '.pth'
                    torch.save(state, filepath)

            if (i * InnerBatchSize) > (len(Training_DataLoader) - 1):
                bar.clear()
                bar.close()
                break

    print('Finished Training')