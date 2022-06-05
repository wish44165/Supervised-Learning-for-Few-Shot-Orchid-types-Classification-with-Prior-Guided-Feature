# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import csv
import random
from matplotlib import pyplot as plt
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from sklearn.metrics import classification_report
from sklearn.cluster import DBSCAN

import ttach as tta


logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    model = VisionTransformer(config, args.img_size, zero_head=False, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.checkpoint))
    model.to(args.device)
    model.eval()
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def test(args, model):
    """ Train the model """

    ################ 1. select mean and std ################
    if args.use_imagenet_mean_std:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    # Prepare dataset
    if args.use_test_aug:
        transform_test = transforms.Compose([
            #transforms.Resize((args.img_size, args.img_size)),
            #transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        transform_test = transforms.Compose([
            #transforms.Resize((args.img_size, args.img_size)),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    testset = datasets.ImageFolder(os.path.join(args.test_dir, args.dataset), transform=transform_test)

    imgList = [list(testset.imgs[i])[0].split('/')[-1] for i in range(len(testset))]
    #print(imgList)
    
    test_sampler = SequentialSampler(testset)
    
    test_loader = DataLoader(testset,
                            sampler=test_sampler,
                            batch_size=1,
                            num_workers=4,
                            pin_memory=True) if testset is not None else None

    test_bar = tqdm(test_loader, desc=f'Testing')
    all_preds, all_label, all_logit = [], [], []
    with torch.no_grad():
        for batch_data in test_bar:
            image, label = batch_data
            image = image.to(device)
            label = label.to(device)

            if args.use_test_aug:    ################ 2. select tta modules ################
                # test augmentation

                tta_transforms = tta.Compose(
                    [
                        tta.HorizontalFlip(),
                        #tta.VerticalFlip(),
                        #tta.Rotate90(angles=[0, 90, 180, 270]),
                        #tta.Add(values=[-0.05, 0.05]),
                        #tta.Multiply(factors=[0.95, 1, 1.05]),
                        tta.FiveCrops(args.img_size, args.img_size)
                    ]
                )

                interPreds = []
                for transformer in tta_transforms: # custom transforms or e.g. tta.aliases.d4_transform() 
                    
                    # augment image
                    augmented_image = transformer.augment_image(image)
                    #print(augmented_image.shape)
                    #>torch.Size([1, 3, 480, 480])

                    # pass to model
                    model_output = model(augmented_image)
                    #print(model_output[0].shape)
                    #>torch.Size([1, 219])
                    
                    # reverse augmentation for mask and label
                    #deaug_mask = transformer.deaugment_mask(model_output['mask'])
                    #deaug_label = transformer.deaugment_label(model_output['label'])
                    
                    # save results
                    interPreds.append(model_output[0])


                for i in range(len(interPreds)):
                    if i==0:
                        logits = interPreds[i]
                    else:
                        logits = torch.add(logits, interPreds[i])
                
                logits /= len(interPreds)
                #print(logits.shape)
                #>torch.Size([1, 219])
                preds = torch.argmax(logits, dim=-1)

            else:
                logits = model(image)[0]
                #print(logits.shape)
                #>torch.Size([1, 219])

                preds = torch.argmax(logits, dim=-1)
            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(label.detach().cpu().numpy())
                all_logit.append(logits.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(
                    all_preds[0], preds.detach().cpu().numpy(), axis=0
                )
                all_label[0] = np.append(
                    all_label[0], label.detach().cpu().numpy(), axis=0
                )
                all_logit[0] = np.append(
                    all_logit[0], logits.detach().cpu().numpy(), axis=0
                )
        test_bar.close()

    #print(classification_report(all_label[0], all_preds[0], target_names=[str(i) for i in range(args.num_classes)], digits=6))

    # covariance matrix
    #print(all_preds[0])
    #print(all_preds[0].reshape(-1, 1))
    #print(all_preds[0].reshape(args.num_classes, 2))
    #print(all_label[0].reshape(args.num_classes, 2))
    #print(np.corrcoef(all_preds[0].reshape((args.num_classes, 2)), all_label[0].reshape((args.num_classes, 2))))
    return all_preds[0], all_label[0], all_logit[0], imgList
    


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--model_type",
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--checkpoint", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for trained ViT models.")
    parser.add_argument("--img_size", default=224, type=str,
                        help="Resolution size")
    parser.add_argument("--test_dir", default='../data/fold1/test',    ################ 3. change to test path ################
                        help="Where to do the inference.")
    parser.add_argument("--dataset", default='',
                        help="What kind of dataset to do the inference.")

    parser.add_argument("--num_classes", default=219, type=int,
                        help="Number of classes")

    # mean and std
    parser.add_argument('--use_imagenet_mean_std', action=argparse.BooleanOptionalAction,
                        help='Whether to use mean and std of imagenet.')

    # tta
    parser.add_argument('--use_test_aug', action=argparse.BooleanOptionalAction,
                        help="Whether to use testing augmentations.")

    parser.add_argument("--local_rank", type=int, default=-1,
                            help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    args = parser.parse_args()


    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)



    ################
    # ensemble
    ################
    def softmax(x):
        
        x = x - np.max(x)
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    # Model & Tokenizer Setup
    models = args.model_type
    models = models[1:-1]
    modelTypes = models.split(',')
    #print(modelTypes)
    #>['ViT-B_16', 'ViT-B_16']

    cp = args.checkpoint
    cp = cp[1:-1]
    cpFiles = cp.split(',')
    #print(cpFiles)
    #>['results/ViT-B_16_1/orchid_ViT-B_16_checkpoint.bin', 'results/ViT-B_16_1/orchid_ViT-B_16_checkpoint.bin']

    imgSize = args.img_size
    imgSize = imgSize[1:-1]
    imgSize = imgSize.split(',')
    imgSizes = [int(a) for a in imgSize]
    #print(imgSizes)
    #>[384, 384]


    imgNum = 438    ################ 4. change to test number ################
    #print(imgNum)]

    predictList = [[] for i in range(imgNum)]
    Logits = []

    for i in range(len(modelTypes)):
        args.model_type = modelTypes[i]
        args.checkpoint = cpFiles[i]
        args.img_size = imgSizes[i]

        args, model = setup(args)

        # Test
        pred, groundTruth, lt, imgList = test(args, model)
        #print(np.shape(lt[i]))
        #>(219,)

        for j in range(imgNum):
            predictList[j].append(int(pred[j]))

            if i==0:
                Logits.append(softmax(lt[j]))
            Logits[j]+=softmax(lt[j])


    ################
    # vote ensemble
    ################
    #print(predictList)
    #>[[0, 0], [0, 0], [1, 1], [1, 1], [2, 2], [104, 104], [3, 3], [3, 3], [15, 15], [4, 4], [5, 5], [5, 5], [6, 6] ......
    ##[[model1, model2], [model1, model2] ......
    ensemblePred = np.array([np.argmax(np.bincount(a)) for a in predictList])
    #print(ensemblePred)
    #>[  0   0   1   1   2 104   3   3  15   4   5   5   6   6 ......
    print(ensemblePred)

    with open('./submit_voteEnsemble.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'category'])
        for i in range(imgNum):
            l = [imgList[i], ensemblePred[i]]
            writer.writerow(l)

    ################ 5-1. comment while submit ################
    print(classification_report(groundTruth, ensemblePred, target_names=[str(i) for i in range(args.num_classes)], digits=6))


    ################
    # mean ensemble
    ################
    ensembleLogits = np.array([np.argmax(a) for a in Logits])
    print(ensembleLogits)

    with open('./submit_meanEnsemble.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'category'])
        for i in range(imgNum):
            l = [imgList[i], ensembleLogits[i]]
            writer.writerow(l)

    ################ 5-2. comment while submit ################
    print(classification_report(groundTruth, ensembleLogits, target_names=[str(i) for i in range(args.num_classes)], digits=6))



if __name__ == "__main__":
    main()