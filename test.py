# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
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
            transforms.Resize((args.img_size, args.img_size)),
            #transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    testset = datasets.ImageFolder(os.path.join(args.test_dir+'/fold'+str(args.foldn), args.dataset), transform=transform_test)
    test_sampler = SequentialSampler(testset)
    test_loader = DataLoader(testset,
                            sampler=test_sampler,
                            batch_size=1,
                            num_workers=4,
                            pin_memory=True) if testset is not None else None
    #print(test_loader)
    test_bar = tqdm(test_loader, desc=f'Testing')
    all_preds, all_label, all_logit = [], [], []
    with torch.no_grad():
        for batch_data in test_bar:
            image, label = batch_data
            image = image.to(device)
            label = label.to(device)

            if args.use_test_aug:
                # test augmentation

                tta_transforms = tta.Compose(
                    [
                        #tta.HorizontalFlip(),
                        #tta.VerticalFlip(),
                        #tta.Rotate90(angles=[0, 90, 270]),
                        #tta.Add(values=[-0.1, -0.05, 0.05, 0.1]),
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

    print(classification_report(all_label[0], all_preds[0], target_names=[str(i) for i in range(args.num_classes)], digits=6))


    ################
    # correlation matrix
    ################

    # cluster data
    import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN

    # apply softmax activation
    #print(all_logit[0])
    #>[[16.308935   -0.07852727 -1.0707934  ...  5.3738036  -0.45179817] [...] ...]
    pred = torch.softmax(torch.tensor(all_logit[0]), dim=1).numpy()
    #print(pred)
    clustering = DBSCAN(eps=1.0, min_samples=2)
    clusters   = clustering.fit_predict(pred.T)
    #print(clusters)
    covariance = np.corrcoef(pred.T)
    #print(covariance)

    # permute correlation coefficient
    index       = np.argsort(-clusters)
    permutation = np.eye(args.num_classes)[index]
    sorted_cov  = covariance
    sorted_cov  = permutation @ covariance @ permutation.T

    # plot
    fig, ax  = plt.subplots(figsize=(18, 12))
    mappable = ax.matshow(sorted_cov, cmap="coolwarm", vmin=-1, vmax=1)

    plt.colorbar(mappable=mappable, ax=ax)
    plt.title(args.model_type)
    plt.show()

    # show the highly correlated classes
    np.where((sorted_cov[:50, :50] > 0.5) & (sorted_cov[:50, :50] < 0.999))
    


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--checkpoint", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for trained ViT models.")
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--test_dir", default='../data',
                        help="Where to do the inference.")
    parser.add_argument("--foldn", type=int,
                        help="Which fold.")
    parser.add_argument("--dataset", default='test',
                        help="What kind of dataset to do the inference.")

    parser.add_argument("--num_classes", default=219, type=int,
                        help="Number of classes")

    # augmentation
    parser.add_argument('--use_imagenet_mean_std', action=argparse.BooleanOptionalAction,
                        help='Whether to use mean and std of imagenet.')

    # tta
    parser.add_argument('--use_test_aug', action=argparse.BooleanOptionalAction,
                        help="Whether to use testing augmentations.")
    parser.add_argument('--rot_degree', type=float, default=10,
                        help='degree of rotation')


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

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Test
    test(args, model)

if __name__ == "__main__":
    main()