from functools import lru_cache
import pytorch_lightning as pl
from pytorch_lightning import LightningModule,Trainer, seed_everything
import torch #do I need this? was in link you sent me for resnets
import torch.nn as nn
import torch.nn.functional as F
import argparse
import math
import os
import shutil
import time
#from logging import getLogger

#These two lines from playground
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim

from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule

AVAIL_GPUS = min(1, torch.cuda.device_count())

#logger = getLogger()

parser = argparse.ArgumentParser(description="Implementation of SwAV")

#########################
#### data parameters ####
#########################
parser.add_argument("--dataset", type=str, default="imagenet",
                    help="path to dataset repository")
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")

#########################
## swav specific params #
#########################
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")
parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss")
parser.add_argument("--epsilon", default=0.05, type=float,
                    help="regularization parameter for Sinkhorn-Knopp algorithm")
parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                    help="number of iterations in Sinkhorn-Knopp algorithm")
parser.add_argument("--feat_dim", default=128, type=int,
                    help="feature dimension")
parser.add_argument("--nmb_prototypes", default=3000, type=int,
                    help="number of prototypes")
parser.add_argument("--queue_length", type=int, default=0,
                    help="length of the queue (0 for no queue)")
parser.add_argument("--epoch_queue_starts", type=int, default=15,
                    help="from this epoch, we start using a queue")

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=64, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                    help="freeze the prototypes during this many iterations from the start")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

#########################
#### other parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--hidden_mlp", default=2048, type=int,
                    help="hidden layer dimension in projection head")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=25,
                    help="Save the model periodically")
parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                    https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")


##1 
global args
args = parser.parse_args()
#init_distributed_mode(args)
#fix_random_seeds(args.seed)
seed_everything(args.seed)
#logger, training_stats = initialize_exp(args, "epoch", "loss")
##########################

#returns either cifar 10 or imagenet data module from specified data path
def create_data_module(dataset):
    if dataset == 'imagenet':
        return ImagenetDataModule(
            data_dir='c',
            batch_size=args.batch_size,
            num_workers=args.workers)
    elif dataset == 'cifar10':
        return CIFAR10DataModule(
            data_dir='/om2/user/opalinav/FieteLab-Nonparametric-SwAV/non_parametric_swav/cifar-10-batches-py',
            batch_size=args.batch_size,
            num_workers=args.workers)
    else:
        raise Exception("Only CIFAR 10 and Imagenet allowed, please check your data path")

#builds resnet model based on specified resnet nn
def build_model(arch: str, pretrained: bool = False)  -> torch.nn.Module:
    #pretrained true or false?
    model = torch.hub.load('pytorch/vision:v0.10.0', arch, pretrained=pretrained)
    return model



data_module = create_data_module(args.dataset)
model = build_model(arch=args.arch)




class LitModel(pl.LightningModule):
    def __init__(self,
                model:  torch.nn.Module,
                lr: float = 4.8,
                weight_decay: float = 1e-6):

        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        #8
        with torch.no_grad():
            w = self.model.module.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.model.module.prototypes.weight.copy_(w)

        # ============ multi-res forward passes ... ============
        embedding, output = self.model(x)

       

        return (embedding, output)

    @staticmethod
    def distributed_sinkhorn(out):
        Q = torch.exp(out / args.epsilon).t() # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * args.world_size # number of samples to assign
        K = Q.shape[0] # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(args.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

    def training_step(self, batch, batch_idx):
        embedding, output = self.forward(batch)  
        embedding = embedding.detach()
        bs = x[0].size(0)    
            
        loss = 0
        for i, crop_id in enumerate(args.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                # # time to use the queue
                # if queue is not None:
                #     if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                #         use_the_queue = True
                #         out = torch.cat((torch.mm(
                #             queue[i],
                #             model.module.prototypes.weight.t()
                #         ), out))
                #     # fill the queue
                #     queue[i, bs:] = queue[i, :-bs].clone()
                #     queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                # get assignments
                q = self.distributed_sinkhorn(out)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                x = output[bs * v: bs * (v + 1)] / args.temperature
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            loss += subloss / (np.sum(args.nmb_crops) - 1)
        loss /= len(args.crops_for_assign)
        return loss

    # def training_step_end(self, batch_parts):
    #     #what should I put for batch_parts? args.gpu_to_work_on

    #     # predictions from each GPU
    #     predictions = batch_parts["pred"]
    #     # losses from each GPU
    #     losses = batch_parts["loss"]

    #     gpu_0_prediction = predictions[0]
    #     gpu_1_prediction = predictions[1]

    #     # do something with both outputs
    #     return (losses[0] + losses[1]) / 2

    def configure_optimizers(self):
        ####3
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=args.base_lr,
            momentum=0.9,
            weight_decay=args.wd)

        # optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
        # warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
        # iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
        # cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
        #                     math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
        # lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        #logger.info("Building optimizer done.")
        #I added
        return optimizer


trainer = Trainer(
    progress_bar_refresh_rate=10,
    max_epochs=2,
    gpus=AVAIL_GPUS,
    logger=TensorBoardLogger("lightning_logs/", name="resnet"),
    callbacks=[LearningRateMonitor(logging_interval="step")],
)
lightning_system = LitModel(model=model)
trainer.fit(lightning_system, data_module)
trainer.test(lightning_system, datamodule=data_module)
