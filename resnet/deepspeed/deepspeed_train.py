'''
references:
- torch DDP: https://debuggercafe.com/training-resnet18-from-scratch-using-pytorch/
- deepSpeed: https://github.com/microsoft/DeepSpeedExamples/blob/master/training/cifar/cifar10_deepspeed.py
'''

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import argparse
import deepspeed
from deepspeed.accelerator import get_accelerator

# deepspeed config args
def add_argument():

    parser = argparse.ArgumentParser(description='CIFAR')

    #data
    # cuda
    parser.add_argument('--with_cuda',
                        default=False,
                        action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema',
                        default=False,
                        action='store_true',
                        help='whether use exponential moving average')

    # train
    parser.add_argument('-b',
                        '--batch_size',
                        default=96,
                        type=int,
                        help='mini-batch size (default: 100)')   
    parser.add_argument('-e',
                        '--epochs',
                        default=5,
                        type=int,
                        help='number of total epochs (default: 5)')   
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--log-interval',
                        type=int,
                        default=2000,
                        help="output logging information at a given interval")
    parser.add_argument('--moe',
                        default=False,
                        action='store_true',
                        help='use deepspeed mixture of experts (moe)')
    parser.add_argument('--ep-world-size',
                        default=1,
                        type=int,
                        help='(moe) expert parallel world size')
    parser.add_argument('--num-experts',
                        type=int,
                        nargs='+',
                        default=[
                            1,
                        ],
                        help='number of experts list, MoE related.')
    parser.add_argument(
        '--mlp-type',
        type=str,
        default='standard',
        help=
        'Only applicable when num-experts > 1, accepts [standard, residual]')
    parser.add_argument('--top-k',
                        default=1,
                        type=int,
                        help='(moe) gating top 1 and 2 supported')
    parser.add_argument(
        '--min-capacity',
        default=0,
        type=int,
        help=
        '(moe) minimum capacity of an expert regardless of the capacity_factor'
    )
    parser.add_argument(
        '--noisy-gate-policy',
        default=None,
        type=str,
        help=
        '(moe) noisy gating (only supported with top-1). Valid values are None, RSample, and Jitter'
    )
    parser.add_argument(
        '--moe-param-group',
        default=False,
        action='store_true',
        help=
        '(moe) create separate moe param groups, required when using ZeRO w. MoE'
    )
    parser.add_argument(
        '--dtype',
        default='bf16',
        type=str,
        choices=['bf16', 'fp16', 'fp32'],
        help=
        'Datatype used for training'
    )
    parser.add_argument(
        '--stage',
        default=0,
        type=int,
        choices=[0, 1, 2, 3],
        help=
        'Datatype used for training'
    )

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args 


# Training function.
def train_epoch(epoch: int, 
                num_epochs: int,
                model: nn.Module, 
                criterion: nn.Module,
                train_dataloader: DataLoader,
                device,
                model_engine, 
                target_dtype):
    model.train()
    with tqdm(train_dataloader, desc=f'Epoch [{epoch + 1}/{num_epochs}]') as pbar:
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            if target_dtype != None:
                images = images.to(target_dtype)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            model_engine.backward(loss)
            model_engine.step()

            # Print log info
            pbar.set_postfix({'loss': loss.item()})


if __name__ == '__main__':
   
    # Initialize DeepSpeed to use the following features
    # 1) Distributed model
    # 2) Distributed data loader
    # 3) DeepSpeed optimizer

    deepspeed.init_distributed()

    args = add_argument()

    ds_config = {
    "train_batch_size": 96,
    "steps_per_print": 2000,
    "optimizer": {
        "type": "Adam",
        "params": {
        "lr": 0.001,
        "betas": [
            0.8,
            0.999
        ],
        "eps": 1e-8,
        "weight_decay": 3e-7
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
        "warmup_min_lr": 0,
        "warmup_max_lr": 0.001,
        "warmup_num_steps": 1000
        }
    },
    "gradient_clipping": 1.0,
    "prescale_gradients": False,
    "bf16": {
        "enabled": args.dtype == "bf16"
    },
    "fp16": {
        "enabled": args.dtype == "fp16",
        "fp16_master_weights_and_grads": False,
        "loss_scale": 0,
        "loss_scale_window": 500,
        "hysteresis": 2,
        "min_loss_scale": 1,
        "initial_scale_power": 15
    },
    "wall_clock_breakdown": False,
    "zero_optimization": {
        "stage": args.stage,
        "allgather_partitions": True,
        "reduce_scatter": True,
        "allgather_bucket_size": 50000000,
        "reduce_bucket_size": 50000000,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "cpu_offload": False
    }
    }

    # model
    model = torchvision.models.resnet18(num_classes=10).cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters()) # note that we need to pre-process parameters

    # data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='../data',
                                            train=True,
                                            download=True,
                                            transform=transform)

    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=args, model=model, model_parameters=parameters, training_data=trainset, config=ds_config)

    # distributed setup
    local_device = get_accelerator().device_name(model_engine.local_rank)
    local_rank = model_engine.local_rank

    # For float32, target_dtype will be None so no datatype conversion needed
    target_dtype = None
    if model_engine.bfloat16_enabled():
        target_dtype=torch.bfloat16
    elif model_engine.fp16_enabled():
        target_dtype=torch.half

    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    num_epochs = 5
    for epoch in range(start_epoch, num_epochs):
        train_epoch(epoch, num_epochs, model, criterion, trainloader, local_device, model_engine, target_dtype)


