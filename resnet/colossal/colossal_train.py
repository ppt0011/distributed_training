'''
references:
- torch DDP: https://debuggercafe.com/training-resnet18-from-scratch-using-pytorch/
- colossalAI: https://github.com/hpcaitech/ColossalAI/blob/main/examples/images/resnet/train.py
'''

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.booster.plugin.dp_plugin_base import DPPluginBase
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device


# colossalAI config args
def add_argument():

    parser = argparse.ArgumentParser()
    # FIXME(ver217): gemini is not supported resnet now
    parser.add_argument('-p',
                        '--plugin', 
                        type=str,
                        default='torch_ddp',
                        choices=['torch_ddp', 'torch_ddp_fp16', 'low_level_zero'],
                        help="plugin to use")
    parser.add_argument('-r', '--resume', type=int, default=-1, help="resume from the epoch's checkpoint")
    parser.add_argument('-c', '--checkpoint', type=str, default='./checkpoint', help="checkpoint directory")
    parser.add_argument('-i', '--interval', type=int, default=5, help="interval of saving checkpoint")
    parser.add_argument('--target_acc',
                        type=float,
                        default=None,
                        help="target accuracy. Raise exception if not reached")
    parser.add_argument("--local-rank", type=int)

    args = parser.parse_args()
    return args 

# modify data loader
def build_dataloader(batch_size: int, coordinator: DistCoordinator, plugin: DPPluginBase):
    
    # Transform
    transform_train = transforms.Compose(
        [transforms.Pad(4),
         transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32),
         transforms.ToTensor()])
    transform_test = transforms.ToTensor()

    # CIFAR-10 dataset
    data_path = os.environ.get('DATA', '../data')
    with coordinator.priority_execution():
        train_dataset = torchvision.datasets.CIFAR10(root=data_path,
                                                     train=True,
                                                     transform=transform_train,
                                                     download=True)
        test_dataset = torchvision.datasets.CIFAR10(root=data_path,
                                                    train=False,
                                                    transform=transform_test,
                                                    download=True)

    # Data loader
    train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataloader = plugin.prepare_dataloader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_dataloader, test_dataloader

# Training function.
def train_epoch(epoch: int, 
                model: nn.Module, 
                criterion: nn.Module,
                train_dataloader: DataLoader,
                booster: Booster,
                coordinator: DistCoordinator):
    model.train()
    with tqdm(train_dataloader, desc=f'Epoch [{epoch + 1}/{NUM_EPOCHS}]', disable=not coordinator.is_master()) as pbar:
        for images, labels in pbar:
            images = images.cuda()
            labels = labels.cuda()
            #if target_dtype != None:
            #    images = images.to(target_dtype)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            booster.backward(loss, optimizer)
            optimizer.step()
            optimizer.zero_grad()

            # Print log info
            pbar.set_postfix({'loss': loss.item()})

# ==============================
# Launch Distributed Environment
# ==============================
colossalai.launch_from_torch(config={})
coordinator = DistCoordinator()

# ==============================
# Prepare Hyperparameters
# ==============================
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
BATCH_SIZE = 100

# update the learning rate with linear scaling
# old_gpu_num / old_lr = new_gpu_num / new_lr
LEARNING_RATE *= coordinator.world_size

# ==============================
# Instantiate Plugin and Booster
# ==============================
args = add_argument()
booster_kwargs = {}
if args.plugin == 'torch_ddp_fp16':
    booster_kwargs['mixed_precision'] = 'fp16'
if args.plugin.startswith('torch_ddp'):
    plugin = TorchDDPPlugin()
elif args.plugin == 'gemini':
    plugin = GeminiPlugin(placement_policy='cuda', strict_ddp_mode=True, initial_scale=2**5)
elif args.plugin == 'low_level_zero':
    plugin = LowLevelZeroPlugin(initial_scale=2**5)

booster = Booster(plugin=plugin, **booster_kwargs)

# ==============================
# Prepare Dataloader
# ==============================
train_dataloader, test_dataloader = build_dataloader(100, coordinator, plugin)

# ====================================
# Prepare model, optimizer, criterion
# ====================================
# resent50
model = torchvision.models.resnet18(num_classes=10)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = HybridAdam(model.parameters(), lr=LEARNING_RATE)
#lr_scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=1 / 3)

# ==============================
# Boost with ColossalAI
# ==============================
model, optimizer, criterion, _, _ = booster.boost(model,
                                                    optimizer,
                                                    criterion=criterion)

start_epoch = 0
for epoch in range(start_epoch, NUM_EPOCHS):
    train_epoch(epoch, model, criterion, train_dataloader, booster, coordinator)
