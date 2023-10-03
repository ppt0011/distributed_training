# reference: https://debuggercafe.com/training-resnet18-from-scratch-using-pytorch/

import os
from tqdm import tqdm


import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader

# distributed training
import sys
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


# Data loader with explicit sampler for distributed training
def build_dataloader(batch_size: int):
    # transform
    transform_train = transforms.Compose(
        [transforms.Pad(4),
         transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32),
         transforms.ToTensor()])
    transform_test = transforms.ToTensor()

    # CIFAR-10 dataset
    data_path = os.environ.get('DATA', '../data')
    train_dataset = torchvision.datasets.CIFAR10(root=data_path,
                                                 train=True,
                                                 transform=transform_train,
                                                 download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=data_path,
                                                train=False,
                                                transform=transform_test,
                                                download=True)

    # Data loader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, sampler=DistributedSampler(train_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, sampler=DistributedSampler(test_dataset))
    return train_dataloader, test_dataloader


# Training function.
def train_epoch(epoch: int, 
                num_epochs: int,
                model: nn.Module, 
                optimizer: Optimizer,
                criterion: nn.Module,
                train_dataloader: DataLoader,
                device):
    model.train()
    with tqdm(train_dataloader, desc=f'Epoch [{epoch + 1}/{num_epochs}]') as pbar:
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() #?1

            # Print log info
            pbar.set_postfix({'loss': loss.item()})



def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    init_process_group(backend = "nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def ddp_cleanup():
    destroy_process_group()


# Packaged overall fn to be passed as a param to mp.spawn (Unique for DDP)
def ddp(rank, world_size, num_epochs, learning_rate, batch_size):
    ddp_setup(rank, world_size)
    # resent50
    model = DDP(torchvision.models.resnet18(num_classes=10).cuda())
    train_dataloader, test_dataloader = build_dataloader(batch_size)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    start_epoch = 0
    for epoch in range(start_epoch, num_epochs):
        train_dataloader.sampler.set_epoch(epoch) # shuffling
        train_epoch(epoch, num_epochs, model, optimizer, criterion, train_dataloader, rank)
    
    ddp_cleanup()

if __name__ == '__main__':
    num_epochs = 5
    world_size = 2
    learning_rate = 1e-3 * world_size
    batch_size = 100
    mp.spawn(ddp,
            args=(world_size, num_epochs,learning_rate, batch_size),
            nprocs=world_size)




