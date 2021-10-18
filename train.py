
import os
import time
import random
import numpy as np
from glob import glob
from argparse import ArgumentParser
import cv2
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from data import load_data, KvasirDataset
from utils import (
    seeding,shuffling, make_channel_first, make_channel_last, create_dir, epoch_time, print_and_save
)
from model import CompNet
from loss import DiceLoss, DiceBCELoss, IoUBCELoss

def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0

    model.train()
    for i, (x, y) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        yp = model(x)
        loss = loss_fn(yp, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data', required=True)
    args = parser.parse_args()
    
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files")

    """ Training logfile """
    train_log_path = "files/train_log.txt"
    if os.path.exists(train_log_path):
        print("Log file exists")
    else:
        train_log = open("files/train_log.txt", "w")
        train_log.write("\n")
        train_log.close()

    """ Load dataset """
    # train_x = sorted(glob("new_data/train/image/*"))
    # train_y = sorted(glob("new_data/train/mask/*"))
    #
    # valid_x = sorted(glob("new_data/test/image/*"))
    # valid_y = sorted(glob("new_data/test/mask/*"))

    path = args.data
    (train_x, train_y) = load_data(path)

#     train_x, train_y = shuffling(train_x, train_y)
    data_str = f"Dataset Size:\nTrain: {len(train_x)}"
    print_and_save(train_log_path, data_str)

    """ Hyperparameters """
    size = (512, 512)
    batch_size = 10
    num_epochs = 20
    lr = 1e-4
    checkpoint_path = "files/checkpoint.pth"

    """ Dataset and loader """
    train_dataset = KvasirDataset(train_x, train_y, size)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    """ Model """
    device = torch.device('cuda')
    model = CompNet()
    # model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()
    # loss_fn = IoUBCELoss()
    loss_name = "BCE Dice Loss"

    data_str = f"Hyperparameters:\nImage Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    """ Training the model. """
    best_valid_loss = float('inf')

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        scheduler.step(train_loss)

        torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        print_and_save(train_log_path, data_str)
