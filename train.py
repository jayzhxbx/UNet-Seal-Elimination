import torch
import torch.nn as nn
from torch import optim
import os
from unet import UNet
from datasets import UNetDataset
import transforms as Transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import config


def train():
    transforms = [
        Transforms.RondomFlip(),
        Transforms.RandomRotate(15),
        Transforms.Log(0.5),
        Transforms.Blur(0.2),
        Transforms.ToGray(),
        Transforms.ToTensor()
    ]
    train_dataset = UNetDataset('./data/train/', './data/train_cleaned/', transform=transforms)
    train_dataLoader = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)

    valid_dataset = UNetDataset('./data/valid/', './data/valid_cleaned/', transform=transforms)
    valid_dataLoader = DataLoader(dataset=valid_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)

    net = UNet(n_channels=config.n_channels, n_classes=config.n_classes).to(config.device)
    writer = SummaryWriter()
    optimizer = optim.Adam(net.parameters(), lr=config.LR)
    if config.n_classes > 1:
        loss_func = nn.CrossEntropyLoss().to(config.device)
    else:
        loss_func = nn.BCEWithLogitsLoss().to(config.device)
    best_loss = float('inf')

    if os.path.exists(config.weight_with_optimizer):
        checkpoint = torch.load(config.weight_with_optimizer, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('load weight')

    for epoch in range(config.EPOCH):
        train_loss = 0
        net.train()
        for step, (batch_x, batch_y) in enumerate(train_dataLoader):
            batch_x = batch_x.to(device=config.device)
            batch_y = batch_y.squeeze(1).to(device=config.device)
            output = net(batch_x)
            loss = loss_func(output, batch_y)
            train_loss += loss.item()
            if loss < best_loss:
                best_loss = loss
                torch.save({'net': net.state_dict(), 'optimizer': optimizer.state_dict()},
                           config.best_model_with_optimizer)
                torch.save({'net': net.state_dict()}, config.best_model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        net.eval()
        eval_loss = 0
        for step, (batch_x, batch_y) in enumerate(valid_dataLoader):
            batch_x = batch_x.to(device=config.device)
            batch_y = batch_y.squeeze(1).to(device=config.device)
            output = net(batch_x)
            valid_loss = loss_func(output, batch_y)
            eval_loss += valid_loss.item()

        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("eval_loss", eval_loss, epoch)
        print("*" * 80)
        print('epoch: %d | train loss: %.4f | valid loss: %.4f' % (epoch, train_loss, eval_loss))
        print("*" * 80)

        if (epoch + 1) % 10 == 0:
            torch.save({
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }, config.weight_with_optimizer)
            torch.save({
                'net': net.state_dict()
            }, config.weight)
            print('saved')

    writer.close()


if __name__ == '__main__':
    train()
