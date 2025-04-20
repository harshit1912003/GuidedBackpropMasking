import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch import optim
import argparse
import time
import json



def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data_dir')
    parser.add_argument('model_ckpt_dir')
    args = parser.parse_args()

    train_data_dir = args.train_data_dir
    model_ckpt_dir = args.model_ckpt_dir
    print('arg1', train_data_dir)
    print('arg2', model_ckpt_dir)

    class ResidualBlock(nn.Module):
        def __init__(self, numchannels):
            super(ResidualBlock, self).__init__()

            self.conv1 = nn.Sequential(
                nn.Conv2d(numchannels, numchannels,
                        kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm2d(numchannels),
                nn.ReLU()
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(numchannels, numchannels,
                        kernel_size=3, stride=1, padding='same'),
                nn.BatchNorm2d(numchannels),
                nn.ReLU()
            )

            self.relu = nn.ReLU()

        def forward(self, x):
            residual = x

            out = self.conv1(x)
            out = self.conv2(out)

            out = out + residual
            out = self.relu(out)

            return out

    class ResNet(nn.Module):
        def __init__(self, n, r):
            super(ResNet, self).__init__()

            self.inpprocess = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding='same')

            self.downsample1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) #--> 112,112,64
            self.downsample2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) #--> 56,56,128

            self.N = n
            self.R = r

            self.blocks1= self.__blocks__(32)
            self.blocks2= self.__blocks__(64)
            self.blocks3= self.__blocks__(128)

            self.relu = nn.ReLU()
            self.bn1 = nn.BatchNorm2d(32)
            self.fc = nn.Linear(128, self.R)


        def __blocks__(self, numfilters):
            layers=[]
            for _ in range(1, self.N):
                layers.append(ResidualBlock(numfilters))
            return nn.Sequential(*layers)

        def forward(self, x):
            # x:(224, 224, 3)
            y = self.inpprocess(x)
            y = self.bn1(y)
            y = self.relu(y)
            # y =(224, 224, 32)

            y = self.blocks1(y)
            # y =(224, 224, 32)

            y = self.downsample1(y)
            # y =(112, 112, 64)
            y = self.blocks2(y)
            # y =(112, 112, 64)

            y = self.downsample2(y)
            # y =(56, 56, 128)
            y = self.blocks3(y)
            # y =(56, 56, 128)

            y = nn.AdaptiveAvgPool2d(1)(y)
            # y = 128

            y = y.view(y.size(0), -1)
            # print(y.size())
            y = self.fc(y)
            return y


    trainDset = datasets.ImageFolder(root=args.train_data_dir, transform=transforms.ToTensor())
    trainLoader = torch.utils.data.DataLoader(trainDset, batch_size=128, shuffle=True, num_workers=1)
    r = len(trainDset.classes)
    print('r', r)
    model = ResNet(2,r)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    s = time.time()
    model.train()
    curloss = 0
    for img, label in trainLoader:
        img = img.to(device)
        label = label.to(device)

        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        curloss += loss.item() * img.size(0)
    e = time.time()

    time1eph = e-s
    print('time1eph', time1eph)

    numeph = 3600 / time1eph
    # print('numeph', numeph)
    numeph -= 2
    # print('numeph', numeph)
    numeph *= 0.8
    # print('numeph', numeph)
    numeph = int(numeph)
    # print('numeph', numeph)

    # print('numeph', numeph)
    # print('time1eph', time1eph)

    for epoch in range(numeph):
        s = time.time()
        model.train()
        curloss = 0
        for img, label in trainLoader:
            img = img.to(device)
            label = label.to(device)

            output = model(img)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            curloss += loss.item() * img.size(0)
        e = time.time()

        print('loss:', curloss, 'time:', e-s)


    ckpt_path = f"{args.model_ckpt_dir.rstrip('/')}/resnet_model.pth" 
    # print('ckpt_path', ckpt_path)
    torch.save(model.state_dict(), ckpt_path)

    class_names_path = 'class_names.json'
    with open(class_names_path, 'w') as f:
        json.dump(trainDset.classes, f)

if __name__ == '__main__':
    train()
