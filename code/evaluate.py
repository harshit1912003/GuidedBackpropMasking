import numpy as np
import torch
import torch.nn as nn
import argparse
import os
from torchvision.io import read_image, write_png
import csv
import json


def eval():

    parser = argparse.ArgumentParser()
    parser.add_argument('model_ckpt_path')
    parser.add_argument('test_imgs_dir')
    args = parser.parse_args()

    model_ckpt_path = args.model_ckpt_path
    test_imgs_dir = args.test_imgs_dir
    # print('modelckptpath', model_ckpt_path)

    with open('class_names.json', 'r') as f:
        classnames = json.load(f)


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
            self.downsample1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) 
            self.downsample2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) 
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
            y = self.inpprocess(x)
            y = self.bn1(y)
            y = self.relu(y)
            y = self.blocks1(y)
            y = self.downsample1(y)
            y = self.blocks2(y)
            y = self.downsample2(y)
            y = self.blocks3(y)
            y = nn.AdaptiveAvgPool2d(1)(y)
            y = y.view(y.size(0), -1)
            y = self.fc(y)
            return y

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ResNet(2, 100)
    model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()

    os.makedirs('seg_maps', exist_ok=True)
    csvrows = []

    imgs = sorted([f for f in os.listdir(test_imgs_dir)])

    for img in imgs:
        # print('imgpath;', img)
        imgpath = os.path.join(test_imgs_dir, img)
        imgtnsr = read_image(imgpath).float()/255.0  
        imgtnsr = imgtnsr.to(device).unsqueeze(0)
        # print('imgsize', imgtnsr.size())

        with torch.no_grad():
            output = model(imgtnsr)
        ypred = output.argmax(dim=1).item()

        # print('yprd', classnames[ypred])
        
        csvrows.append([img, classnames[ypred]])

        imgtnsr.requires_grad = True
        output = model(imgtnsr)
        score = output[0, ypred]
        model.zero_grad()
        score.backward()

        grads = imgtnsr.grad.data[0].cpu().numpy()
        print('gradssize', len(grads), len(grads[0]), len(grads[0][0]))
        grads = np.max(grads, axis=0)
        thresh = np.percentile(grads, 80)
        mask = (grads >= thresh).astype(np.float32)
        print('masksize', len(mask), len(mask[0]))

        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        maskMP = torch.nn.functional.max_pool2d(mask, kernel_size=5, stride=1, padding=2)

        maskMP = (maskMP.squeeze().cpu().numpy() * 255).astype(np.uint8)
        masktnsr = torch.from_numpy(maskMP).unsqueeze(0)
        csvpath = os.path.join('seg_maps', img)
        write_png(masktnsr, csvpath)


    csvfile = 'submission.csv'
    with open(csvfile, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'label'])
        writer.writerows(csvrows)

if __name__ == '__main__':
    eval()
