import os, torch
import torchvision.transforms as transforms
from torchvision import datasets
import pandas as pd
import numpy as np
# os.makedirs("./data/MNIST", exist_ok=True)

trainset = datasets.MNIST(
        "../data",
        # "./data",
        train=True,
        download=False,
        transform=transforms.Compose(
            [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    )

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True,
)
testset = datasets.MNIST(
        "../data",
        # "./data",
        train=False,
        download=False,
        transform=transforms.Compose(
            [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    )

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=64,
    shuffle=True,
)

train = pd.DataFrame()
for i, (imgs, labels) in enumerate(trainloader):
    batchsize = imgs.shape[0]
    print(i, len(trainset)/64.0)
    which = labels<=1
    data = torch.cat([imgs.reshape(batchsize, 28*28)[which], labels.reshape(batchsize, 1)[which].float()], dim=1)
    data = pd.DataFrame(data.numpy())
    train = pd.concat([train, data], axis=0)
l = list(range(28*28))
l.append('y')
train.columns = l
train.to_csv('../data/MNIST_CNN_train.csv', index=False)
print(train.shape)

test = pd.DataFrame()
for i, (imgs, labels) in enumerate(testloader):
    batchsize = imgs.shape[0]
    print(i, len(testset)/64.0)
    which = labels<=1
    data = torch.cat([imgs.reshape(batchsize, 28*28)[which], labels.reshape(batchsize, 1)[which].float()], dim=1)
    data = pd.DataFrame(data.numpy())
    test = pd.concat([test, data], axis=0)
l = list(range(28*28))
l.append('y')
test.columns = l
test.to_csv('../data/MNIST_CNN_test.csv', index=False)
print(test.shape)