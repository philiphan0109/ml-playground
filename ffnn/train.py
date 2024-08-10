import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

data = torchvision.datasets.MNIST(root = "data/", train = True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=True)

n = next(iter(trainloader))
images, labels = n
print("Images batch shape:", images.shape)
print("Labels batch shape:", labels.shape)
print("Labels:", labels)