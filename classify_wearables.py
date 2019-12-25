import torch
from torchvision import datasets, transforms
import helper

# transform to normalize data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# download and load training data
trainset = datasets.FashionMNIST("~/.pytorch/F_MNIST_data", download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# download and load test data
testset = datasets.FashionMNIST("~/.pytorch/F_MNIST_data", download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

image, label = next(iter(trainloader))
helper.imshow(image[0,:])
