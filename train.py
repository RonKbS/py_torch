import torch
from torch import nn
from torchvision import datasets, transforms

# Define transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load training data
trainset = datasets.MNIST(
    "~/.pytorch/MNINST_data/",
    download=True,
    train=True,
    transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


'''
# build feed-forward network
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# Define the loss
criterion = nn.CrossEntropyLoss()

# Get unser data
images, labels = next(iter(trainloader))
# Flatten images
images = images.view(images.shape[0], -1)

# Forward pass, get our logits
logits = model(images)
# Calculate the loss with the logits and labels
loss = criterion(logits, labels)

print(loss)
'''


# ~build model with log-softmax output using nn.LogSoftmax
    # or F.log_softmax. Actual probabilities then gotten with
    # torch.exp(output). For this case, negative-log-likelihood-loss
    # nn.NLLLoss is more appropriate

# model returnning log-softmax as output
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    # removing the layer below returns a 2.3299 loss, much closer to that shown
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.LogSoftmax(dim=1)
)

criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

# forward-pass to get log-probabilities
logps = model(images)

# calculate loss with logps and labels
loss = criterion(logps, labels)

# print(loss)

print("Before backward pass : \n", model[0].weight.grad)

loss.backward()

print("After backward pass: \n", model[0].weight.grad)
