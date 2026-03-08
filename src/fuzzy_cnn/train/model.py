from torch import nn
import torch

class CIFAR10ClassifierModel(nn.Module):
    def __init__(self):
        super(CIFAR10ClassifierModel, self).__init__()

        # Two convolutional layers feels like a good place to start with this, because we're doing a kernel size of 3 though
        # we need to bump the spatial width & height back up with padding, otherwise we'd drop some accidentally:
        # output_size = input_size - kernel_size + 1 + 2 * padding

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)

        # The second layer of features is gonna zoom out on the image because we're looking at higher level features now which is nice
        # we make 64 features from those 32 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Because we plug it through 2x2 pooling with a stride of 2, we cut the matrix in half for each pass through,
        # so we go from 32x32 -> 16x16 -> 8x8
        # we don't really care about the specific location of a feature just the general area which we'll build on top of so
        self.fc1 = nn.Linear(64 * 8 * 8, 256) # 4096 -> 256
        self.fc2 = nn.Linear(256, 10)         # 256 -> 10

        # The big squash on the first fully connected layer is good, should cut down on overfitting
        # but we can always bump it back up if it's underfitting
        
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # 32 x 16 x 16
        x = self.pool(torch.relu(self.conv1(x)))
        
        # 64 x 8 x 8
        x = self.pool(torch.relu(self.conv2(x)))
        
        # Flatten each sample to a shape of a 1D array of 4,096 feature values
        x = x.view(x.size(0), -1)

        return self.fc2(torch.relu(self.fc1(x)))