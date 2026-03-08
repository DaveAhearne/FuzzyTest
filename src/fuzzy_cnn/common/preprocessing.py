from torchvision import transforms
from fuzzy_cnn.common.config import CIFAR10_MEAN, CIFAR10_STD, IMAGE_SIZE

# Adding a little noise with random crops, flips and a bit of padding for variety so it doens't overfit
def get_train_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(IMAGE_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

# We need to resize these images here because in the real world they could be vastly different e.g. 
# could be 1920x1080 so we want it down to a format that we recognise
def get_inference_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])