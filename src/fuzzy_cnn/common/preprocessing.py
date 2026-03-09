from torchvision import transforms
from fuzzy_cnn.common.config import settings

# Adding a little noise with random crops, flips and a bit of padding for variety so it doens't overfit
def get_train_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(settings.image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(settings.cifar10_mean, settings.cifar10_std),
    ])

# We need to resize these images here because in the real world they could be vastly different e.g. 
# could be 1920x1080 so we want it down to a format that we recognise
def get_inference_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((settings.image_size,settings.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(settings.cifar10_mean, settings.cifar10_std),
    ])