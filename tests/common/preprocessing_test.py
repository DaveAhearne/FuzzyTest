import torch
from PIL import Image
from fuzzy_cnn.common.preprocessing import get_train_transforms, get_inference_transforms

def test_train_transforms_output_shape():
    # We make a new fake image in the same shape as our expected input
    image = Image.new("RGB", (32, 32))
    transform = get_train_transforms()
    result = transform(image)

    assert isinstance(result, torch.Tensor)
    
    # Basically, the image should be the same dimensions and type as it went in as
    # only the content of the image is changed
    assert result.shape == (3, 32, 32) # channels, height, width  
    assert result.dtype == torch.float32

def test_inference_transforms_output_shape():
    # inference is different it can be somehing of any size
    image = Image.new("RGB", (256, 128))
    transform = get_inference_transforms()
    result = transform(image)

    assert isinstance(result, torch.Tensor)
    
    # our input image should get resized to 32x32
    assert result.shape == (3, 32, 32) # channels, height, width  
    assert result.dtype == torch.float32