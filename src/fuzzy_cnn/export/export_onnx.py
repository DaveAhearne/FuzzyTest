import torch
from fuzzy_cnn.common.config import CHECKPOINT_DIR, ONNX_MODEL_PATH
from fuzzy_cnn.common.io import load_checkpoint
from fuzzy_cnn.train.model import CIFAR10ClassifierModel

def export_onnx():
    # The dummy tensor in the shape that we expect the input
    dummy_input = torch.randn(1,3,32,32) # 1 image, 3 channels, 32 x 32 pixels    

    # Make a new model, load in all the weights etc
    # we also set it back to eval mode (makes the weights behave properly)
    model = CIFAR10ClassifierModel()
    load_checkpoint(CHECKPOINT_DIR / "final.pt", model)
    model.eval()

    # Make the dir if it doesn't exist just incase
    ONNX_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,                          
        dummy_input,                    
        ONNX_MODEL_PATH,         
        # These are effectively the names for the schema in and out of the model      
        input_names=["image"],          
        output_names=["logits"],
        opset_version=17 # pinning it so it doesn't change
    )