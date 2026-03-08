from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "src" / "cifar_image_classifier" / "train" / "checkpoints"
ONNX_MODEL_PATH = PROJECT_ROOT / "src" / "cifar_image_classifier" / "serve" / "model_store" / "cifar10.onnx"

# TODO: I'm just gonna use the magic values to start, i'll recompute these properly later
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

IMAGE_SIZE = 32

TRAIN_EPOCHS = 20
LEARNING_RATE = 0.001

TRAIN_BATCH_SIZE = 128
# TODO: The test one could probably be bigger but it's fine for now
TEST_BATCH_SIZE = 128