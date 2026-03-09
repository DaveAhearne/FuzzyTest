from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LABELS_PATH = Path(__file__).resolve().parent / "labels.json"
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "src" / "fuzzy_cnn" / "train" / "checkpoints"
ONNX_MODEL_PATH = PROJECT_ROOT / "src" / "fuzzy_cnn" / "serve" / "model_store" / "cifar10.onnx"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    cifar10_mean: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    cifar10_std: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)

    image_size: int = 32

    train_epochs: int = 20
    learning_rate: float = 0.001
    train_batch_size: int = 128
    test_batch_size: int = 128
    num_workers: int = 2

    host: str = "0.0.0.0"
    port: int = 1234
    workers: int = 1

    api_key: str = "my_very_secret_key"
    log_to_file: bool = False
    log_level: str = "INFO"

    @field_validator("image_size", "train_epochs", "train_batch_size", "test_batch_size", "num_workers", "port", "workers")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"must be a positive integer, got {v}")
        return v

settings = Settings()