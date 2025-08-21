from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    seed: int = 42
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 32
    epochs: int = 10

    data_dir: Path = Path("data")
    train_csv: Path = data_dir / "train-metadata.csv"
    images_train_dir: Path = data_dir / "train"

    models_dir: Path = Path("models")
    outputs_dir: Path = Path("outputs")

CFG = Config()
