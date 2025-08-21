from pathlib import Path
from src.config import CFG
from src.data_loader import load_metadata, build_image_ds
from src.models.cnn import build_cnn
from src.utils.seeding import set_global_seed

def main():
    set_global_seed(CFG.seed)
    df = load_metadata(CFG.train_csv)
    ds_train = build_image_ds(df, "image_path", "target", CFG.images_train_dir,
                              CFG.image_size, CFG.batch_size, shuffle=True)
    model = build_cnn(input_shape=(*CFG.image_size, 3), base_trainable=False)
    model.fit(ds_train, epochs=CFG.epochs)
    Path(CFG.models_dir).mkdir(exist_ok=True, parents=True)
    model.save(CFG.models_dir / "cnn_pretrained_trial.keras")

if __name__ == "__main__":
    main()
