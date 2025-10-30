from tadaconv.models.base.builder import build_model
from tadaconv.utils.config import Config
from tadaconv.datasets.base.builder import build_dataset, build_loader




if __name__ == "__main__":
    cfg = Config(load=True)

    ds = build_dataset(cfg.TRAIN.DATASET, cfg, "train")

    instance = ds[0]
    print("Dataset built.")

    dl = build_loader(cfg, "train")

    model, _ = build_model(cfg)
    print("DataLoader and Model built.")
    i = 0
    for batch in dl:
        print(batch[0].shape)
        print(batch[1].shape)
        print(batch[2])
        outputs = model(batch[0], mask=batch[1])
        i += 1
        if i >= 3:
            break