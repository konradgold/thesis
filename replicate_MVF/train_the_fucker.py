from tadaconv.models.base.builder import build_model
from tadaconv.utils.config import Config
from tadaconv.datasets.base.builder import build_dataset, build_loader

if __name__ == "__main__":
    cfg = Config(load=True)

    ds = build_dataset(cfg.TRAIN.DATASET, cfg, "train")

    instance = ds[0]

    dl = build_loader(cfg, "train")
    model, _ = build_model(cfg)

    print("DataLoader built.")
    for batch in dl:
        print(batch[0].shape)
        print(batch[1].shape)
        print(batch[2])
        out = model(batch[0], batch[1])
        print(out[0]['severity'].shape)
        print(out[0]['type'].shape)
        break