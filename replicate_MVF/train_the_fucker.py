from tadaconv.models.base.builder import build_model
from tadaconv.utils.config import Config
from tadaconv.datasets.base.builder import build_dataset, build_loader




if __name__ == "__main__":
    cfg = Config(load=True)

    dl = build_loader(cfg, "train")

    print("DataLoader built.")
    i = 0
    for batch in dl:
        print(batch[0].shape)
        print(batch[1].shape)
        print(batch[2])