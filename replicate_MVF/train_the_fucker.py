from runs.train import train
from tadaconv.utils.config import Config

cfg = Config(load=True)
train(cfg)