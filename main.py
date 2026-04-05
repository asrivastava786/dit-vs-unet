from training.train import train
import yaml

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

train(config)
