import torch
from typing import Iterable
from lightning.pytorch import seed_everything
import random


def configure_environment(device=None, seed=None):
    import os

    device = device or "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        # load KerasModels on pytorch backed into cpu
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["KERAS_BACKEND"] = "torch"
    torch.set_default_device("cpu")
    seed_everything(seed or random.randint(0, 123456))
    print("Device set to", device)


def onehot(sequence: Iterable, tokens: int):
    return torch.nn.functional.one_hot(
        torch.tensor(sequence).long(), num_classes=tokens
    ).float()

