import ml_collections
import copy
import os

CONFIG = ml_collections.ConfigDict(
    {
        "debug": False,
        "baseline_only": False,
        "max_epochs": 100,
        "batch_size": 48,
        "device": "cuda:0",
        "optimizer": {
            "lr": 1e-4,
            "betas": (0.95, 0.95),
            "eps": 1e-8,
        },
        "scheduler": {
            "step_size": 300,
            "gamma": 0.75,
        },
        "model_save_path": "model_weights/model_",
        "loss_save_path": "model_weights/loss_",
        "test_result_path": "model_weights/test_",
    }
)


def get_config():
    config = copy.deepcopy(CONFIG)
    return config
