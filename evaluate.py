import os

from src.server import Server
from config.config import Config


def evaluate():
    cfg = Config()

    cfg.set_name = "test"
    cfg.pos_dataset = cfg.data_path + "pos_" + cfg.set_name + ".pkl"
    cfg.hugging_face = False
    svr = Server(cfg=cfg)

    # inference
    accuracy, total_diff, total_num = svr.evaluate(tags=["JJ", "VB"])
    print(total_diff, total_num)
    print(accuracy)


if name__=="__main__":
    evaluate()
