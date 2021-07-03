"""
Utilities

2020-11-17 first created
"""

import os
from time import time, strftime, gmtime

import tensorflow as tf

tfk = tf.keras
tfkc = tfk.callbacks

__all__ = ["NBatchLogger", "UpdateLossFactor", "tfk", "tfkc"]


class NBatchLogger(tfkc.Callback):
    """A Logger that logs the average performance per `display` steps.

    See: https://gist.github.com/jaekookang/7e2ca4dc2b1ab10dbb80b9e65ca91179
    """

    def __init__(
        self,
        n_display: int,
        max_epoch: int,
        save_dir: str = None,
        suffix: str = None,
        silent=False,
    ):
        super().__init__()
        self.epoch = 0
        self.display = n_display
        self.max_epoch = max_epoch
        self.logs = {}
        self.save_dir = save_dir
        self.silent = silent
        if self.save_dir is not None:
            assert os.path.exists(self.save_dir), Exception(
                f"Path:{self.save_dir} does not exist!"
            )
            fname = "train.log"
            if suffix is not None:
                fname = f"train_{suffix}.log"
            self.fid = open(os.path.join(save_dir, fname), "w")
        self.t0 = time()

    def on_train_begin(self, logs: dict = None):
        if logs is None:
            logs = {}
        logs = logs or self.logs
        txt = f"=== Started at {self.get_time()} ==="
        self.write_log(txt)
        if not self.silent:
            print(txt)

    def on_epoch_end(self, epoch: int, logs: dict = None):
        if logs is None:
            logs = {}
        self.epoch += 1
        fstr = " {} | Epoch: {:0{}d}/{:0{}d} | "
        precision = len(str(self.max_epoch))
        if (self.epoch % self.display == 0) | (self.epoch == 1):
            txt = fstr.format(
                self.get_time(), self.epoch, precision, self.max_epoch, precision
            )
            # txt = f' {self.get_time()} | Epoch: {self.epoch}/{self.max_epoch} | '
            if not self.silent:
                print(txt, end="")
            for i, key in enumerate(logs.keys()):
                if (i + 1) == len(logs.keys()):
                    _txt = f"{key}={logs[key]:4f}"
                    if not self.silent:
                        print(_txt, end="\n")
                else:
                    _txt = f"{key}={logs[key]:4f} "
                    if not self.silent:
                        print(_txt, end="")
                txt = txt + _txt
            self.write_log(txt)
        self.logs = logs

    def on_train_end(self, logs: dict = None):
        if logs is None:
            logs = {}
        logs = logs or self.logs
        t1 = time()
        txt = f"=== Time elapsed: {(t1 - self.t0) / 60:.4f} min ==="
        if not self.silent:
            print(txt)
        self.write_log(txt)

    def get_time(self):
        return strftime("%Y-%m-%d %Hh:%Mm:%Ss", gmtime())

    def write_log(self, txt: str):
        if self.save_dir is not None:
            self.fid.write(txt + "\n")
            self.fid.flush()


class UpdateLossFactor(tfkc.Callback):
    def __init__(self, n_epochs: int):
        super(UpdateLossFactor, self).__init__()
        self.n_epochs = n_epochs

    def on_epoch_end(self, epoch: int, logs: dict = None):
        if logs is None:
            logs = {}
        self.model.loss_factor = min(
            1.0, 2.0 * 0.002 ** (1.0 - (float(epoch) / self.n_epochs))
        )
