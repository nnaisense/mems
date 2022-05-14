import sys
from pathlib import Path

import requests
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers.csv_logs import ExperimentWriter
from pytorch_lightning.utilities import rank_zero_only
from torch.optim import SGD, Adam, AdamW

from mems.sampler import gen_samples


def get_optimizer(params, optimizer_config: DictConfig):
    if optimizer_config["class"].lower() == "adam":
        opt = Adam
    elif optimizer_config["class"].lower() == "adamw":
        opt = AdamW
    elif optimizer_config["class"].lower() == "sgd":
        opt = SGD
    else:
        raise NotImplementedError(optimizer_config["class"])

    return opt(params, **optimizer_config["config"])


def download_file(url, filename):
    with open(filename, "wb") as f:
        response = requests.get(url, stream=True)
        total = response.headers.get("content-length")

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write("\r[{}{}]".format("█" * done, "." * (50 - done)))
                sys.stdout.flush()
    sys.stdout.write("\n")


class VizCallback(Callback):
    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        writer = pl_module.logger.experiment

        if hasattr(pl_module, "score"):  # If it is a MUVB or MEM
            samples, image_grid = gen_samples(pl_module, pl_module.cfg.training.sampling)

            if samples is not None:
                if isinstance(writer, ExperimentWriter):
                    save_path = Path(writer.log_dir) / f"samples_epoch{trainer.current_epoch}.png"
                    image_grid.save(save_path, format="PNG")
                else:
                    try:
                        # Assume NeptuneLogger in use
                        writer.log_image(
                            "samples", y=image_grid, x=trainer.current_epoch, image_name=f"epoch{trainer.current_epoch}"
                        )
                    except:
                        print("Samples generated but not saved! Please use NeptuneLogger or CSVLogger.")
