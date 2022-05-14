from flatten_dict import flatten
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from sacred import SETTINGS, Experiment

from mems.datasets import get_data_module
from mems.mem import MEM
from mems.mdae import MDAE
from mems.muvb import MUVB
from mems.utils import VizCallback

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

ex = Experiment("MEMS", save_git_info=False)


def prepare(_config: dict) -> DictConfig:
    cfg: DictConfig = OmegaConf.create(_config)
    cfg.pop("seed")
    OmegaConf.set_struct(cfg, True)
    OmegaConf.set_readonly(cfg, True)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    seed_everything(cfg.model.training.seed)
    return cfg


@ex.automain
def train(_config, _run):
    cfg = prepare(_config)
    root_dir = cfg.trainer.dir
    checkpoint_dirpath = None
    neptune_run_id = None
    pbar = None
    resume = cfg.trainer.resume

    dm = get_data_module(cfg.data)
    model_dict = {"MUVB": MUVB, "MEM": MEM, "MDAE": MDAE}
    model_cls = model_dict[cfg.model["class"]]
    model_cls.finalize_config(cfg.model)

    if "load" in cfg.trainer.keys() and cfg.trainer.load is not None:
        print("Loading model weights from", cfg.trainer.load)
        model = model_cls.load_from_checkpoint(cfg.trainer.load, map_location="cpu", cfg=cfg.model)
    else:
        print(f"Initializing {model_cls} model")
        model = model_cls(cfg=cfg.model)

    checkpoint_last_epoch = ModelCheckpoint(dirpath=checkpoint_dirpath, monitor=None, filename="last{epoch}")
    checkpoint_every_n = ModelCheckpoint(
        dirpath=checkpoint_dirpath,
        monitor="train_loss_epoch",
        filename="{epoch}",
        every_n_epochs=50,
        save_top_k=cfg.trainer.epochs,
        save_last=False,
    )

    if cfg.trainer.neptune is not None:
        params = flatten(OmegaConf.to_container(cfg, resolve=True), reducer="dot")
        from pytorch_lightning.loggers import NeptuneLogger
        logger = NeptuneLogger(
            experiment_id=neptune_run_id,
            project_name=cfg.trainer.neptune,
            experiment_name=cfg.trainer.name.upper(),
            params=params,
            upload_source_files=["run/*.py", "src/mems/*.py", "src/mems/**/*.py"],
            upload_stdout=False,
        )
    else:
        from pytorch_lightning.loggers import CSVLogger
        logger = CSVLogger(save_dir=cfg.trainer.dir, name=cfg.trainer.name)

    if cfg.trainer.auto:
        model.automatic_optimization = True
        gradient_clip_val = cfg.model.training.grad_clip
        trainer_grad_acc = cfg.trainer.grad_acc
    else:
        model.automatic_optimization = False
        gradient_clip_val = 0
        model.grad_acc = cfg.trainer.grad_acc
        trainer_grad_acc = 1

    acc = "ddp" if cfg.trainer.gpus > 1 and cfg.trainer.acc is None else cfg.trainer.acc
    benchmark = cfg.trainer.get("benchmark", True)

    trainer = Trainer(
        accumulate_grad_batches=trainer_grad_acc,
        benchmark=benchmark,
        gradient_clip_val=gradient_clip_val,
        plugins=DDPPlugin(find_unused_parameters=False) if acc == "ddp" else None,
        callbacks=[checkpoint_last_epoch, checkpoint_every_n, VizCallback()],
        weights_summary="full",
        logger=logger,
        max_epochs=cfg.trainer.epochs,
        progress_bar_refresh_rate=pbar,
        gpus=cfg.trainer.gpus,
        accelerator=acc,
        precision=cfg.trainer.precision,
        limit_train_batches=cfg.trainer.limit_train,
        limit_val_batches=cfg.trainer.limit_val,
        default_root_dir=root_dir,
        resume_from_checkpoint=resume,
        log_every_n_steps=cfg.trainer.log_freq,
    )
    if cfg.trainer.test:
        trainer.test(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)
