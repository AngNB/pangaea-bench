import hashlib
import os as os
import pathlib
import pprint
import time

import hydra
import torch
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pangaea.datasets.base import GeoFMDataset, GeoFMSubset, RawGeoFMDataset
from pangaea.decoders.base import Decoder
from pangaea.encoders.base import Encoder
from pangaea.engine.evaluator import Evaluator
from pangaea.engine.trainer import Trainer
from pangaea.utils.collate_fn import get_collate_fn
from pangaea.utils.logger import init_logger
from pangaea.utils.subset_sampler import get_subset_indices
from pangaea.utils.utils import (
    fix_seed,
    get_best_model_ckpt_path,
    get_final_model_ckpt_path,
    get_generator,
    seed_worker,
)


def get_exp_info(hydra_config: HydraConf) -> dict[str, str]:
    """Create a unique experiment name based on the choices made in the config.

    Args:
        hydra_config (HydraConf): hydra config.

    Returns:
        str: experiment information.
    """
    choices = OmegaConf.to_container(hydra_config.runtime.choices)
    cfg_hash = hashlib.sha1(
        OmegaConf.to_yaml(hydra_config).encode(), usedforsecurity=False
    ).hexdigest()[:6]
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    fm = choices["encoder"]
    decoder = choices["decoder"]
    ds = choices["dataset"]
    task = choices["task"]
    exp_info = {
        "timestamp": timestamp,
        "fm": fm,
        "decoder": decoder,
        "ds": ds,
        "task": task,
        "exp_name": f"{timestamp}_{cfg_hash}_{fm}_{decoder}_{ds}",
    }
    return exp_info


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Geofm-bench main function.

    Args:
        cfg (DictConfig): main_config
    """
    # fix all random seeds
    fix_seed(cfg.seed)
    # distributed training variables
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl")

    # true if training else false
    train_run = cfg.train
    if train_run:
        exp_info = get_exp_info(HydraConfig.get())
        exp_name = exp_info["exp_name"]
        task_name = exp_info["task"]
        exp_dir = pathlib.Path(cfg.work_dir) / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        logger_path = exp_dir / "train.log"
        config_log_dir = exp_dir / "configs"
        config_log_dir.mkdir(exist_ok=True)
        # init wandb
        if cfg.task.trainer.use_wandb and rank == 0:
            import wandb

            wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
            wandb.init(
                project="geofm-bench",
                name=exp_name,
                config=wandb_cfg,
            )
            cfg["wandb_run_id"] = wandb.run.id
        OmegaConf.save(cfg, config_log_dir / "config.yaml")

    else:
        exp_dir = pathlib.Path(cfg.ckpt_dir)
        exp_name = exp_dir.name
        logger_path = exp_dir / "test.log"
        # load training config
        cfg_path = exp_dir / "configs" / "config.yaml"
        cfg = OmegaConf.load(cfg_path)
        if cfg.task.trainer.use_wandb and rank == 0:
            import wandb

            wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
            wandb.init(
                project="geofm-bench",
                name=exp_name,
                config=wandb_cfg,
                id=cfg.get("wandb_run_id"),
                resume="allow",
            )

    logger = init_logger(logger_path, rank=rank)
    logger.info("============ (0) Initialized logger ============")
    logger.info(pprint.pformat(OmegaConf.to_container(cfg), compact=True).strip("{}"))
    logger.info("The experiment is stored in %s\n" % exp_dir)
    logger.info(f"Device used: {device}")
    print("==================================== (1) Experiment configurations set up ====================================")

    print("==================================== (2) instantiate Encoder       (OUT) ====================================")
    encoder: Encoder = instantiate(cfg.encoder)
    print("==================================== (2) instantiate Encoder       (IN) ====================================")

    print("==================================== HERE IS THE ENCODER SATMAE_BASE ====================================")
    #print(encoder)

    print("==================================== (3) Load Encoder Weights      (OUT) ====================================")
    encoder.load_encoder_weights(logger)
    print("==================================== (3) Load Encoder Weights      (IN) ====================================")

    print("==================================== (4) Built {}.".format(encoder.model_name))

    # prepare the decoder (segmentation/regression)
    print("==================================== (5) instantiate Decoder       (OUT) ====================================")

    decoder: Decoder = instantiate(
        cfg.decoder,
        encoder=encoder,
    )
    decoder.to(device)
    decoder = torch.nn.parallel.DistributedDataParallel(
        decoder,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=cfg.finetune,
    )
    print("==================================== (5) instantiate Decoder       (IN) ====================================")

    print("==================================== HERE IS THE DECODER REGUPERNET ====================================")
    #print(decoder)
    logger.info(
        "Built {} for with {} encoder.".format(
            decoder.module.model_name, type(encoder).__name__
        )
    )
    print("==================================== (6) defining modalities by calling encoder.py and calling get_collate_fn for later use in DataLoader()       (OUT) ====================================")
    modalities = list(encoder.input_bands.keys())
    collate_fn = get_collate_fn(modalities)
    print("==================================== (6) defining modalities by calling encoder.py and calling get_collate_fn for later use in DataLoader()       (IN) ====================================")


    print("==================================== (7) starting training if train_run == true ====================================")
    # training
    if train_run:
        # get preprocessor
        print("==================================== (8) instatiate train_preprocessor and val_preprocessor  (OUT) ====================================")
        train_preprocessor = instantiate(
            cfg.preprocessing.train,
            dataset_cfg=cfg.dataset,
            encoder_cfg=cfg.encoder,
            _recursive_=False,
        )
        val_preprocessor = instantiate(
            cfg.preprocessing.val,
            dataset_cfg=cfg.dataset,
            encoder_cfg=cfg.encoder,
            _recursive_=False,
        )
        print("==================================== (8) instatiate train_preprocessor and val_preprocessor  (IN) ====================================")


        # get datasets
        print("==================================== (9) get raw data by calling class RawGeoFMDataset for train and val   (OUT) ====================================")

        raw_train_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="train")
        raw_val_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="val")
        print("==================================== (9) get raw data by calling class RawGeoFMDataset for train and val   (IN) ====================================")

        print("==================================== (10) check if we just take subset ====================================")
        print("check if 0 < cfg.limited_label_train < 1:", 0 < cfg.limited_label_train < 1)

        if 0 < cfg.limited_label_train < 1:
            indices = get_subset_indices(
                raw_train_dataset,
                task=task_name,
                strategy=cfg.limited_label_strategy,
                label_fraction=cfg.limited_label_train,
                num_bins=cfg.stratification_bins,
                logger=logger,
            )
            raw_train_dataset = GeoFMSubset(raw_train_dataset, indices)
        print("check if 0 < cfg.limited_label_val < 1:", 0 < cfg.limited_label_val < 1)

        if 0 < cfg.limited_label_val < 1:
            indices = get_subset_indices(
                raw_val_dataset,
                task=task_name,
                strategy=cfg.limited_label_strategy,
                label_fraction=cfg.limited_label_val,
                num_bins=cfg.stratification_bins,
                logger=logger,
            )
            raw_val_dataset = GeoFMSubset(raw_val_dataset, indices)
        print("==================================== (10) check if we just take subset completed - proceed ====================================")
        
        print("==================================== (11) define dataset for dataloader later by calling GeoFMDataset()        (OUT) ====================================")

        train_dataset = GeoFMDataset(
            raw_train_dataset, train_preprocessor, cfg.data_replicate
        )
        print("--------------TRAIN_DATASET COMPLETED--------------")

        val_dataset = GeoFMDataset(
            raw_val_dataset, val_preprocessor, cfg.data_replicate
        )
        print("--------------VAL_DATASET COMPLETED--------------")
        print("==================================== (11) define dataset for dataloader later by calling GeoFMDataset()        (IN) ====================================")

        logger.info("Built {} dataset.".format(cfg.dataset.dataset_name))

        logger.info(
            f"Total number of train patches: {len(train_dataset)}\n"
            f"Total number of validation patches: {len(val_dataset)}\n"
        )
        print("==================================== (12) define dataloaders for train and val dataset           (OUT - EXTERNAL) ====================================")

        # get train val data loaders
        train_loader = DataLoader(
            train_dataset,
            sampler=DistributedSampler(train_dataset),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=True,
            # persistent_workers=True causes memory leak
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=get_generator(cfg.seed),
            drop_last=True,
            collate_fn=collate_fn,
        )
        print("--------------TRAIN_LOADER COMPLETED--------------")
        assert train_dataset == train_loader.dataset, "train_dataset in run.py is not the same as the dataset in train_loader"
        val_loader = DataLoader(
            val_dataset,
            sampler=DistributedSampler(val_dataset),
            batch_size=cfg.test_batch_size,
            num_workers=cfg.test_num_workers,
            pin_memory=True,
            persistent_workers=False,
            worker_init_fn=seed_worker,
            # generator=g,
            drop_last=False,
            collate_fn=collate_fn,
        )
        print("--------------VAL_LOADER COMPLETED--------------")
        print("==================================== check if dataset before and after loading in dataloader are the same ====================================")
        print("TRAIN:",train_dataset == train_loader.dataset)
        print("VAL:",val_dataset == val_loader.dataset)
        print("==================================== (12) define dataloaders for train and val dataset           (IN) ====================================")

        print("==================================== (13) instantiate criterion, optimizer and lr_scheduler         (OUT - EXTERNAL) ====================================")

        criterion = instantiate(cfg.criterion)
        optimizer = instantiate(cfg.optimizer, params=decoder.parameters())
        lr_scheduler = instantiate(
            cfg.lr_scheduler,
            optimizer=optimizer,
            total_iters=len(train_loader) * cfg.task.trainer.n_epochs,
        )
        print("==================================== (13) instantiate criterion, optimizer and lr_scheduler         (IN) ====================================")

        print("==================================== (14) instantiate the evaluator for the validation dataset         (OUT) ====================================")

        val_evaluator: Evaluator = instantiate(
            cfg.task.evaluator, val_loader=val_loader, exp_dir=exp_dir, device=device
        )
        print("==================================== (14) instantiate the evaluator for the validation dataset         (IN) ====================================")


        print("==================================== (15) instantiate the trainer         (OUT) ====================================")

        trainer: Trainer = instantiate(
            cfg.task.trainer,
            model=decoder,
            train_loader=train_loader,
            lr_scheduler=lr_scheduler,
            optimizer=optimizer,
            criterion=criterion,
            evaluator=val_evaluator,
            exp_dir=exp_dir,
            device=device,
        )
        # resume training if model_checkpoint is provided
        if cfg.ckpt_dir is not None:
            trainer.load_model(cfg.ckpt_dir)
        print("==================================== (15) instantiate the trainer         (IN) ====================================")

        print("==================================== (16) trainer.train() is being called         (OUT) ====================================")

        trainer.train()

    print("==================================== TRAINING FINISHED - EVALUATION STARTS         ====================================")

    # Evaluation
    print("==================================== (17) instantiate the test_preprocessor         (OUT) ====================================")

    test_preprocessor = instantiate(
        cfg.preprocessing.test,
        dataset_cfg=cfg.dataset,
        encoder_cfg=cfg.encoder,
        _recursive_=False,
    )
    print("==================================== (17) instantiate the test_preprocessor         (IN) ====================================")

    # get datasets
    print("==================================== (18) get raw data by calling class RawGeoFMDataset for test   (OUT) ====================================")

    raw_test_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="test")
    print("==================================== (18) get raw data by calling class RawGeoFMDataset for test   (IN) ====================================")
    print("==================================== (19) define dataset for dataloader later by calling GeoFMDataset()        (OUT) ====================================")
    test_dataset = GeoFMDataset(raw_test_dataset, test_preprocessor)
    print("--------------TEST_DATASET COMPLETED--------------")
    print("==================================== (19) define dataset for dataloader later by calling GeoFMDataset()        (IN) ====================================")

    logger.info(
            f"Total number of test patches: {len(test_dataset)}\n"
    )
    print("==================================== (20) define dataloaders for train and val dataset           (OUT - EXTERNAL) ====================================")

    test_loader = DataLoader(
        test_dataset,
        sampler=DistributedSampler(test_dataset),
        batch_size=cfg.test_batch_size,
        num_workers=cfg.test_num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    print("--------------TEST_LOADER COMPLETED--------------")
    print("==================================== check if test dataset before and after loading in dataloader are the same ====================================")
    print(test_dataset == test_loader.dataset)
    print("==================================== (20) define dataloaders for train and val dataset           (IN) ====================================")
    print("==================================== (21) instantiate the evaluator for the evaluation dataset        (OUT) ====================================")
    test_evaluator: Evaluator = instantiate(
        cfg.task.evaluator, val_loader=test_loader, exp_dir=exp_dir, device=device
    )
    print("The results of the instatiation: \n", test_evaluator)
    print("==================================== (21) instantiate the evaluator for the evaluation dataset         (IN) ====================================")

    if cfg.use_final_ckpt:
        model_ckpt_path = get_final_model_ckpt_path(exp_dir)
    else:
        model_ckpt_path = get_best_model_ckpt_path(exp_dir)
    test_evaluator.evaluate(decoder, "test_model", model_ckpt_path)

    if cfg.use_wandb and rank == 0:
        wandb.finish()
    print("==================================== COMPLETED ====================================")

if __name__ == "__main__":
    main()
