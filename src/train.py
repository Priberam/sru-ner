"""
This file is used to train a model. It automatically performs an evaluation step
of the trained model on the test split when training is finished.

Run it by doing 

python src/train.py experiment={experiment_path}

where {experiment_path} is the relative path of the experiment config,
rooted at configs/experiment, without .yaml.
For example, to run an experiment with config file
configs/experiment/folder/example.yaml
do

python src/train.py experiment=folder/example

One can add additional arguments to alter the config according to Hydra's syntax.
For example, restricting the number of training batches as

python src/train.py experiment=folder/example train.limit_train_batches=100

"""

import hydra
import autorootcwd
import lightning.pytorch
import lightning.pytorch.loggers
from omegaconf import DictConfig
from lightning import seed_everything
from src.utils import logger
from typing import Tuple, Optional
from omegaconf import OmegaConf

from src.modules.utils.metrics import flatten_dictionary
from src.modules.ner_module import LitNERmodel
from src.modules.data_module import NERDataModule
from src.callbacks.callback_setup import (
    get_early_stop_callback,
    get_metric_based_checkpoint_callback,
    get_recent_model_callback,
    ClearCUDACacheCallback,
)
import os
import lightning
from omegaconf import open_dict
from src.test import test
import torch
import pickle

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(os.path.join(os.getcwd(), "configs")),
    "config_name": "train",
}

log = logger.get_pylogger("NER")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")


def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best
    weights obtained during training.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated
        objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        log.info(f"Seed everything with <{cfg.seed}>")
        seed_everything(cfg.seed, workers=True)
    else:
        log.info("Skipping seeding.")

    # Init lightning datamodule
    log.info("Instantiating datamodule...")
    if (
        cfg.data_transform.merge_sents.normalize.use
        and cfg.data_transform.merge_sents.maximize.use
    ):
        raise ValueError("Can only either normalize the sentence length or maximize.")
    elif (
        cfg.data_transform.merge_sents.normalize.use
        and cfg.data_transform.merge_sents.normalize.mode not in ["global", "local"]
    ):
        raise ValueError("normalize.mode can be 'global' or 'local'.")

    if cfg.data_transform.merge_sents.normalize.use:
        merge_sents_mode = ["normalize", cfg.data_transform.merge_sents.normalize.mode]

        if (
            "force_merge" in cfg.data_transform.merge_sents
            and cfg.data_transform.merge_sents.force_merge is not None
        ):
            list_dts_force_merge = list(cfg.data_transform.merge_sents.force_merge)
            merge_sents_mode.append(list_dts_force_merge)

    elif cfg.data_transform.merge_sents.maximize.use:
        merge_sents_mode = ["maximize", cfg.data_transform.merge_sents.maximize.swap]

        if (
            "force_merge" in cfg.data_transform.merge_sents
            and cfg.data_transform.merge_sents.force_merge is not None
        ):
            list_dts_force_merge = list(cfg.data_transform.merge_sents.force_merge)
            merge_sents_mode.append(list_dts_force_merge)
    else:
        merge_sents_mode = None

    if not cfg.data_transform.change_tags:
        for dts_name in cfg.datasets:
            cfg.datasets[dts_name]["tag_dict"] = None

    if "ablation" in cfg:
        ablat = cfg.ablation
        print(f"---> Ablation: {ablat}")
    else:
        print("---> Ablation: DISABLED")
        ablat = None

    # Experimental feature:
    #
    # By specifying an entry
    #
    # perturbations:
    #   use: True
    #   probs:
    #       source_dataset_name:
    #           target_dataset_name: probability
    #
    # in the config, whenever a sample sentence from source_dataset_name is picked
    # to integrate a training batch, a sentence from target_dataset_name will be
    # appended to the former with the given probability.
    # This strategy could be useful to encorage the model to predict mentions of types
    # of other datasets.
    if "perturbations" in cfg and cfg.perturbations.use:
        print("---> PERTURBATIONS ACTIVE")
        dts = sorted(cfg.datasets.keys())
        perturb_probs = {src: {pert: 0.0 for pert in dts} for src in dts}

        in_cfg = set(d for d in dts if d in cfg.perturbations.probs)
        not_in_cfg = set(dts) - in_cfg

        if len(not_in_cfg):
            print("Not perturbing sentences from the dts: ", sorted(not_in_cfg))

        if len(in_cfg):
            print(
                "Pertubing sents from a given source dts with sents"
                + " from other dts, with a given prob:"
            )
            for src in in_cfg:
                for pert, prob in cfg.perturbations.probs.get(src).items():
                    if pert == src and prob != 0.0:
                        print(f"Skipping perturbing sentences of {src} with own sents.")
                        continue
                    perturb_probs[src][pert] = prob

            for src in sorted(perturb_probs.keys()):
                non_zero_probs = {
                    val: prob for val, prob in perturb_probs[src].items() if prob != 0.0
                }
                if len(non_zero_probs):
                    print(src, ":")
                    for pert, prob in non_zero_probs.items():
                        print(f"\t{pert}: {prob}")
            print()

    else:
        print("---> Training WITHOUT perturbations.")
        perturb_probs = None

    datamodule = NERDataModule(
        batch_size=cfg.train.batch_size,
        mode="trainer",
        input_data=cfg.datasets,
        tokenizer_model_name=cfg.models.embed_model.pretrained_name,
        change_out_to_shift=cfg.data_transform.change_out_to_shift,
        max_tokens=cfg.data_transform.max_tokens,
        train_with_separate_tags=cfg.data_transform.train_with_separate_tags,
        merge_sents_mode=merge_sents_mode,
        random_seed=cfg.seed,
        id_to_tag=None,
        perturb_probs=perturb_probs,
    )

    # Manually call the datamodule setup in order to have the actions vocab
    datamodule.setup(stage="fit")

    # Init lightning model
    log.info("Instantiating lightning model...")
    model = LitNERmodel(
        actions_vocab=datamodule.actions_vocab,
        embed_model_name=cfg.models.embed_model.pretrained_name,
        pred_model_hparams=cfg.models.pred_model,
        opt_cfg=cfg.train.optimizers,
        loss_fn=cfg.train.loss,
        train_with_separate_tags=cfg.data_transform.train_with_separate_tags,
        ablat=ablat,
    )

    # Init callbacks
    log.info("Instantiating callbacks...")

    callbacks = []
    if cfg.train.early_stop:
        # We can only early stop if the metric-based checkpoints are enabled
        if not cfg.train.checkpoints.metric_based:
            raise ValueError("Can't early stop if metric-based metrics are disabled")

        metric_to_track = "F1-average"
        if cfg.data_transform.train_with_separate_tags:
            metric_to_track = f"{model.joined_ner_metrics.name_prep}{metric_to_track}"

        log.info(
            f"Early stopping with patience {cfg.train.early_stop} "
            f"on metric {metric_to_track}"
        )
        callbacks.append(
            get_early_stop_callback(
                metric_name=metric_to_track, patience=cfg.train.early_stop
            )
        )

    checkpoint_callbacks = []
    if cfg.train.checkpoints.most_recent:
        log.info("Checkpoint enabled: most recent model")

        checkpoint_callbacks.append(
            get_recent_model_callback(dir_path=cfg.checkpoint_dir)
        )

    if cfg.train.checkpoints.metric_based:

        # If we train with separate tags, we checkpoint on joined metrics only
        if cfg.data_transform.train_with_separate_tags:
            metric_object_to_track = model.joined_ner_metrics
            metric_name_prepend = model.joined_ner_metrics.name_prep
        else:
            metric_object_to_track = model.ner_metrics
            metric_name_prepend = ""

        metric_names_for_ckpt = []
        # If we are training with one dataset only, we do the checkpoint based on the
        # 'F1-average' only. We can also force this from the config.
        if (
            len(metric_object_to_track.dataset_gold_tags) == 1
            or cfg.train.checkpoints.save_average_only
        ):
            metric_names_for_ckpt.append("F1-average")
        # Otherwise we save a checkpoint per dataset and the average
        else:
            for metric_name in metric_object_to_track.metric_names:
                if metric_name.startswith("F1") and metric_name.endswith("average"):
                    metric_names_for_ckpt.append(metric_name)

        metric_logger_names_for_ckpt = [
            f"{metric_name_prepend}{x}" for x in metric_names_for_ckpt
        ]

        log.info(
            f"Checkpoint enabled: based on the metrics {metric_logger_names_for_ckpt}"
        )

        for metric_name in metric_logger_names_for_ckpt:
            checkpoint_callbacks.append(
                get_metric_based_checkpoint_callback(
                    metric_name=metric_name, dirpath=cfg.checkpoint_dir
                )
            )

    # Add checkpoint paths to the cfg
    if checkpoint_callbacks:
        with open_dict(cfg):
            cfg.saved_ckpt_paths = OmegaConf.create({})
            for chk_call in checkpoint_callbacks:
                name = chk_call.filename
                filepath = os.path.join(
                    chk_call.dirpath, f"{name}{chk_call.FILE_EXTENSION}"
                )
                cfg.saved_ckpt_paths[name] = filepath

    callbacks.extend(checkpoint_callbacks)

    # Add clear CUDA cache callback to manage memory consumption, if needed
    if cfg.train.clear_CUDA_cache_after_n_steps is not None:
        print(
            f"Clear CUDA cache, every {cfg.train.clear_CUDA_cache_after_n_steps} steps"
        )
        callbacks.append(
            ClearCUDACacheCallback(
                every_n_steps=cfg.train.clear_CUDA_cache_after_n_steps
            )
        )

    # Save resolved config
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    from hydra.core.hydra_config import HydraConfig

    hydra_output_dir = HydraConfig.get().runtime.output_dir
    resolved_config_path = f"{hydra_output_dir}/resolved_cfg.yaml"
    OmegaConf.save(OmegaConf.create(resolved_cfg), resolved_config_path)

    # Init logger
    if cfg.MLFlowLogger.use:
        log.info("Instantiating MLFlow logger...")
        logger = lightning.pytorch.loggers.MLFlowLogger(
            experiment_name=cfg.MLFlowLogger.experiment_name,
            run_name=cfg.MLFlowLogger.run_name,
            tracking_uri=cfg.MLFlowLogger.tracking_uri,
        )
        conf = OmegaConf.to_container(cfg=cfg, resolve=True)
        # Delete tag_dict because it may cause errors with MLFlow parameter names
        for dts_name in conf["datasets"]:
            del conf["datasets"][dts_name]["tag_dict"]
        # Add final tags per dataset
        tag_info_per_dataset = {
            ds_name: "/".join([datamodule.actions_vocab.id_to_tag[ti] for ti in tid])
            for ds_name, tid in datamodule.actions_vocab.dataset_to_tag_ids.items()
        }
        conf["datasets"]["tags"] = tag_info_per_dataset
        conf = flatten_dictionary(conf)

        logger.log_hyperparams(conf)
    else:
        log.info("Not using any logger. Metrics are solely printed to terminal.")
        logger = lightning.pytorch.loggers.CSVLogger(save_dir=cfg.checkpoint_dir,
                                                     version=None)

    # Init lightning trainer
    log.info("Instantiating trainer")
    accum_steps = (
        cfg.train.accumulate_grad_batches
        if "accumulate_grad_batches" in cfg.train
        else 1
    )
    if accum_steps != 1:
        log.info(f"Using gradient accumulation: {accum_steps}")

    trainer = lightning.Trainer(
        default_root_dir=cfg.checkpoint_dir,
        max_epochs=cfg.train.num_epochs,
        devices=1,
        accelerator="gpu",
        enable_progress_bar=True,
        deterministic=cfg.train.use_deterministic,
        callbacks=callbacks,
        gradient_clip_val=cfg.train.grad_norm_clip,
        num_sanity_val_steps=cfg.train.num_sanity_val_steps,
        logger=logger,
        enable_checkpointing=bool(checkpoint_callbacks),
        accumulate_grad_batches=accum_steps,
        limit_train_batches=cfg.train.limit_train_batches,
        limit_val_batches=cfg.train.limit_val_batches,
        enable_model_summary=False,
    )

    # To resume from a checkpoint, should include the key
    # cfg.train.resume_from_checkpoint with value the path of the checkpoint
    resume = (
        cfg.train.resume_from_checkpoint
        if "resume_from_checkpoint" in cfg.train
        else None
    )

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume)

    # Report on sampling
    samples_container_path = os.path.join(
        os.path.dirname(cfg.checkpoint_dir), "sampling.pkl"
    )
    with open(samples_container_path, "wb") as f:
        pickle.dump(model.sample_ids, f)


@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> Optional[float]:

    if "datasets" not in cfg:
        raise ValueError(
            "Missing mandatory 'datasets' key in config.\n"
            "Make sure it is specified in the experiment config."
        )

    print("#" * 20, " MODEL TRAINING ", "#" * 20)
    train(cfg)

    print("#" * 20, " MODEL TESTING ", "#" * 20)
    test(cfg)


if __name__ == "__main__":
    main()
