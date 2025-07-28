""" Script used to perform test of a trained model on a corpus different
than the training one.

Use it like:

python scripts/cross_corpus_evaluation/predict_on_cross_corpus.py \
-- model_cfg_path {path to the resolved config generated during training} \
-- inference_corpus_path {path to a yaml containing inference corpus specificiation}

See for argument details in the main() function.

Note: make sure that the inference corpora is tagged with entity types that belong to the
set of entity types that the trained model recognizes. The names of the entity types 
of the inference corpora can be changed by specifying a 'tag_dict' argument in the
yaml.

"""

import autorootcwd
from lightning import seed_everything
from src.utils import logger
import os
import lightning
import argparse
from omegaconf import OmegaConf
import torch
import pickle
from src.modules.ner_module import LitNERmodel
from src.modules.data_module import NERDataModule
from src.modules.utils.action_utils import ActionsVocab
from src.modules.utils.metrics import NERMetrics

log = logger.get_pylogger("NER")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")
RANDOM_SEED = 42

MERGE_TAGS = True # Whether to remove the dataset prepend from the predicted tags
PRINT_TO_CONSOLE = False


class ModelForOutsidePreds(LitNERmodel):
    def on_val_test_epoch_end(self, mode):

        if MERGE_TAGS:
            # Remove dataset info from pred tags
            self.ner_metrics.data["pred_tag"] = self.ner_metrics.data["pred_tag"].apply(
                lambda x: (
                    sorted(set(tag.split("$")[-1] for tag in x))
                    if x is not None
                    else None
                )
            )

        self.ner_metrics.compute_metrics()


def main(cfg_path, inference_corpus_path, ckpt_path=None, pickle_save_file=None):
    # Parse the model config
    cfg = OmegaConf.load(cfg_path)
    # Parse the inference corpus
    inference_corpus_data = OmegaConf.load(inference_corpus_path)

    # ----
    # Initialize the model
    # ----

    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        log.info(f"Seed everything with <{cfg.seed}>")
        seed_everything(cfg.seed, workers=True)
    else:
        log.info("Skipping seeding.")

    # Get the checkpoint info
    if ckpt_path is not None:
        check_path = ckpt_path
    else:
        check_path = None
        if "saved_ckpt_paths" in cfg:
            # Look for F1-average checkpoint
            for k, v in cfg.saved_ckpt_paths.items():
                if "F1-average" in k:
                    check_path = v

            if not check_path:
                log.error("Checkpoint not found!")
            else:
                log.info(f"Loading checkpoint at {check_path}")
        else:
            raise ValueError("Checkpoint not found!")

    # Import datamodule in order to get the actions_vocab
    datamodule = NERDataModule.load_from_checkpoint(check_path)
    datamodule.actions_vocab = ActionsVocab.build_for_prediction(
        datamodule.hparams.id_to_tag, not datamodule.hparams.change_out_to_shift
    )
    overwrite_args = {
        "actions_vocab": datamodule.actions_vocab,
        "train_with_separate_tags": False,
        "use_metrics": False,
    }
    model = ModelForOutsidePreds.load_from_checkpoint(check_path, **overwrite_args)
    model.train_with_separate_tags = False
    trainer = lightning.Trainer(
        default_root_dir=cfg.checkpoint_dir,
        max_epochs=cfg.train.num_epochs,
        devices=1,
        accelerator="gpu",
        enable_progress_bar=True,
        enable_model_summary=False,
        deterministic=cfg.train.use_deterministic,
        enable_checkpointing=False,
        logger=False,
        # limit_test_batches=0.1,  # Limit the number of batches
    )

    # ----
    # Set up the inference corpus
    # ----
    inference_corpus = NERDataModule(
        batch_size=datamodule.hparams.batch_size,
        mode="trainer",
        input_data=inference_corpus_data,
        tokenizer_model_name=datamodule.hparams.tokenizer_model_name,
        change_out_to_shift=datamodule.hparams.change_out_to_shift,
        max_tokens=datamodule.hparams.max_tokens,
        train_with_separate_tags=False,
        merge_sents_mode=None,
        random_seed=datamodule.hparams.batch_size,
        id_to_tag=None,
        perturb_probs=None,
    )
    inference_corpus.setup("test")

    # ----
    # Get a metrics object for the inference corpus
    # ----
    model.ner_metrics = NERMetrics(
        tag_to_id=inference_corpus.actions_vocab.tag_to_id,
        dataset_to_tag_ids=inference_corpus.actions_vocab.dataset_to_tag_ids,
    )

    # ----
    # Eval on the inference corpus
    # ----
    trainer.test(model=model, datamodule=inference_corpus)

    # Get gold/predicted span log
    if PRINT_TO_CONSOLE:
        print("\n\nGold/pred spans:\n\n")
        for sent_id in range(len(inference_corpus.test_dataset)):
            source_dataset = inference_corpus.test_dataset.data["source_dataset"][
                sent_id
            ]
            raw_sent = inference_corpus.test_dataset.data["raw_sentences"][sent_id]
            wp = inference_corpus.test_dataset.data["wordpieces"][sent_id]
            spans_df = model.ner_metrics.data[
                model.ner_metrics.data["sent_id"] == sent_id
            ]
            spans_df = spans_df.sort_values(
                by=["start_idx", "end_idx"], ascending=[True, False]
            )
            if not len(spans_df):
                continue
            else:
                print(f"sent_id={sent_id} source_dataset={source_dataset}")
                print(raw_sent)
                for span_idx, span in spans_df.iterrows():
                    print(
                        span["start_idx"],
                        span["end_idx"],
                        wp[span["start_idx"] : span["end_idx"]],
                        span["gold_tag"],
                        span["pred_tag"],
                    )
                print()

    if pickle_save_file is not None:
        output = {
            "data": inference_corpus.test_dataset.data,
            "pred_df": model.ner_metrics.data,
        }
        with open(pickle_save_file, "wb") as f:
            pickle.dump(output, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get results on a corpus different than the train one."
    )

    # Get the resolved cfg of the model
    parser.add_argument(
        "--model_cfg_path",
        type=str,
        required=True,
        help="Path to the resolved config, generated by Hydra during training.",
    )

    # Get the info on the corpus to eval on
    parser.add_argument(
        "--inference_corpus_path",
        type=str,
        required=True,
        help="Path to a yaml with the eval corpus.",
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=False,
        help="Path to checkpoint, overwriding the one in the cfg.",
    )

    parser.add_argument(
        "--pickle_save_file",
        type=str,
        required=False,
        help="Path to the output pickle.",
    )

    args = parser.parse_args()

    main(
        cfg_path=args.model_cfg_path,
        inference_corpus_path=args.inference_corpus_path,
        ckpt_path=args.ckpt_path,
        pickle_save_file=args.pickle_save_file,
    )
