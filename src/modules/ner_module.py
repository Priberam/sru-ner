import lightning as L
import torch
import torch.nn as nn
from functools import partial
from transformers.optimization import _get_linear_schedule_with_warmup_lr_lambda as LSW
from transformers.optimization import LambdaLR
from torch.optim.optimizer import Optimizer
from torch_scatter import scatter_max
from src.modules.models.embedding_model import EmbeddingModel
from src.modules.utils.action_utils import ActionsVocab, LabeledSubstring
from torch.optim import AdamW
from src.modules.utils.metrics import NERMetrics, flatten_dictionary
from omegaconf import DictConfig
from src.modules.models.prediction_model import PredictionModel
import pandas as pd
from collections import Counter
from pytorch_lightning.utilities.model_summary import summarize


class LitNERmodel(L.LightningModule):
    # Flag to make sure that, when loading from a checkpoint, we do not need to
    # load the original weights of the encoder and then replacing them with the weights
    # from the checkpoint
    is_loading_from_checkpoint = False

    def __init__(
        self,
        actions_vocab: ActionsVocab,
        embed_model_name: str,
        pred_model_hparams: DictConfig,
        opt_cfg: DictConfig,
        train_with_separate_tags: bool,
        loss_fn: str = "MultiLabel",
        use_metrics: bool = True,
        ablat: DictConfig = None,
    ):
        super().__init__()

        # In the case of ablation, alter dropouts
        if ablat is not None and not ablat.pred_model.dropouts:
            pred_model_hparams["attention_module"]["dropout_classes"] = 0.0
            pred_model_hparams["attention_module"]["dropout_pos_embeds"] = 0.0
            pred_model_hparams["attention_module"]["dropout_scores"] = 0.0
            pred_model_hparams["parser_state"]["dropout"] = 0.0

        # Save arguments to self.hparams
        self.save_hyperparameters(logger=False, ignore="actions_vocab")

        # Save optimizer config
        self.opt_cfg = opt_cfg

        # Save actions_vocab
        self.actions_vocab = actions_vocab

        if use_metrics:
            # Initialize metrics
            self.ner_metrics = NERMetrics(
                tag_to_id=self.actions_vocab.tag_to_id,
                dataset_to_tag_ids=self.actions_vocab.dataset_to_tag_ids,
            )

            self.train_with_separate_tags = train_with_separate_tags
            # If we train with separate tags, we also do eval with joined tags
            if self.train_with_separate_tags:
                merged_tag_to_id = {}
                self.tag_id_to_merged_tag_id = {}
                for id in sorted(self.actions_vocab.id_to_tag.keys()):
                    inner_tag = self.actions_vocab.id_to_tag[id].split("$")[-1]
                    if inner_tag not in merged_tag_to_id:
                        merged_tag_to_id[inner_tag] = len(merged_tag_to_id)

                    self.tag_id_to_merged_tag_id[id] = merged_tag_to_id[inner_tag]

                merged_dataset_to_tag_ids = {}
                for dts_name in self.actions_vocab.dataset_to_tag_ids:
                    merged_dataset_to_tag_ids[dts_name] = sorted(
                        set(
                            self.tag_id_to_merged_tag_id[id]
                            for id in self.actions_vocab.dataset_to_tag_ids[dts_name]
                        )
                    )

                self.joined_ner_metrics = NERMetrics(
                    tag_to_id=merged_tag_to_id,
                    dataset_to_tag_ids=merged_dataset_to_tag_ids,
                )

                # Separator to distinguish metric in logger
                self.joined_ner_metrics.name_prep = "joined-"

                print()
                print("-> Created joined metrics with tags:")
                print(merged_tag_to_id)
                print("belonging to the datasets:")
                print(merged_dataset_to_tag_ids)
                print()

        # Initialize the model
        if hasattr(self.actions_vocab, 'allowed_for_pred'):
            multitask_mode = (len(self.actions_vocab.allowed_for_pred) > 1)
            print(f'Setting multitask_mode to {multitask_mode}')
        else:
            multitask_mode = True
        self.__init_model__(
            embed_model_name=embed_model_name,
            pred_model_hparams=pred_model_hparams,
            multitask_mode=multitask_mode,
            ablat=ablat,
        )

        # Define the loss function
        if loss_fn not in ["BCE", "MultiLabel"]:
            raise ValueError("Incorrect loss function")
        else:
            if loss_fn == "BCE":
                self.loss = nn.BCEWithLogitsLoss(reduction="mean")
            elif loss_fn == "MultiLabel":
                self.loss = nn.MultiLabelSoftMarginLoss(reduction="mean")

        LitNERmodel.is_loading_from_checkpoint = False

    @classmethod
    def load_from_checkpoint(cls, *args, **kwargs):
        cls.is_loading_from_checkpoint = True
        return super().load_from_checkpoint(*args, **kwargs)

    def __init_model__(
        self, embed_model_name: str, pred_model_hparams: DictConfig, multitask_mode: bool, ablat: DictConfig
    ):

        # Embedding model
        self.embedding_model = EmbeddingModel(
            model_name=embed_model_name,
            loading_from_ckpt=LitNERmodel.is_loading_from_checkpoint,
        )

        # Prediction model
        self.pred_model = PredictionModel(
            hparams=pred_model_hparams,
            embed_dim=self.embedding_model.embed_dim,
            actions_vocab=self.actions_vocab,
            multitask_mode=multitask_mode,
            ablat=ablat,
        )

    def compute_loss(self, gold_tensor: torch.Tensor, pred_tensor: torch.Tensor):
        """Given `gold_tensor` (batch_size, gold_seq_len, num_actions) with
        probabilities and `pred_tensor` (batch_size, pred_seq_len, num_actions) with
        logits, computes BCEWithLogitsLoss. We select only the timesteps in
        `gold_tensor` that have some action with non-zero probability."""

        all_gold = gold_tensor.view(-1, gold_tensor.size(-1))
        all_pred = pred_tensor.view(-1, pred_tensor.size(-1))
        timesteps_with_gold = all_gold.any(dim=-1)
        relevant_gold = all_gold[timesteps_with_gold]
        relevant_pred = all_pred[timesteps_with_gold]

        loss = self.loss(input=relevant_pred, target=relevant_gold)

        return loss

    def forward(self, mode: str, batch: dict):

        # Get embeddings of all tokens
        embeds = self.embedding_model.encode(
            input_ids=batch["token_ids"],
            attention_mask=batch["tokens_ids_attention_mask"],
        )

        batch_size, seq_token_len, embed_dim = embeds.size()

        # Use max-pooling across subwords
        index = torch.zeros(
            size=(batch_size, seq_token_len), dtype=torch.long, device=embeds.device
        )

        # Recall the meaning of the numbers in batch['token_subword_type'][i]:
        # 3: the token corresponds to a single word
        # 1: the token corresponds to the first sub-word
        # 2: the token corresponds to the last sub-word
        # 0: the token corresponds to one of the middle sub-words
        # -1: the token is a special token

        # We give SEP its own embedding and maximize all padding tokens
        # into the last padding token
        pos_last_pad = max([len(x.values()) for x in batch["subwords_map"]]) + 1
        for sent_numb, sent_mask in enumerate(batch["token_subword_type"]):
            id = -1
            for pos, msk in enumerate(sent_mask):
                if msk not in [0, 2]:
                    id += 1
                index[sent_numb, pos] = id

            for pad_pos in range(pos + 1, seq_token_len):
                index[sent_numb, pad_pos] = pos_last_pad

        embeds = scatter_max(src=embeds, index=index, dim=1)[0]
        not_padding = scatter_max(
            src=batch["tokens_ids_attention_mask"].float(), index=index, dim=1
        )[0].bool()
        updated_lengths = not_padding.count_nonzero(dim=1).to(torch.int)

        # Note that now we have tensors with the number of words
        # in the longest sentence + 2 (due to CLS and SEP).
        # We call this number `seq_len`
        batch_size, seq_len, embed_dim = embeds.size()

        if mode == "train":
            model_output = self.pred_model.fw_train(
                embeds=embeds,
                att_mask=not_padding,
                in_gold_actions=batch["actions_tensor"],
                allowed_for_pred=batch["allowed_for_pred"],
            )

        elif mode == "eval":
            model_output = self.pred_model.fw_decode(
                embeds=embeds,
                att_mask=not_padding,
            )

        return model_output

    def on_train_start(self):

        if self.hparams.ablat is None or (
            self.hparams.ablat is not None and self.hparams.ablat.embed_model.finetune
        ):
            # Ensure embedding model is in train mode
            self.embedding_model.model.train()
        else:
            self.embedding_model.model.eval()

        # Record the samples picked for each batch
        self.sample_ids = {}

        # Print model summary
        print("\n Model Summary at the start of training:")
        print(summarize(self, max_depth=3))
        print()

        # Record the datasets of pertubations for each batch
        self.perturb_dts = {}

    def on_train_batch_start(self, batch, batch_idx):
        # Record the samples of this batch
        self.sample_ids.setdefault(self.current_epoch, [])
        this_batch = {
            source_dts: []
            for source_dts in self.trainer.datamodule.train_dataset.num_entities.keys()
        }

        for sample_id, src_dts in zip(
            batch["dataset_indices"], batch["source_dataset"]
        ):
            this_batch[src_dts].append(sample_id)

        self.sample_ids[self.current_epoch].append(this_batch)

        if "pert_dts" in batch:
            # Record the dataset of pertubations
            self.perturb_dts.setdefault(self.current_epoch, [])
            this_batch = {
                source_dts: Counter()
                for source_dts in self.trainer.datamodule.train_dataset.num_entities.keys()
            }
            for souce_dts, pert_dts in zip(batch["source_dataset"], batch["pert_dts"]):
                this_batch[souce_dts] += Counter([pert_dts])

            self.perturb_dts[self.current_epoch].append(this_batch)

    def on_train_epoch_end(self):
        # Save the number of samples per dataset in this epoch
        nr_samples_per_dts = {
            source_dts: 0
            for source_dts in self.trainer.datamodule.train_dataset.num_entities.keys()
        }
        nr_repeated_samples_per_dts = {
            source_dts: 0
            for source_dts in self.trainer.datamodule.train_dataset.num_entities.keys()
        }
        for src_dts in nr_samples_per_dts:
            sample_id_this_dts = [
                x
                for batch in self.sample_ids[self.current_epoch]
                for x in batch[src_dts]
            ]
            nr_samples_per_dts[src_dts] = len(sample_id_this_dts)
            nr_repeated_samples_per_dts[src_dts] = sum(
                True
                for sample_id, count in Counter(sample_id_this_dts).items()
                if count > 1
            )

        nr_samples_per_dts_dict = {
            f"epoch_samples_in_{dts}": val for dts, val in nr_samples_per_dts.items()
        }
        nr_rep_samples_per_dts_dict = {
            f"epoch_repeat_samples_in_{dts}": val
            for dts, val in nr_repeated_samples_per_dts.items()
        }
        self.log_dict(nr_samples_per_dts_dict, logger=True, on_epoch=True)
        self.log_dict(nr_rep_samples_per_dts_dict, logger=True, on_epoch=True)

        if len(self.perturb_dts):
            # For pertubations, save the number of pertubations per dataset in
            # this epoch
            nr_pert_per_dts = {
                source_dts: Counter()
                for source_dts in self.trainer.datamodule.train_dataset.num_entities.keys()
            }
            for src_dts in nr_pert_per_dts:
                for batch in self.perturb_dts[self.current_epoch]:
                    nr_pert_per_dts[src_dts] += batch[src_dts]

            nr_pert_per_dts_dict = {
                f"pert_in_{src}_from_{p}": val
                for src in nr_pert_per_dts
                for p, val in nr_pert_per_dts[src].items()
            }
            self.log_dict(nr_pert_per_dts_dict, logger=True, on_epoch=True)

    def training_step(self, batch, batch_idx):
        assert self.pred_model.training, "Pred model is not in train mode!"
        if self.hparams.ablat is None or (
            self.hparams.ablat is not None and self.hparams.ablat.embed_model.finetune
        ):
            assert (
                self.embedding_model.model.training
            ), "Embedding model is not in train mode!"

        results = self.forward(mode="train", batch=batch)

        # Get logits
        logits_all_timesteps = results["timesteps_logits"]
        gold_actions = results["new_gold_actions"]

        # Compute the loss
        loss = self.compute_loss(
            gold_tensor=gold_actions, pred_tensor=logits_all_timesteps
        )

        # Log LRs
        self.log(
            name="LR/embed",
            value=self.optimizers().param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
        )
        self.log(
            name="LR/pred",
            value=self.optimizers().param_groups[2]["lr"],
            on_step=True,
            on_epoch=False,
        )

        # Log loss
        self.log(
            name="train/loss",
            value=loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            reduce_fx="mean",
            batch_size=logits_all_timesteps.size(0),
        )

        return loss

    def val_or_test_step(self, batch, batch_idx):
        assert not self.pred_model.training, "Pred model is not in eval mode!"
        assert (
            not self.embedding_model.model.training
        ), "Embedding model is not in eval mode!"

        # Get results of model
        probs_all_timesteps = self.forward(mode="eval", batch=batch)["timesteps_probs"]

        # Decode the predictions
        pred_spans = self.actions_vocab.decode_batch_action_tensor(
            probs_tensor=probs_all_timesteps, word_list=batch["wordpieces"],
            resolve_clash=False
        )

        # Update the metric dataframe
        results_dict = {}
        for batch_idx, sent_id in enumerate(batch["dataset_indices"]):
            results_dict[sent_id] = {
                "gold_entities_matching_tokens": batch["gold_entities_matching_tokens"][
                    batch_idx
                ],
                "gold_entities_not_matching_tokens": batch[
                    "gold_entities_not_matching_tokens"
                ][batch_idx],
                "predicted_entities": pred_spans[batch_idx],
                "source_dataset": batch["source_dataset"][batch_idx],
            }

        self.ner_metrics.append_batch(data_dict=results_dict)

        # If training with separate tags, add to the joined metric dataframe
        if self.train_with_separate_tags:
            joined_results_dict = {sent_id: {} for sent_id in results_dict}
            for sent_id in joined_results_dict:
                joined_results_dict[sent_id]["source_dataset"] = results_dict[sent_id][
                    "source_dataset"
                ]
                for key in [
                    "gold_entities_matching_tokens",
                    "gold_entities_not_matching_tokens",
                    "predicted_entities",
                ]:
                    joined_results_dict[sent_id][key] = set()
                    for span in results_dict[sent_id][key]:
                        joined_results_dict[sent_id][key].add(
                            LabeledSubstring(
                                start=span.start,
                                end=span.end,
                                tag=span.tag.split("$")[-1],
                                text=span.text,
                            )
                        )
                    joined_results_dict[sent_id][key] = sorted(
                        joined_results_dict[sent_id][key],
                        key=lambda x: (x.start, x.end, x.tag),
                    )

            self.joined_ner_metrics.append_batch(data_dict=joined_results_dict)

        # if not hasattr(self, "val_mem_counter"):
        #     self.val_mem_counter = 0
        # else:
        #     self.val_mem_counter += 1

        # if self.val_mem_counter == 4:
        #     try:
        #         torch.cuda.memory._dump_snapshot("val_memory_usage.pickle")
        #     except Exception as e:
        #         raise ValueError(f"Failed to capture memory snapshot {e}")

        #     print()

    def on_val_test_epoch_start(self):
        """Resets the metric dataframes"""
        self.ner_metrics.reset()

        if hasattr(self, "joined_ner_metrics"):
            self.joined_ner_metrics.reset()

    def on_val_test_epoch_end(self, mode):
        """If at val, we output to logger and the console,
        if at test, we just output to console."""

        # Log the number of predictions that are made from other datasets
        if mode == "val" and len(self.ner_metrics.dataset_gold_tags) > 1:
            preds_other_dts = {
                source_dts: {
                    other_dts: 0
                    for other_dts in set(self.ner_metrics.dataset_gold_tags.keys())
                    - {source_dts}
                }
                for source_dts in self.ner_metrics.dataset_gold_tags.keys()
            }
            for source_dts, group in self.ner_metrics.data.groupby("source_dataset"):
                other_preds = (
                    group["pred_tag"]
                    .apply(
                        lambda row: (
                            [
                                tag
                                for tag in row
                                if tag
                                not in self.ner_metrics.dataset_gold_tags[source_dts]
                            ]
                            if row is not None
                            else []
                        )
                    )
                    .to_list()
                )
                dts_of_other_preds = [
                    xs.split("$")[0] for x in other_preds for xs in x if "$" in xs
                ]
                dts_of_other_preds_count = dict(Counter(dts_of_other_preds))
                for other_dts in set(self.ner_metrics.dataset_gold_tags.keys()) - {
                    source_dts
                }:
                    if other_dts in dts_of_other_preds_count:
                        preds_other_dts[source_dts][other_dts] = (
                            dts_of_other_preds_count[other_dts]
                        )

            flatten_data = {}
            for source_dts in preds_other_dts:
                for other_dts, val in preds_other_dts[source_dts].items():
                    flatten_data[f"pred_from_{other_dts}_on_{source_dts}"] = val

            self.log_dict(flatten_data, logger=True, on_epoch=True)

            # Print to console
            print("\n")
            print("-" * 10)
            print("Predicted tags of other datasets\n")
            for source_dts in preds_other_dts:
                print(f"On source corpus {source_dts}:")
                for other_dts, val in preds_other_dts[source_dts].items():
                    print(f"\t{other_dts}: {val}")
            print("-" * 10)
            print("\n")

        if self.train_with_separate_tags:
            metric_objects = [self.ner_metrics, self.joined_ner_metrics]
            metric_names = ["separate", "joined"]
            metric_prepend = ["", self.joined_ner_metrics.name_prep]

            for m_o, m_n, m_p in zip(metric_objects, metric_names, metric_prepend):
                print()
                print(f"Results with {m_n} tags")
                print()
                m_o.compute_metrics()

                # Log the metrics if in 'val'
                if mode == "val":
                    flatten_metrics = flatten_dictionary(m_o.metric_dict)
                    flatten_metrics = {
                        f"{m_p}{n}": v for (n, v) in flatten_metrics.items()
                    }
                    # Since MLFlow only allows '_', '/', '.' and ' ' special characters
                    # in metric names, replace '$' by '_' if exists
                    flatten_metrics_cor = {}
                    for key, value in flatten_metrics.items():
                        if "$" in key:
                            new_key = key.replace("$", "_")
                        else:
                            new_key = key
                        flatten_metrics_cor[new_key] = value

                    self.log_dict(flatten_metrics_cor, logger=True, on_epoch=True)

                    # Get average based on tags
                    dts_av = {
                        dts_name: m_o.metric_dict["F1"][dts_name]["average"]
                        for dts_name in m_o.dataset_gold_tags
                    }
                    num_tags = {
                        dts_name: sum(n_ents.values())
                        for dts_name, n_ents in self.trainer.datamodule.val_dataset.num_entities.items()
                    }
                    weights = {
                        dts_name: num_tags[dts_name] / sum(num_tags.values())
                        for dts_name in num_tags
                    }
                    tag_av_f1 = sum(
                        [dts_av[dts_name] * weights[dts_name] for dts_name in dts_av]
                    )

                    self.log(f"{m_p}F1-average-tags", tag_av_f1)
        else:
            # Compute precision, recall and F1 at different levels of granularity
            self.ner_metrics.compute_metrics()

            # Log the metrics if in 'val'
            if mode == "val":
                flatten_metrics = flatten_dictionary(self.ner_metrics.metric_dict)
                flatten_metrics = {f"{n}": v for (n, v) in flatten_metrics.items()}
                self.log_dict(flatten_metrics, logger=True, on_epoch=True)

                # Get average based on tags
                dts_av = {
                    dts_name: self.ner_metrics.metric_dict["F1"][dts_name]["average"]
                    for dts_name in self.ner_metrics.dataset_gold_tags
                }
                num_tags = {
                    dts_name: sum(n_ents.values())
                    for dts_name, n_ents in self.trainer.datamodule.val_dataset.num_entities.items()
                }
                weights = {
                    dts_name: num_tags[dts_name] / sum(num_tags.values())
                    for dts_name in num_tags
                }
                tag_av_f1 = sum(
                    [dts_av[dts_name] * weights[dts_name] for dts_name in dts_av]
                )

                self.log("F1-average-tags", tag_av_f1)

    def validation_step(self, batch, batch_idx):

        return self.val_or_test_step(
            batch=batch,
            batch_idx=batch_idx,
        )

    def on_validation_epoch_start(self):
        self.on_val_test_epoch_start()

    def on_validation_epoch_end(self):
        self.on_val_test_epoch_end(mode="val")

    def test_step(self, batch, batch_idx):
        return self.val_or_test_step(
            batch=batch,
            batch_idx=batch_idx,
        )

    def on_test_epoch_start(self):
        self.on_val_test_epoch_start()

    def on_test_epoch_end(self):
        self.on_val_test_epoch_end(mode="test")

    def predict_step(self, batch, batch_idx):
        assert not self.pred_model.training, "Pred model is not in eval mode!"
        assert (
            not self.embedding_model.model.training
        ), "Embedding model is not in eval mode!"

        # Get results of model
        probs_all_timesteps = self.forward(mode="eval", batch=batch)["timesteps_probs"]

        # Decode the predictions
        pred_spans = self.actions_vocab.decode_batch_action_tensor_predict(
            probs_tensor=probs_all_timesteps,
            raw_sents=batch["raw_sentences"],
            offsets=batch["offset_mapping_sentence"],
        )

        # Update the dataframe
        if self.gold_on_predict:
            results_dict = {}
            for batch_idx, sent_id in enumerate(batch["dataset_indices"]):
                results_dict[sent_id] = {
                    "gold_entities_matching_tokens": batch[
                        "gold_entities_matching_tokens"
                    ][batch_idx],
                    "gold_entities_not_matching_tokens": batch[
                        "gold_entities_not_matching_tokens"
                    ][batch_idx],
                    "predicted_entities": pred_spans[batch_idx],
                }

            for sent_idx, spans_in_sent in results_dict.items():
                # Add a row for each span in `gold_entities_not_matching_tokens`
                for span in spans_in_sent["gold_entities_not_matching_tokens"]:
                    row = {
                        "sent_id": sent_idx,
                        "start_idx": -1,
                        "end_idx": -1,
                        "pred_tag": None,
                        "gold_tag": span.tag,
                        "matches_tokenization": False,
                    }
                    self.predict_df = pd.concat(
                        [self.predict_df, pd.DataFrame([row])], ignore_index=True
                    )

                # Get the (start, end) for the entities that fit the tokens,
                # and add a row for each
                start_end_spans = set(
                    LabeledSubstring(start=span.start, end=span.end)
                    for span in (
                        spans_in_sent["gold_entities_matching_tokens"]
                        + spans_in_sent["predicted_entities"]
                    )
                )

                # For each (start, end) compile gold/pred tags
                for s_e_span in start_end_spans:

                    row = {
                        "sent_id": sent_idx,
                        "start_idx": s_e_span.start,
                        "end_idx": s_e_span.end,
                    }

                    # Gather all predicted tags for this span
                    pred_tags = [
                        span.tag
                        for span in spans_in_sent["predicted_entities"]
                        if span.same_spans(s_e_span)
                    ]
                    if pred_tags:
                        row["pred_tag"] = sorted(pred_tags)
                    else:
                        row["pred_tag"] = None

                    # Check for gold tags
                    # A span can have multiple gold tags in certain datasets, so we add
                    # format the gold_tag as a list

                    # We deal with the gold_matching_tokens separately to the
                    # gold_not_matching_tokens
                    gold_matching_tokens = s_e_span.get_from_list_with_the_same_span(
                        spans_in_sent["gold_entities_matching_tokens"]
                    )

                    if gold_matching_tokens:
                        row["gold_tag"] = [span.tag for span in gold_matching_tokens]
                        row["matches_tokenization"] = True
                    else:
                        row["gold_tag"] = None
                        # Case where the span only has predicted tags, so obviously
                        # matches tokenization
                        row["matches_tokenization"] = True

                    self.predict_df = pd.concat(
                        [self.predict_df, pd.DataFrame([row])], ignore_index=True
                    )
        else:
            # Save on a dictionary
            for batch_idx, sent_id in enumerate(batch["dataset_indices"]):
                self.predict_df[sent_id] = {
                    "raw_sentences": batch["raw_sentences"][batch_idx],
                    "wordpieces": batch["wordpieces"][batch_idx],
                    "start_end_char_idx_of_words": batch["start_end_char_idx_of_words"][
                        batch_idx
                    ],
                    "char_idx_in_post": batch["char_idx_in_post"][batch_idx],
                    "spans": pred_spans[batch_idx],
                }

    def on_predict_epoch_start(self):
        """If we have gold annotations in the predict corpus, results are
        saved on a dataframe. Otherwise, they are saved in a simple dict."""
        self.gold_on_predict = all(
            [
                "gold_entities_matching_tokens"
                in self.trainer.datamodule.data["predict"][name]
                for name in self.trainer.datamodule.data["predict"]
            ]
        )

        if self.gold_on_predict:
            self.predict_df = pd.DataFrame(
                {
                    "sent_id": pd.Series(dtype="int"),
                    "start_idx": pd.Series(dtype="int"),
                    "end_idx": pd.Series(dtype="int"),
                    "gold_tag": pd.Series(dtype="object"),
                    "pred_tag": pd.Series(dtype="object"),
                    "matches_tokenization": pd.Series(dtype="bool"),
                }
            )
        else:
            self.predict_df = {}

    def configure_optimizers(self):
        # Optmizers for each module
        embed_params_with_weight_decay = get_names_params_should_have_decay(
            self.embedding_model
        )
        pred_params_with_weight_decay = get_names_params_should_have_decay(
            self.pred_model
        )
        optimizer = AdamW(
            [
                dict(
                    params=[
                        p
                        for n, p in self.embedding_model.named_parameters()
                        if (p.requires_grad and n in embed_params_with_weight_decay)
                    ],
                    lr=self.opt_cfg.embed_model.lr,
                    weight_decay=self.opt_cfg.embed_model.weight_decay,
                    betas=(0.9, 0.98),
                    eps=1e-6,
                ),
                dict(
                    params=[
                        p
                        for n, p in self.embedding_model.named_parameters()
                        if p.requires_grad and n not in embed_params_with_weight_decay
                    ],
                    lr=self.opt_cfg.embed_model.lr,
                    weight_decay=(
                        self.opt_cfg.embed_model.weight_decay
                        if self.opt_cfg.use_weight_decay_on_bias
                        else 0.0
                    ),
                    betas=(0.9, 0.98),
                    eps=1e-6,
                ),
                dict(
                    params=[
                        p
                        for n, p in self.pred_model.named_parameters()
                        if p.requires_grad and n in pred_params_with_weight_decay
                    ],
                    lr=self.opt_cfg.pred_model.lr,
                    weight_decay=self.opt_cfg.pred_model.weight_decay,
                    betas=(0.9, 0.98),
                    eps=1e-6,
                ),
                dict(
                    params=[
                        p
                        for n, p in self.pred_model.named_parameters()
                        if p.requires_grad and n not in pred_params_with_weight_decay
                    ],
                    lr=self.opt_cfg.pred_model.lr,
                    weight_decay=(
                        self.opt_cfg.pred_model.weight_decay
                        if self.opt_cfg.use_weight_decay_on_bias
                        else 0.0
                    ),
                    betas=(0.9, 0.98),
                    eps=1e-6,
                ),
            ]
        )

        # We define a number of warmup steps for each parameter group of the optimizer
        nr_total_training_steps = self.trainer.estimated_stepping_batches
        t_steps_per_epoch = self.trainer.num_training_batches
        warmups = [
            t_steps_per_epoch * self.opt_cfg.embed_model.warm_up_in_epochs
        ] * 2 + [t_steps_per_epoch * self.opt_cfg.pred_model.warm_up_in_epochs] * 2

        scheduler = {
            "scheduler": custom_get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmups,
                num_training_steps=nr_total_training_steps,
            ),
            "interval": "step",  # update every batch
        }

        return [optimizer], [scheduler]


def get_names_params_should_have_decay(model: nn.Module) -> list[str]:
    """Given a PyTorch Module `model`, gives the name of the parameters that should have
    weight decay applied to them. These are all the parameters except the bias ones of
    nn.LayerNorm's.
    See discussion
    https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994
    """
    params_should_have_decay = get_parameter_names(
        model=model, forbidden_layer_types=[nn.LayerNorm]
    )
    params_should_have_decay = [n for n in params_should_have_decay if "bias" not in n]

    return params_should_have_decay


def get_parameter_names(model: nn.Module, forbidden_layer_types: list[nn.Module]):
    """
    Returns the names of the `model` parameters that are not inside a layer which has
    type one of the in `forbidden_layer_types`.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter)
    # since they are not in any child.
    result += list(model._parameters.keys())
    return result


def custom_get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: list[int],
    num_training_steps: int,
    last_epoch: int = -1,
):
    """
    Uses the same implementation as 'get_linear_schedule_with_warmup' in transformers,
    but allows to pass a list `num_warmup_steps`.
    """

    lr_lambdas = [
        partial(
            LSW,
            num_warmup_steps=warmup,
            num_training_steps=num_training_steps,
        )
        for warmup in num_warmup_steps
    ]

    return LambdaLR(optimizer=optimizer, lr_lambda=lr_lambdas, last_epoch=last_epoch)
