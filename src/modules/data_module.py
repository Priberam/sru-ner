import lightning as L
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from typing import Union, Dict
from src.modules.utils.action_utils import ActionsVocab
from src.modules.utils.tokenizer import Tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import os
import random
from src.modules.utils.preprocess_datasets import (
    process_datasets,
    get_collection_datasets_from_file,
)


def merge_dictionaries(list_dicts):
    """Given a list of dictionaries [{0 : x, 1: y, 2: z}, {0: a, 1: b, 2: c}]
    it merges the dictionaries and outputs {0 : x, 1: y, 2: z, 3: a, 4: b, 5: c}"""

    merged_dict = {}
    key_offset = 0

    for d in list_dicts:
        for k, v in d.items():
            merged_dict[key_offset + k] = v
        key_offset += len(d)

    return merged_dict


class NERMultiDataset(Dataset):
    def __init__(
        self,
        training_mode: bool,
        dataset_dict: Dict,
        origin: Dict,
        vocab: ActionsVocab,
        max_tokens: int,
        tokenizer_pad_id: int,
        name: str,
        perturb_probs: Union[dict, None] = None,
    ):
        """This represents
            if `training_mode`:
                a train/dev/test split of a dataset
                made from merging the corresponding splits of several datasets
            else:
                a dataset where we want to make predictions

        Args:
            `dataset_dict` {name_dataset : dictionary of data},
            `origin` {name_dataset : str},
            `vocab` actions vocabulary,
            `max_tokens` the max number of tokens allowed per sent,
            `tokenizer_pad_id` the id of the padding token,
            `name` the chosen name for the merged dataset,
            `perturb_probs` info about how to perturb a training batch
        """
        super().__init__()
        self.name = name
        self.training_mode = training_mode
        self.actions_vocab = vocab
        self.max_tokens = max_tokens
        self.perturb_probs = perturb_probs

        # Check that all datasets have the same properties
        all_properties = set()
        for _ in [set(dataset_dict[name].keys()) for name in dataset_dict]:
            all_properties = all_properties.union(_)
        assert all(
            [set(dataset_dict[name].keys()) == all_properties for name in dataset_dict]
        )

        # If there are any properties that are not dicts, expand them
        for dts_name in dataset_dict:
            for key in dataset_dict[dts_name]:
                if not isinstance(dataset_dict[dts_name][key], dict):
                    common_value = dataset_dict[dts_name][key]
                    dataset_dict[dts_name][key] = {
                        id: common_value for id in dataset_dict[dts_name]["doc_id"]
                    }

        # Merge the datasets
        self.data = {}
        for property in all_properties:
            self.data[property] = merge_dictionaries(
                [dataset_dict[name][property] for name in dataset_dict]
            )

        if self.training_mode:
            self.data["source_dataset"] = merge_dictionaries(
                [
                    {_: name for _ in range(len(dataset_dict[name]["doc_id"]))}
                    for name in dataset_dict
                ]
            )
            self.data["source_dataset_path"] = merge_dictionaries(
                [
                    {_: origin[name] for _ in range(len(dataset_dict[name]["doc_id"]))}
                    for name in dataset_dict
                ]
            )

        # Change `tokens_ids` to be tensors
        for sent_id in self.data["token_ids"]:
            self.data["token_ids"][sent_id] = torch.tensor(
                data=self.data["token_ids"][sent_id], dtype=torch.int
            )
        # Add `lengths_tokens_ids`
        self.data["lengths_tokens_ids"] = {}
        for sent_id in self.data["token_ids"]:
            self.data["lengths_tokens_ids"][sent_id] = len(
                self.data["token_ids"][sent_id]
            )

        # Check that there are no sentences with more tokens than the limit
        assert max(self.data["lengths_tokens_ids"].values()) <= max_tokens

        if self.training_mode:
            # Encode the actions_simple as tensors using the provided vocab,
            # with multiple actions per timestep
            self.data["actions_tensor"] = {}
            for sent_id, actions_dec in self.data["actions_simple"].items():
                self.data["actions_tensor"][sent_id] = (
                    self.actions_vocab.create_multilabel_action_tensor(
                        actions_dec + ["EOA"]
                    )
                )

            # Record the number of sentences per dataset
            self.num_sents = dict(Counter(self.data["source_dataset"].values()))

            # Record the number of entities per dataset
            self.num_entities = {
                name: Counter(
                    [
                        x.tag
                        for xs in dataset_dict[name]["gold_entities"].values()
                        for x in xs
                    ]
                )
                for name in dataset_dict
            }

            # Report on the dataset tags
            print(
                f"Created multi-dataset corpus '{self.name}' with "
                f"{len(self)} sentences:"
            )
            for dts_name in sorted(self.num_sents.keys()):
                print(
                    f"{dts_name}\tnumb_sents={self.num_sents[dts_name]:<{8}}"
                    f"\tnumb_tags={dict(self.num_entities[dts_name])}"
                    f" (total: {sum(self.num_entities[dts_name].values())})"
                )
            print()

            # Report on the number of spans not matching tokenization
            num_entities_not_matching_tokens = {
                name: Counter(
                    [
                        x.tag
                        for xs in dataset_dict[name][
                            "gold_entities_not_matching_tokens"
                        ].values()
                        for x in xs
                    ]
                )
                for name in dataset_dict
            }
            if any(len(x) for x in num_entities_not_matching_tokens.values()):
                print("Some spans do not match the tokenization:")
                for dts_name in sorted(num_entities_not_matching_tokens.keys()):
                    if len(num_entities_not_matching_tokens[dts_name]):
                        percents = {
                            tag: round(
                                num_entities_not_matching_tokens[dts_name][tag]
                                / self.num_entities[dts_name][tag]
                                * 100,
                                2,
                            )
                            for tag in num_entities_not_matching_tokens[dts_name]
                        }

                        print(
                            f"{dts_name}"
                            "\tnumb_tags_no_match="
                            f"{dict(num_entities_not_matching_tokens[dts_name])}"
                            f"\tpercentage={percents}"
                        )
                print()

            # For training, use self.num_sents to compute sampler weights
            # The weight is inversally propotional to the number of sentences in
            # the dataset
            if self.name == "train":
                total_num_sents = len(self)
                dataset_weights = {
                    dts_name: total_num_sents / self.num_sents[dts_name]
                    for dts_name in self.num_sents
                }
                weights_list = [
                    dataset_weights[self.data["source_dataset"][id]]
                    for id in sorted(self.data["source_dataset"].keys())
                ]
                self.sampler_weights = torch.DoubleTensor(weights_list)

                if self.perturb_probs is not None:
                    # For pertubations
                    # A dictionary
                    #   source_dataset_name : list of sent ids in NERMultiDataset
                    self.source_to_sent_ids = {
                        src: [
                            id
                            for id, src_i in self.data["source_dataset"].items()
                            if src_i == src
                        ]
                        for src in self.num_sents.keys()
                    }

                    # Print perturb probs
                    print("--> Pertubations summary:")
                    for src in sorted(self.num_sents.keys()):
                        print("Source dts:", src)
                        for pert in sorted(self.num_sents.keys()):
                            val = self.perturb_probs[src][pert]
                            if val != 0:
                                print("\t", pert, val)

        # Define a padding for tokens for the collate function
        self.tokenizer_padding_id = tokenizer_pad_id

    def __len__(self):
        return len(self.data["words_by_space"].keys())

    def __getitem__(self, index):
        return index, {item: self.data[item][index] for item in self.data}

    def merge_tok_info(
        self, data_this_merge: dict, sentence_sep_token: str, sentence_sep_token_id: int
    ):
        """Given:
        a dictionary `data_this_merge` with keys:
                "words_by_space",
                "raw_sentences",
                "tokens",
                "token_ids",
                "offset_mapping",
                "offset_mapping_sentence",
                "subwords_map",
                "token_subword_type",
                "wordpieces",
                "start_end_char_idx_of_words"
        and with List values, and
        a string `sentence_sep_token`, and its corresponding id `sentence_sep_token_id`,

        returns a dictionary with the same keys and merged values."""

        required_keys = [
            "words_by_space",
            "raw_sentences",
            "tokens",
            "token_ids",
            "offset_mapping",
            "offset_mapping_sentence",
            "subwords_map",
            "token_subword_type",
            "wordpieces",
            "start_end_char_idx_of_words",
        ]
        if any(key not in data_this_merge for key in required_keys):
            raise ValueError("Missing keys!")
        number_of_sentences_in_merge = len(data_this_merge["words_by_space"])
        if any(
            len(l) != number_of_sentences_in_merge for l in data_this_merge.values()
        ):
            raise ValueError("Values have different sizes")

        merged_data = {}

        # Join 'words_by_space' with the `sentence_sep_token`
        m_words = []
        for sent_iter, words in enumerate(data_this_merge["words_by_space"]):
            m_words.extend(words)
            if sent_iter != number_of_sentences_in_merge - 1:
                m_words.extend([sentence_sep_token])
        merged_data["words_by_space"] = m_words

        # Join 'raw_sentences' by adding ' {sentence_sep_token} ' in between
        m_raw = ""
        add_to_raw_sents = f" {sentence_sep_token} "
        for sent_iter, raw_s in enumerate(data_this_merge["raw_sentences"]):
            m_raw += raw_s
            if sent_iter != number_of_sentences_in_merge - 1:
                m_raw += add_to_raw_sents
        merged_data["raw_sentences"] = m_raw

        # Join 'tokens' by changing the CLS token in between sentences
        # to the sentence_sep_token, and removing the SEP at the end of all
        # sents that are not the last
        m_tokens = []
        for sent_iter, tok_s in enumerate(data_this_merge["tokens"]):
            tok_list = [t for t in tok_s]
            if sent_iter != number_of_sentences_in_merge - 1:
                tok_list.pop()
            if sent_iter > 0:
                tok_list[0] = sentence_sep_token

            m_tokens.extend(tok_list)
        merged_data["tokens"] = m_tokens

        # Join the 'tokens_ids' by the same procedure, but with ids
        m_tokens_id = []
        for sent_iter, tok_id_s in enumerate(data_this_merge["token_ids"]):
            tok_s_list = [t for t in tok_id_s]
            if sent_iter != number_of_sentences_in_merge - 1:
                tok_s_list.pop()
            if sent_iter > 0:
                tok_s_list[0] = sentence_sep_token_id

            m_tokens_id.extend(tok_s_list)
        merged_data["token_ids"] = m_tokens_id

        # Join the 'offset_mapping' by considering the sentence_sep_token
        # as a word
        m_offset_mapping = []
        for sent_iter, offmap_s in enumerate(data_this_merge["offset_mapping"]):
            os_list = [o for o in offmap_s]
            if sent_iter != number_of_sentences_in_merge - 1:
                os_list.append([(0, len(sentence_sep_token))])

            m_offset_mapping.extend(os_list)
        merged_data["offset_mapping"] = m_offset_mapping

        # Join 'offset_mapping_sentence' using the same idea
        # Also update char indices
        m_offset_mapping_sentence = []
        rsent_lens_with_sent_sep_after = [
            (len(data_this_merge["raw_sentences"][id]) + len(add_to_raw_sents))
            for id in range(len(data_this_merge["raw_sentences"]))
        ]
        add_offset = [0] + [
            sum(rsent_lens_with_sent_sep_after[: i + 1])
            for i in range(len(rsent_lens_with_sent_sep_after) - 1)
        ]

        for sent_iter, offmap_sent_s in enumerate(
            data_this_merge["offset_mapping_sentence"]
        ):
            offseted = []
            for word_span in offmap_sent_s:
                off_tuples = [
                    (
                        tok_span[0] + add_offset[sent_iter],
                        tok_span[1] + add_offset[sent_iter],
                    )
                    for tok_span in word_span
                ]
                offseted.append(off_tuples)

            # Add sentence_sep_token info
            if sent_iter != number_of_sentences_in_merge - 1:
                sep_tok_info = (
                    offseted[-1][-1][1] + 1,
                    offseted[-1][-1][1] + 1 + len(sentence_sep_token),
                )
                offseted.append([sep_tok_info])

            m_offset_mapping_sentence.extend(offseted)
        merged_data["offset_mapping_sentence"] = m_offset_mapping_sentence

        # Update 'start_end_char_idx_of_words' using
        # updated 'offset_mapping_sentence'
        merged_data["start_end_char_idx_of_words"] = [
            (x[0][0], x[-1][-1]) for x in merged_data["offset_mapping_sentence"]
        ]

        # Update wordpieces using updated 'start_end_char_idx_of_words'
        merged_data["wordpieces"] = [
            merged_data["raw_sentences"][s:e]
            for (s, e) in merged_data["start_end_char_idx_of_words"]
        ]

        # Join 'subwords_map', by joining the original ones with the
        # 'offset_mapping_sentence', and adding for the separator
        word_offset = 0
        tok_offset = 0
        m_subwords_map = {}
        merged_data["subwords_map"] = {}
        for sent_iter, swm in enumerate(data_this_merge["subwords_map"]):
            for original_word_id, original_swm in swm.items():
                m_subwords_map[original_word_id + word_offset] = [
                    x + tok_offset for x in original_swm
                ]

            # Add the separator
            if sent_iter != number_of_sentences_in_merge - 1:
                sep_word_id = max(m_subwords_map.keys()) + 1
                sep_tok_id = max([xs for x in m_subwords_map.values() for xs in x]) + 1
                m_subwords_map[sep_word_id] = [sep_tok_id]

            last_word_id = max(m_subwords_map.keys())
            last_tok_id = max([xs for x in m_subwords_map.values() for xs in x])
            word_offset = last_word_id + 1
            tok_offset = last_tok_id + 1
        merged_data["subwords_map"] = m_subwords_map

        # Join the 'token_subword_type' by dropping the last element if
        # the sentence is not the last one
        m_token_subword_type = []
        for sent_iter, t_list in enumerate(data_this_merge["token_subword_type"]):
            if sent_iter != number_of_sentences_in_merge - 1:
                m_token_subword_type.extend(t_list[:-1])
            else:
                m_token_subword_type.extend(t_list)
        merged_data["token_subword_type"] = m_token_subword_type

        return merged_data

    def merge_with_perturbed(self, original_sent: dict, pert_sent: dict):
        # Step 1: Merge word data
        tok_keys = [
            "words_by_space",
            "raw_sentences",
            "tokens",
            "token_ids",
            "offset_mapping",
            "offset_mapping_sentence",
            "subwords_map",
            "token_subword_type",
            "wordpieces",
            "start_end_char_idx_of_words",
        ]

        data_this_merge = {}
        for item in tok_keys:
            if item == "token_ids":
                data_this_merge[item] = [
                    original_sent["token_ids"].tolist(),
                    pert_sent["token_ids"].tolist(),
                ]
            else:
                data_this_merge[item] = [
                    original_sent[item],
                    pert_sent[item],
                ]

        new_sent = self.merge_tok_info(
            data_this_merge=data_this_merge,
            sentence_sep_token=original_sent["tokens"][-1],
            sentence_sep_token_id=original_sent["token_ids"][-1].tolist(),
        )

        # Put token_ids back into tensor
        new_sent["token_ids"] = torch.tensor(
            data=new_sent["token_ids"],
            dtype=original_sent["token_ids"].dtype,
            device=original_sent["token_ids"].device,
        )

        # Change `lengths_tokens_ids`
        new_sent["lengths_tokens_ids"] = new_sent["token_ids"].size(0)

        # Step 2: Merge action data
        # To perturb a sentence in training, the only relevant key
        # is `actions_tensor`
        # We change the last action of the original sent from EOA to SH (because of
        # the separator token of the senteneces), and append the actions of the pert
        # sentence
        timestep_of_EOA = original_sent["actions_tensor"].size(0) - 1
        cat_ac = torch.cat(
            (original_sent["actions_tensor"], pert_sent["actions_tensor"]), dim=0
        )
        cat_ac[timestep_of_EOA, 0] = 0.0
        cat_ac[timestep_of_EOA, self.actions_vocab.shift_ix] = 1.0
        new_sent["actions_tensor"] = cat_ac

        # Step 3: Leave other keys as in the original sentence
        for key in set(original_sent.keys()) - set(new_sent.keys()):
            new_sent[key] = original_sent[key]

        return new_sent

    def collate_fn(self, batch):
        # Indice of the sentence in the NERMultiDataset object
        batch_idx = tuple(batch[id][0] for id in range(len(batch)))
        # Data itself
        batch_data = tuple(batch[id][1] for id in range(len(batch)))

        if self.perturb_probs is not None:
            # Perturb with sentences from other datasets

            new_batch_data = []
            for o_sent in batch_data:
                src_dts = o_sent["source_dataset"]
                this_sent_len = o_sent["lengths_tokens_ids"]

                # Step 1: Select the sentence ids of the other datasets
                # that could be merged with the current sentence
                ids_other_dts = []
                probs = []

                for pert_dts, prob in self.perturb_probs[src_dts].items():
                    if prob != 0:
                        # Add the sentences that fit the tokenizer max len
                        for id in self.source_to_sent_ids[pert_dts]:
                            tok_len = self.data["lengths_tokens_ids"][id]

                            if (this_sent_len + tok_len) - 1 <= self.max_tokens:
                                ids_other_dts.append(id)
                                probs.append(prob)

                # Step 2: From that pool, select the sentence
                if len(ids_other_dts):
                    pert_sent_id = random.choices(
                        population=ids_other_dts, weights=probs, k=1
                    )[0]
                    pert_sent = {
                        item: self.data[item][pert_sent_id] for item in self.data
                    }

                    new_sent = self.merge_with_perturbed(
                        original_sent=o_sent, pert_sent=pert_sent
                    )

                    new_sent["pert_dts"] = pert_sent["source_dataset"]
                    new_batch_data.append(new_sent)
                else:
                    o_sent["pert_dts"] = None
                    new_batch_data.append(o_sent)

            batch_data = tuple(new_batch_data)

        lengths_of_tokens = tuple(x["lengths_tokens_ids"] for x in batch_data)
        positions = tuple(
            index
            for index, _ in sorted(
                enumerate(lengths_of_tokens), key=lambda x: x[1], reverse=True
            )
        )

        # Organize by {property : data in list}, with list data sorted
        # with longest sentence first
        all_properties = tuple(batch_data[0].keys())
        ordered_batch = {}
        for key in all_properties:
            ordered_batch[key] = [batch_data[pos][key] for pos in positions]

        # Actually stack (with padding) the data which is already a tensor
        # i.e. `tokens_ids`, `actions_tensor`
        ordered_batch["token_ids"] = pad_sequence(
            sequences=ordered_batch["token_ids"],
            batch_first=True,
            padding_value=self.tokenizer_padding_id,
        )
        if self.training_mode:
            # Get actions tensor. We pad with 0s
            ordered_batch["actions_tensor"] = pad_sequence(
                sequences=ordered_batch["actions_tensor"],
                batch_first=True,
                padding_value=0.0,
            )

            # Add information on which actions are allowed to be predicted by the model
            # at training time, without incurring in loss
            ordered_batch["allowed_for_pred"] = torch.stack(
                [
                    self.actions_vocab.allowed_for_pred[source]
                    for source in ordered_batch["source_dataset"]
                ]
            )

            if self.perturb_probs is not None:
                # If sentence has been perturbed, make sure the actions of the perturbed
                # sentence are not allowed
                allowed_for_pert = torch.ones_like(ordered_batch["allowed_for_pred"])
                for s_idx, pert_dts in enumerate(ordered_batch["pert_dts"]):
                    if pert_dts is not None:
                        allowed_for_pert[s_idx] = self.actions_vocab.allowed_for_pred[
                            pert_dts
                        ]

                ordered_batch["allowed_for_pred"] *= allowed_for_pert

        # Add info about which tokens are padding
        ordered_batch["tokens_ids_attention_mask"] = (
            ordered_batch["token_ids"] != self.tokenizer_padding_id
        )

        # Add info about the index of the samples
        ordered_batch["dataset_indices"] = tuple(batch_idx[pos] for pos in positions)

        return ordered_batch


class NERDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        mode: str,
        input_data: Union[Dict, str],
        tokenizer_model_name: str,
        change_out_to_shift: bool,
        max_tokens: int,
        train_with_separate_tags: bool,
        merge_sents_mode: list,
        random_seed: int,
        id_to_tag: dict = None,
        perturb_probs: Union[dict, None] = None,
    ):
        """
        DataModule for training/validation/testing.
        The `mode` parameter should be set to 'trainer'.
        `input_data` should be a dictionary like:
                        dts_name:
                            split_paths:
                                split_name: split_path
                                ...
                            mode: 'simple' or 'simple-char'
                            encoding: e.g. 'utf-8'
                            source: should be set to 'local'
                            tag_dict: None or a dict
        `id_to_tag` can either be provided, or is created from data if None.
        """
        super().__init__()

        assert "mode" == 'trainer'

        # Save arguments to self.hparams
        self.save_hyperparameters(logger=False)

        # Track which stages have been set up
        self.hparams.setup_called = []

        # Initilize the tokenizer
        self.tokenizer = Tokenizer(model_name=tokenizer_model_name)

        # Initilize variables used to construct the PyTorch Dataset
        self.data = {}
        self.data_origin = {}

    def setup(self, stage):
        # Skip the setup if already called
        if stage in self.hparams.setup_called:
            return
        self.hparams.setup_called.append(stage)

        # Step 1: Get the data needed for the stage
        # from the self.hparams.input_data.
        needed_splits = []
        if stage == "fit":
            needed_splits.append("train")
        if stage == "fit" or stage == "validate":
            needed_splits.append("dev")
        elif stage == "test":
            needed_splits.append("test")

        # Step 1.1: Make sure all the datasets have the required splits
        for split in needed_splits:
            all_data_have_split = True
            for dts_name in self.hparams.input_data:
                all_data_have_split *= (
                    split in self.hparams.input_data[dts_name]["split_paths"]
                ) and (
                    os.path.exists(
                        self.hparams.input_data[dts_name]["split_paths"][split]
                    )
                )
            if not all_data_have_split:
                raise ValueError(f"Not able to retrive all the {split} datasets.")

        # Only gather the splits we need to process for the `stage`
        dts_paths = {}
        for dts_name, dts_data in self.hparams.input_data.items():
            dts_paths[dts_name] = {}
            for k, v in dts_data.items():
                if k != "split_paths":
                    dts_paths[dts_name][k] = v
                else:
                    dts_paths[dts_name][k] = {}

                    for split in needed_splits:
                        dts_paths[dts_name][k][split] = self.hparams.input_data[
                            dts_name
                        ]["split_paths"][split]

        # Step 1.2: Get the data from local files
        raw_data = get_collection_datasets_from_file(
            local_datasets=dts_paths,
            lower_case=False,
            normalize=False,
        )

        # Step 1.3: Process the datasets
        tag_dictionary = {
            dts_name: dts_paths[dts_name]["tag_dict"] for dts_name in dts_paths
        }

        process_datasets(
            raw_data=raw_data,
            tokenizer=self.tokenizer,
            change_out_to_shift=self.hparams.change_out_to_shift,
            tag_dictionary=tag_dictionary,
            train_with_separate_tags=self.hparams.train_with_separate_tags,
            max_tokens=self.hparams.max_tokens,
            merge_sents_mode=self.hparams.merge_sents_mode,
        )

        # Step 1.4: Add the processed datasets to the self
        self.data = {
            split_name: {
                name: raw_data[name]["splits"][split_name] for name in raw_data
            }
            for split_name in needed_splits
        }
        self.data_origin = {
            split_name: {
                name: raw_data[name]["split_paths"][split_name] for name in raw_data
            }
            for split_name in needed_splits
        }

        # Step 2: Create or load the actions vocab. If we create it, we add it to
        # hparams, so it can be reused from checkpoint
        dataset_to_tag_names_dict = {
            dts_name: raw_data[dts_name]["tags"] for dts_name in raw_data
        }

        if self.hparams.id_to_tag is None:
            id_to_tag = {}
            for dts_name in sorted(dataset_to_tag_names_dict.keys()):
                for tag_name in sorted(dataset_to_tag_names_dict[dts_name]):
                    if tag_name not in id_to_tag.values():
                        id_to_tag[len(id_to_tag)] = tag_name

            # Save this to be able to recreate the same action vocab
            self.hparams.id_to_tag = id_to_tag

        actions_vocab = ActionsVocab.build_for_training(
            id_to_tag=self.hparams.id_to_tag,
            use_out=not self.hparams.change_out_to_shift,
            dataset_to_tag_names_dict=dataset_to_tag_names_dict,
        )

        # Save the actions vocab to self
        self.actions_vocab = actions_vocab

        # Setp 3: Create the PyTorch Dataset, merging the datasets

        # Create merged train dataset
        if stage == "fit":
            self.train_dataset = NERMultiDataset(
                training_mode=True,
                dataset_dict=self.data["train"],
                origin=self.data_origin["train"],
                vocab=self.actions_vocab,
                name="train",
                tokenizer_pad_id=self.tokenizer.tokenizer.pad_token_id,
                max_tokens=self.hparams.max_tokens,
                perturb_probs=self.hparams.perturb_probs,
            )

        # Create merged val dataset
        if stage == "fit" or stage == "validate":
            self.val_dataset = NERMultiDataset(
                training_mode=True,
                dataset_dict=self.data["dev"],
                origin=self.data_origin["dev"],
                vocab=self.actions_vocab,
                name="dev",
                tokenizer_pad_id=self.tokenizer.tokenizer.pad_token_id,
                max_tokens=self.hparams.max_tokens,
                perturb_probs=None,
            )

        # Create merged test dataset
        elif stage == "test":
            self.test_dataset = NERMultiDataset(
                training_mode=True,
                dataset_dict=self.data["test"],
                origin=self.data_origin["test"],
                vocab=self.actions_vocab,
                name="test",
                tokenizer_pad_id=self.tokenizer.tokenizer.pad_token_id,
                max_tokens=self.hparams.max_tokens,
                perturb_probs=None,
            )

    def train_dataloader(self):

        # If using a single dataset
        if len(self.train_dataset.num_sents) == 1:
            return DataLoader(
                dataset=self.train_dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.train_dataset.collate_fn,
                shuffle=True,
                drop_last=False,
                pin_memory=False,
                num_workers=5,
                persistent_workers=True
            )
        # If using more than one dataset
        else:
            # Define the weights for the RandomSampler
            # We use average dataset sampling
            num_samples = int(
                sum(self.train_dataset.num_sents.values())
                / len(self.train_dataset.num_sents)
            )
            generator = torch.Generator()
            generator.manual_seed(self.hparams.random_seed)
            sampler = WeightedRandomSampler(
                weights=self.train_dataset.sampler_weights,
                replacement=True,
                num_samples=num_samples,
                generator=generator,
            )
            print(
                f"TRAIN RANDOM SAMPLER: using {num_samples} samples from train dataset"
            )

            return DataLoader(
                dataset=self.train_dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.train_dataset.collate_fn,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                sampler=sampler,
                num_workers=1,
            )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.val_dataset.collate_fn,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            num_workers=1,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.test_dataset.collate_fn,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            num_workers=1,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.predict_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.predict_dataset.collate_fn,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            num_workers=5,
        )
