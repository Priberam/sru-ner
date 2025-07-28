from typing import Union, Dict
from src.modules.utils.read_local_datasets import ingest_dataset_from_file
from src.modules.utils.tokenizer import Tokenizer
from src.modules.utils.action_utils import (
    from_simple_char_ac_list_to_list_entities_with_char_idx,
    from_simple_ac_list_to_list_entities_with_word_idx,
    list_entities_char_idx_to_word_idx,
    list_entities_to_actions_encoding_with_OUT,
    get_inner_action,
    prepend_to_action_tag,
    add_to_char_ids_in_simple_char_ac_sequence,
    ActionsVocab,
)
from functools import partial
import torch
from src.modules.utils.sentence_merger import (
    normalize_sent_lengths,
    reindex_dts_for_merge,
    maximize_sent_lens,
)
from tqdm import tqdm


def process_datasets(
    raw_data: dict,
    tokenizer: Tokenizer,
    change_out_to_shift: bool,
    tag_dictionary: Union[None, dict],
    train_with_separate_tags: bool,
    max_tokens: int,
    merge_sents_mode: list,
) -> None:
    """Whole processing pipeline for train/dev/test datasets.
    Changes and adds additional information to the argument `raw_data`.

    Arguments:
        `raw_data`: dict with the following structure
            `dataset_name`:
                'split_paths':
                    'train':
                    'dev':
                    ...
                'mode': 'simple' OR 'simple-char'
                'encoding': (e.g. 'uft-8')
                splits:
                    'train':
                        'words_by_space': {0: ['Hepatocyte', 'nuclear', ...], 1: ...}
                        'raw_sentences': {0: 'Hepatocyte nuclear ...', 1: ...}
                        'doc_id': {0: 0, 1: ...}
                        'original_actions':
                            {0: ['TR(Gene,0)', 'RE(Gene,27)', ...], 1: ...}

        `tokenizer`:
        `change_out_to_shift`:
        `tag_dictionary`:
        `train_with_separate_tags`:
        `max_tokens`:
        `merge_sents_mode`:

    """

    # Step 1: Tokenize the datasets
    raw_data = tokenize_datasets(tokenizer=tokenizer, raw_data=raw_data)

    # At this point, raw_data is a nested dictionary with the following structure:
    #
    #   {dataset_name}:
    #       'split_paths':
    #           'train':
    #           'dev':
    #           ...
    #       'mode': 'simple' OR 'simple-char'
    #       'encoding': (e.g. 'uft-8')
    #       'splits':
    #           'train':
    #               'words_by_space': {0: ['Hepatocyte', 'nuclear', ...], 1: ...}
    #               'raw_sentences': {0: 'Hepatocyte nuclear ...', 1: ...}
    #               'doc_id': {0: 0, 1: ...}
    #               'tokens': {0: ['[CLS]', 'hepatocyte', 'nuclear', ...], 1:  ...}
    #               'tokens_ids': {0: [2, 12906, 4253, ...] 1: ...}
    #               'offset_mapping': {0: [[(0,10)], [(0,7)], ...], 1: ...}
    #               'offset_mapping_sentence': {0: [[(0,10)], [(11,18)], ...], ...}
    #               'subwords_map': {0: {0: [0], 1: [1], ...}, 1: ...}
    #               'token_subword_type': {0: [-1, 3, 3, ...] 1: ...}
    #               'wordpieces': {0: ['Hepatocyte', 'nuclear', ...], 1: ...}
    #               'start_end_char_idx_of_words': {0: [(0,10), (11,18), ...], ...}

    #               (and if this info is available:)
    #               'original_actions':
    #                   {0: ['TRANSITION(Gene,0)', 'REDUCE(Gene,27)', ...], 1: ...}
    #           'dev':
    #               same thing...

    # Step 1.1 Check if there are splits that have sentences with more tokens than
    # the limit. If so, split them and re-index dataset
    splits_long_sents = [
        (dts_name, split_name)
        for dts_name in raw_data
        for split_name in raw_data[dts_name]["splits"]
        if any(
            [
                len(x) > max_tokens
                for x in raw_data[dts_name]["splits"][split_name]["token_ids"].values()
            ]
        )
    ]
    if len(splits_long_sents):
        for dts_name, split_name in splits_long_sents:

            break_output = break_long_sents(
                data=raw_data[dts_name]["splits"][split_name],
                gold_spans_mode=raw_data[dts_name]["mode"],
                max_tokens=max_tokens,
            )

            raw_data[dts_name]["splits"][split_name] = break_output["splitted_data"]
            num_splitted_sents = break_output["num_splitted_sents"]
            numb_deleted_sents = break_output["numb_deleted_sents"]

            print(
                f"Splitting in ({dts_name}, {split_name}): split {num_splitted_sents} "
                + f"and deleted {numb_deleted_sents}"
            )

    # Add mode so we know how to merge the actions, if needed
    for dts_name in raw_data:
        for split_name in raw_data[dts_name]["splits"]:
            raw_data[dts_name]["splits"][split_name]["mode"] = raw_data[dts_name][
                "mode"
            ]

    # Step 2: Merge the sentences of the datasets if desired
    if merge_sents_mode is not None:
        # We keep the datasets without docseps as is, and merge on the rest
        keep_dts_names = set()
        for dts_name in raw_data:
            has_doc_seps = True
            for splits in raw_data[dts_name]["splits"].values():
                num_docs = len(set(splits["doc_id"].values()))
                if num_docs == 1:
                    has_doc_seps = False

            if not has_doc_seps:
                keep_dts_names.add(dts_name)
        merge_dts_names = set(raw_data.keys()) - keep_dts_names

        if len(merge_sents_mode) == 3:
            forced_merge = set(merge_sents_mode[2])
            if any(d not in raw_data.keys() for d in forced_merge):
                raise ValueError('Error in forced merge. Dataset name incorrect.')
            print('\n--> FORCING MERGE in the datasets: ', forced_merge, '\n')
            merge_dts_names = merge_dts_names.union(forced_merge)
            keep_dts_names = keep_dts_names - forced_merge

        to_merge = {dts_name: raw_data[dts_name] for dts_name in merge_dts_names}

        if len(to_merge):
            print("Merging sentences of datasets: ", merge_dts_names)
        if len(keep_dts_names):
            print("NOT merging sentences of datasets: ", keep_dts_names)

        if len(to_merge):
            merge_datasets(
                raw_data=to_merge,
                merge_sents_mode=merge_sents_mode,
                max_tokens=max_tokens,
                sentence_sep_token=tokenizer.tokenizer.sep_token,
                sentence_sep_token_id=tokenizer.tokenizer.sep_token_id,
            )
    else:
        print("-> Not merging sentences.")

    # Step 2: Add action information to the datasets
    # We can change the OUTs to SHIFTs, and change the name of the tags
    raw_data = add_action_information(
        raw_data=raw_data,
        change_out_to_shift=change_out_to_shift,
        tag_dictionary=tag_dictionary,
    )

    # This adds:
    #
    # a dict to raw_data[dataset_name][`tags`]:
    #   {dataset_name: set of tag names in the dataset};
    #
    # and the following keys to raw_data[dataset_name]['splits'][split_name]:
    #   `actions_simple`:
    #       {0: ['TRANSITION(Gene)', 'SHIFT', ..., 'REDUCE(Gene)', ...], 1: ...}
    #       NOTE: While raw_data[dataset_name]['splits'][split_name]['original_actions']
    #             is the action sequence copied directly from the dataset,
    #             raw_data[dataset_name]['splits'][split_name]['actions_simple'] is
    #             always in simple format.
    #   `gold_entities`:
    #       {0: ['Hepatocyte nuclear factor-6'(0, 27)//Gene, ...], ...}
    #       NOTE: the indexing matches the mode of the dataset
    #   gold_entities_matching_tokens:
    #       {0: ['Hepatocyte nuclear factor-6'(0, 27)//Gene, ...], ...}
    #       NOTE: indexing is always at wordpiece level
    #   gold_entities_not_matching_tokens:
    #       (similar)
    #       NOTE: indexing matches the mode of the dataset
    #
    # In order to create the actions vocab, we need `id_to_tag`,
    # a dict mapping an index to a tag name. This is used to enumerate
    # the TR/RE actions.
    # For mode == 'prediction', one should provide the `id_to_tag`
    # In mode == 'trainer', one can provide this dict (e.g. for resuming training)
    # or it is created from the names of the datasets

    # If train_with_separate_tags, we change the entries in raw_data to include
    # the dataset name in the tags
    if train_with_separate_tags:
        # Make sure tags do not have underscores
        assert not any(
            [
                any(["$" in x for x in raw_data[dts_name]["tags"]])
                for dts_name in raw_data
            ]
        )

        print("\n-> Training with separate tags enabled\n")

        for dataset_name in raw_data:

            raw_data[dataset_name]["tags"] = set(
                map(
                    lambda x: dataset_name + "$" + x,
                    raw_data[dataset_name]["tags"],
                )
            )

            change_fn = partial(prepend_to_action_tag, dataset_name)

            for split_name in raw_data[dataset_name]["splits"]:
                for sent_id in raw_data[dataset_name]["splits"][split_name][
                    "words_by_space"
                ]:
                    raw_data[dataset_name]["splits"][split_name]["actions_simple"][
                        sent_id
                    ] = list(
                        map(
                            change_fn,
                            raw_data[dataset_name]["splits"][split_name][
                                "actions_simple"
                            ][sent_id],
                        ),
                    )

                    for key in [
                        "gold_entities",
                        "gold_entities_matching_tokens",
                        "gold_entities_not_matching_tokens",
                    ]:
                        for span in raw_data[dataset_name]["splits"][split_name][key][
                            sent_id
                        ]:
                            span.prepend_to_tag(to_add=dataset_name)

    print("Data preprocessing DONE!")


def get_collection_datasets_from_file(
    local_datasets: dict,
    lower_case: bool,
    normalize: bool,
):
    """Used to retrieve data from local file.

    Inputs:
    `local_datasets`:
        dict with keys the names of the datasets, and values dicts
        whose keys should include
            `split_paths` (dict {split: path}),
            `mode` ('simple' or 'simple-char'), and
            `encoding` (e.g. 'utf-8').
    `lower_case`: flag to indicate if we should lower-case all words
    `normalize`: unicode char normalization

    Returns:
        same dict as input but with a new key 'splits'. Under the each split,
        we have a dictionary with keys-values:
            `words_by_space`: list of lists of space separated "words" per sentence
            `raw_sentences`: list of strings representing the original sentences
            `original_actions`: list of lists with the original action sequence
            `doc_id`: list of integers, representing the document of the sentence
    """
    if not local_datasets:
        raise ValueError("No datasets selected")
    else:
        raw_data = {}

        print("Getting datasets from local file")

        for dts_name in local_datasets:

            raw_data[dts_name] = {}

            for k, v in local_datasets[dts_name].items():
                raw_data[dts_name][k] = v

            raw_data[dts_name]["splits"] = {}

            for split_name, split_path in raw_data[dts_name]["split_paths"].items():
                raw_data[dts_name]["splits"][split_name] = ingest_dataset_from_file(
                    file_path=split_path,
                    mode=raw_data[dts_name]["mode"],
                    lower_case=lower_case,
                    normalize=normalize,
                    encoding=raw_data[dts_name]["encoding"],
                )

    return raw_data


def tokenize_datasets(tokenizer: Tokenizer, raw_data: dict):
    """Input `raw_data` should have the following structure

    {`dts_name`:
        {'splits':
            {`split_name`:
                {
                'words_by_space' : {id: [word0, word1, ...]},
                'raw_sentences' : {id: whole sentence}
                }
            }
        }
        'mode': 'simple' or 'simple-char',
    }

    Can be used both for training and prediction. In the latter case,
    select mode='simple-char' to use the tokenizer on the whole sentence.
    """

    # We use the provided pre-tokenization if in simple mode, or the
    # full sentence otherwise
    for dts_name in raw_data:
        for split in raw_data[dts_name]["splits"].values():
            split["tokens"] = {}
            split["token_ids"] = {}
            split["offset_mapping"] = {}
            split["offset_mapping_sentence"] = {}
            split["subwords_map"] = {}
            split["token_subword_type"] = {}

            split["wordpieces"] = {}
            split["start_end_char_idx_of_words"] = {}

            already_pretoken = True if raw_data[dts_name]["mode"] == "simple" else False
            word_data = (
                split["words_by_space"]
                if raw_data[dts_name]["mode"] == "simple"
                else split["raw_sentences"]
            )

            for sent_id, words in word_data.items():
                tokenization_data = tokenizer.tokenize_sent(words, already_pretoken)
                for k, v in tokenization_data.items():
                    split[k][sent_id] = v

                split["start_end_char_idx_of_words"][sent_id] = [
                    (x[0][0], x[-1][-1])
                    for x in split["offset_mapping_sentence"][sent_id]
                ]
                split["wordpieces"][sent_id] = [
                    split["raw_sentences"][sent_id][s:e]
                    for (s, e) in split["start_end_char_idx_of_words"][sent_id]
                ]

    return raw_data


def add_action_information(
    raw_data: dict,
    change_out_to_shift: bool,
    tag_dictionary: Union[None, dict],
):
    # Step 1: Get the gold entity list
    # At this stage, the entity list of datasets in simple-char format are
    # with char-level indexes
    # We also add the tag set of the dataset to raw_data[dts_name]
    for dts_name in raw_data:
        tags_this_dataset = set()
        for split in raw_data[dts_name]["splits"].values():
            split["gold_entities"] = {}

            decoding_fn = (
                from_simple_char_ac_list_to_list_entities_with_char_idx
                if raw_data[dts_name]["mode"] == "simple-char"
                else from_simple_ac_list_to_list_entities_with_word_idx
            )
            word_data = (
                split["words_by_space"]
                if raw_data[dts_name]["mode"] == "simple"
                else split["raw_sentences"]
            )

            for sent_id, original_ac_seq in split["original_actions"].items():
                ents = sorted(
                    decoding_fn(original_ac_seq, word_data[sent_id]),
                    key=lambda x: (x.start, -x.end),
                )

                split["gold_entities"][sent_id] = ents

                for ent in ents:
                    tags_this_dataset.add(ent.tag)

        raw_data[dts_name]["tags"] = tags_this_dataset

    # Step 2: Alter the gold entity list according to tag_dictionary
    if tag_dictionary is not None:
        print("Tag modifications:")
        for dts_name in raw_data:
            if (
                dts_name not in tag_dictionary
                or tag_dictionary[dts_name] is None
                or all(
                    [
                        o_tag == n_tag
                        for o_tag, n_tag in tag_dictionary[dts_name].items()
                    ]
                )
            ):
                print(f"{dts_name}: keeping all tags the same")
                tag_dictionary[dts_name] = {
                    o_tag: o_tag for o_tag in raw_data[dts_name]["tags"]
                }
            else:
                if any(
                    x not in tag_dictionary[dts_name].keys()
                    for x in raw_data[dts_name]["tags"]
                ):
                    raise ValueError(
                        (
                            f"There are tags in dataset {dts_name} that are not in the"
                            + "config tag_dict"
                        )
                    )

                changed_tags = {
                    o_tag: tag_dictionary[dts_name][o_tag]
                    for o_tag in raw_data[dts_name]["tags"]
                    if (
                        o_tag != tag_dictionary[dts_name][o_tag]
                        and tag_dictionary[dts_name][o_tag] is not None
                    )
                }
                removed_tags = {
                    o_tag
                    for o_tag in raw_data[dts_name]["tags"]
                    if tag_dictionary[dts_name][o_tag] is None
                }
                if len(changed_tags) or len(removed_tags):
                    print(f"{dts_name}:")
                    if len(changed_tags):
                        print("-> Changing the following tags")
                        for o_tag, n_tag in changed_tags.items():
                            print(f"{o_tag} -> {n_tag}")
                    if len(removed_tags):
                        print("-> Removing the following tags:", sorted(removed_tags))

                # Actually change the tags
                for split_name in raw_data[dts_name]["splits"]:
                    modified_gold_actions = {}
                    for sent_id in raw_data[dts_name]["splits"][split_name][
                        "gold_entities"
                    ]:
                        new_ents = []
                        for ent in raw_data[dts_name]["splits"][split_name][
                            "gold_entities"
                        ][sent_id]:
                            if ent.tag in removed_tags:
                                continue
                            elif ent.tag in changed_tags:
                                new_tag = ent.copy()
                                new_tag.tag = changed_tags[ent.tag]
                                new_ents.append(new_tag)
                            else:
                                new_ents.append(ent)
                        modified_gold_actions[sent_id] = new_ents
                    raw_data[dts_name]["splits"][split_name][
                        "gold_entities"
                    ] = modified_gold_actions

                # Change the tag dict at the level of dataset
                for tag in removed_tags:
                    raw_data[dts_name]["tags"].remove(tag)
                for o_tag, n_tag in changed_tags.items():
                    raw_data[dts_name]["tags"].remove(o_tag)
                    raw_data[dts_name]["tags"].add(n_tag)

    # Step 2: Convert gold entity list to action sequence `actions_simple`
    # based on tokens
    for dts_name in raw_data:
        for split_name in raw_data[dts_name]["splits"]:
            split = raw_data[dts_name]["splits"][split_name]
            split["actions_simple"] = {}

            if raw_data[dts_name]["mode"] == "simple":
                for sent_id, gold_entities in tqdm(
                    split["gold_entities"].items(),
                    desc=f"Preparing actions for ({dts_name}, {split_name})",
                ):
                    split["actions_simple"][sent_id] = (
                        list_entities_to_actions_encoding_with_OUT(
                            words=split["words_by_space"][sent_id],
                            labels=gold_entities,
                        )
                    )

                # By construction all entities match the tokens
                split["gold_entities_matching_tokens"] = {
                    sent_id: [ent.copy() for ent in ent_list]
                    for sent_id, ent_list in split["gold_entities"].items()
                }
                split["gold_entities_not_matching_tokens"] = {
                    sent_id: [] for sent_id in split["gold_entities_matching_tokens"]
                }

            # In the simple-char case, we only allow entities that fit in a word
            elif raw_data[dts_name]["mode"] == "simple-char":

                split["gold_entities_matching_tokens"] = {}
                split["gold_entities_not_matching_tokens"] = {}

                for sent_id, gold_entities in tqdm(
                    split["gold_entities"].items(),
                    desc=f"Preparing actions for ({dts_name}, {split_name})",
                ):

                    accepted_entities, removed_entities = (
                        list_entities_char_idx_to_word_idx(
                            entities_this_sent=gold_entities,
                            s_e_token=split["start_end_char_idx_of_words"][sent_id],
                        )
                    )

                    split["gold_entities_matching_tokens"][sent_id] = sorted(
                        accepted_entities, key=lambda x: (x.start, -x.end)
                    )
                    split["gold_entities_not_matching_tokens"][sent_id] = sorted(
                        removed_entities, key=lambda x: (x.start, -x.end)
                    )

                    split["actions_simple"][sent_id] = (
                        list_entities_to_actions_encoding_with_OUT(
                            words=split["wordpieces"][sent_id],
                            labels=split["gold_entities_matching_tokens"][sent_id],
                        )
                    )

    # Step 4: If desired, change the actions_simple by removing all OUTs
    # and transform into SHIFTs
    if change_out_to_shift:
        print("WARNING: Changing OUTs to SHIFTs")
        for dts_name in raw_data:
            for split_name in raw_data[dts_name]["splits"]:
                split = raw_data[dts_name]["splits"][split_name]
                modified_actions = {}

                for sent_id, ac_seq in split["actions_simple"].items():
                    modified_actions[sent_id] = [
                        "SHIFT" if item == "OUT" else item for item in ac_seq
                    ]

                split["actions_simple"] = modified_actions

    return raw_data


def merge_datasets(
    raw_data: dict,
    merge_sents_mode: list,
    max_tokens: int,
    sentence_sep_token: str,
    sentence_sep_token_id: int,
):

    split_names = set(s for dts in raw_data.values() for s in set(dts["splits"].keys()))

    for split_name in split_names:
        split_data = {
            dts_name: raw_data[dts_name]["splits"][split_name] for dts_name in raw_data
        }

        if merge_sents_mode[0] == "maximize":
            sents_to_merge = maximize_sent_lens(
                data=split_data,
                max_numb_tokens=max_tokens,
                swap=merge_sents_mode[1],
            )
        elif merge_sents_mode[0] == "normalize":
            sents_to_merge = normalize_sent_lengths(
                data=split_data,
                max_numb_tokens=max_tokens,
                mode=merge_sents_mode[1],
            )

        split_data = reindex_dts_for_merge(
            data=split_data,
            merge_ids=sents_to_merge,
            sentence_sep_token=sentence_sep_token,
            sentence_sep_token_id=sentence_sep_token_id,
        )

        print(
            f"Changes due to merge on split '{split_name}' (mode={merge_sents_mode}):"
        )
        for dts_name in split_data:
            sent_lens_before = [
                len(x)
                for x in raw_data[dts_name]["splits"][split_name]["wordpieces"].values()
            ]
            sent_lens_after = [
                len(x) for x in split_data[dts_name]["wordpieces"].values()
            ]
            documents_before = [
                raw_data[dts_name]["splits"][split_name]["doc_id"][i]
                for i in sorted(
                    raw_data[dts_name]["splits"][split_name]["doc_id"].keys()
                )
            ]
            documents_after = [
                split_data[dts_name]["doc_id"][i]
                for i in sorted(split_data[dts_name]["doc_id"].keys())
            ]

            numb_sents_before = len(sent_lens_before)
            numb_sents_after = len(sent_lens_after)
            av_sent_before = round(sum(sent_lens_before) / numb_sents_before, 2)
            av_sent_after = round(sum(sent_lens_after) / numb_sents_after, 2)
            max_len_before = max(sent_lens_before)
            max_len_after = max(sent_lens_after)
            nvariance_before = round(
                sum([(x - av_sent_before) ** 2 for x in sent_lens_before])
                / len(sent_lens_before)
                / (av_sent_before**2),
                2,
            )
            nvariance_after = round(
                sum([(x - av_sent_after) ** 2 for x in sent_lens_after])
                / len(sent_lens_after)
                / (av_sent_after**2),
                2,
            )

            from torch_scatter import scatter_mean

            sent_lens_before = torch.tensor(sent_lens_before, dtype=torch.float)
            documents_before = torch.tensor(documents_before)
            sent_lens_after = torch.tensor(sent_lens_after, dtype=torch.float)
            documents_after = torch.tensor(documents_after)

            av_sent_per_doc_before = round(
                scatter_mean(sent_lens_before, documents_before).mean().item(), 2
            )
            av_sent_per_doc_after = round(
                scatter_mean(sent_lens_after, documents_after).mean().item(), 2
            )

            nvariance_doc_before = 0
            for doc_id in documents_before.unique():
                sents_lens_this_doc = sent_lens_before[documents_before == doc_id]
                nvariance_doc_before += torch.var(
                    sents_lens_this_doc, dim=0, correction=0
                ) / (sents_lens_this_doc.mean() ** 2)
            nvariance_doc_before /= len(documents_before.unique())
            nvariance_doc_before = round(nvariance_doc_before.item(), 4)

            nvariance_doc_after = 0
            for doc_id in documents_after.unique():
                sents_lens_this_doc = sent_lens_after[documents_after == doc_id]
                nvariance_doc_after += torch.var(
                    sents_lens_this_doc, dim=0, correction=0
                ) / (sents_lens_this_doc.mean() ** 2)
            nvariance_doc_after /= len(documents_after.unique())
            nvariance_doc_after = round(nvariance_doc_after.item(), 4)

            print(dts_name)
            print(f"Total # sentences {numb_sents_before}->{numb_sents_after}")
            print(f"Max len of a sentence {max_len_before}->{max_len_after}")
            print("Av. len of sentences")
            print(f"\t Dataset-wise: {av_sent_before}->{av_sent_after}")
            print(
                f"\t Document-wise: {av_sent_per_doc_before}->{av_sent_per_doc_after}"
            )
            print("Normalized variance in len of sentences")
            print(f"\t Dataset-wise: {nvariance_before}->{nvariance_after}")
            print(f"\t Document-wise: {nvariance_doc_before}->{nvariance_doc_after}")
            print()

            """ print(
                f"{dts_name}\tNumb. sents {numb_sents_before}->{numb_sents_after}\t"
                f"Max. len {max_len_before}->{max_len_after}\t"
                f"Av. len {av_sent_before}->{av_sent_after}\t"
                f"Var. len {variance_before}->{variance_after}"
            ) """

        for dts_name in raw_data:
            raw_data[dts_name]["splits"][split_name] = split_data[dts_name]


def break_long_sents(data: dict, gold_spans_mode: str, max_tokens: int) -> dict:
    """Assuming:
    a dict `data` where each key maps to a dict {sent_id: value}; example keys:
        'doc_id': {sent_id: doc_id},
        'words_by_space': {sent_id: list of strings},
        'raw_sentences': {sent_id: string},
        'tokens': {sent_id: list of strings},
        'token_ids': {sent_id: list of integers},
        'offset_mapping': {sent_id: list of lists of tuples},
        'offset_mapping_sentence': {sent_id: list of lists of tuples},
        'subwords_map': {sent_id: {word_idx : list of token ids that map to it}},
        'token_subword_type': {sent_id: list of integers},
        'wordpieces': {sent_id: list of strings},
        'start_end_char_idx_of_words': {sent_id: list of tuples}
        'original_actions': {sent_id: list of actions},

    a string `gold_spans_mode` equal to:
        'simple' or 'simple-char': when data has info about gold spans; in this case,
            we never split a sentence in the middle of a gold span
        None: otherwise

    and the number of allowed tokens `max_tokens`,

    returns `data` after splitting sentences and re-indexing."""

    keys = set(data.keys())
    splitted_data = {k: {} for k in keys}
    original_len = [len(data[k]) for k in keys][0]
    new_sent_iter = 0
    start_sent_tok = data["tokens"][0][0]
    start_sent_tok_id = data["token_ids"][0][0]
    end_sent_tok = data["tokens"][0][-1]
    end_sent_tok_id = data["token_ids"][0][-1]
    start_sent_sw_type = data["token_subword_type"][0][0]
    end_sent_sw_type = data["token_subword_type"][0][-1]

    numb_deleted_sents = 0
    num_splitted_sents = 0

    for o_sent_iter in range(original_len):
        if len(data["token_ids"][o_sent_iter]) <= max_tokens:
            # Just copy
            for k in keys:
                splitted_data[k][new_sent_iter] = data[k][o_sent_iter]

            new_sent_iter += 1
        else:
            # Break sentences

            # If we have gold spans, we don't want to break inside a mention
            # We record the idx of wordpieces inside mentions
            discount_on_max = {}
            if gold_spans_mode is not None:

                if not (gold_spans_mode in ["simple-char", "simple"]):
                    raise ValueError("Mode should be 'simple' or 'simple-char'.")

                decoding_fn = (
                    from_simple_char_ac_list_to_list_entities_with_char_idx
                    if gold_spans_mode == "simple-char"
                    else from_simple_ac_list_to_list_entities_with_word_idx
                )
                word_data = (
                    data["words_by_space"][o_sent_iter]
                    if gold_spans_mode == "simple"
                    else data["raw_sentences"][o_sent_iter]
                )

                ents = sorted(
                    decoding_fn(data["original_actions"][o_sent_iter], word_data),
                    key=lambda x: (x.start, -x.end),
                )

                wp_idx_with_mentions = []
                if gold_spans_mode == "simple-char":
                    for ent in ents:
                        wp_idxs = []
                        for wp_i, wp in enumerate(
                            data["offset_mapping_sentence"][o_sent_iter]
                        ):
                            if ent.start <= wp[0][0] and wp[-1][-1] <= ent.end:
                                wp_idxs.append(wp_i)
                        if len(
                            wp_idxs
                        ):  # Since some entities might not match the tokenization
                            wp_idx_with_mentions.append(wp_idxs)
                elif gold_spans_mode == "simple":
                    for ent in ents:
                        wp_idxs = []
                        for wp_i in range(len(data["words_by_space"][o_sent_iter])):
                            if wp_i in range(ent.start, ent.end):
                                wp_idxs.append(wp_i)
                        wp_idx_with_mentions.append(wp_idxs)

                for wp_ent in wp_idx_with_mentions:
                    numb_toks_ent = sum(
                        [
                            len(data["offset_mapping"][o_sent_iter][wpi])
                            for wpi in wp_ent
                        ]
                    )
                    discount_on_max[wp_ent[0]] = numb_toks_ent

            # ------------------------
            # Step 1: Find where to break in terms of wordpiece index
            # We look at the offset_mapping. Since this maps a wordpiece index to
            # the token index (not including the CLS/SEP), we deduct 2 to max_tokens
            # We also record the corresponding break index in terms of tokens
            wp_spans = []
            tok_s_e = []
            start_word_idx = 0
            start_token_idx = 1
            current_tokens = 0
            len_wps = [len(x) for x in data["offset_mapping"][o_sent_iter]]

            for word_idx, token_count in enumerate(len_wps):
                discount = (
                    discount_on_max[word_idx] if word_idx in discount_on_max else 0
                )
                if current_tokens + token_count > max_tokens - 2 - discount:
                    # Store the current span before exceeding the limit
                    wp_spans.append((start_word_idx, word_idx))
                    tok_s_e.append((start_token_idx, start_token_idx + current_tokens))

                    # Start a new span
                    start_word_idx = word_idx
                    start_token_idx += current_tokens
                    current_tokens = 0  # Reset token count

                current_tokens += token_count

            # Add the last segment if any words remain
            if start_word_idx < len(len_wps):
                wp_spans.append((start_word_idx, len(len_wps)))
                tok_s_e.append((start_token_idx, start_token_idx + current_tokens))

            if gold_spans_mode is not None:
                # Check we have not split on mentions
                completely_contained = lambda x: (
                    x[1][0] <= x[0][0] <= x[0][-1] < x[1][-1]
                )
                assert all(
                    any(completely_contained((wp_e, s)) for s in wp_spans)
                    for wp_e in wp_idx_with_mentions
                )

            # Check that the split was sucessful. Otherwise, we delete the sentence
            tot_toks_new_sents = [x[1] - x[0] for x in tok_s_e]
            if max(tot_toks_new_sents) > (max_tokens - 2):
                numb_deleted_sents += 1
                continue
            else:
                num_splitted_sents += 1

            # ------------------------
            # Step 2: Split the word data
            split_sents = {k: [] for k in keys}

            # wordpieces
            split_sents["wordpieces"] = [
                data["wordpieces"][o_sent_iter][s[0] : s[1]] for s in wp_spans
            ]

            # offset_mapping
            split_sents["offset_mapping"] = [
                data["offset_mapping"][o_sent_iter][s[0] : s[1]] for s in wp_spans
            ]

            # offset_mapping_sentence and raw_sentences
            split_sents["offset_mapping_sentence"] = []
            split_sents["raw_sentences"] = []
            s_e_char_sents = []
            for sp in wp_spans:

                start_wp = sp[0]
                end_wp = sp[1]

                start_char = data["offset_mapping_sentence"][o_sent_iter][start_wp][0][
                    0
                ]
                end_char = data["offset_mapping_sentence"][o_sent_iter][end_wp - 1][-1][
                    -1
                ]
                s_e_char_sents.append([start_char, end_char])

                split_sents["raw_sentences"].append(
                    data["raw_sentences"][o_sent_iter][start_char:end_char]
                )

                this_sent = []
                for sps in data["offset_mapping_sentence"][o_sent_iter][
                    start_wp:(end_wp)
                ]:
                    this_wp = []
                    for tup in sps:
                        this_wp.append((tup[0] - start_char, tup[1] - start_char))
                    this_sent.append(this_wp)

                split_sents["offset_mapping_sentence"].append(this_sent)

            # start_end_char_idx_of_words
            split_sents["start_end_char_idx_of_words"] = [
                [(wp[0][0], wp[-1][-1]) for wp in sent]
                for sent in split_sents["offset_mapping_sentence"]
            ]

            # Sanity check
            numb_split_sents = len(wp_spans)
            wp_according_to_se_chars = []
            for si in range(numb_split_sents):
                for wp_c in split_sents["start_end_char_idx_of_words"][si]:
                    start = wp_c[0]
                    end = wp_c[1]
                    aux = split_sents["raw_sentences"][si][start:end]
                    wp_according_to_se_chars.append(aux)

            assert wp_according_to_se_chars == data["wordpieces"][o_sent_iter]

            # words_by_space
            split_sents["words_by_space"] = [
                t.split(" ") for t in split_sents["raw_sentences"]
            ]

            # doc_id
            split_sents["doc_id"] = [data["doc_id"][o_sent_iter]] * numb_split_sents

            # subwords_map
            split_sents["subwords_map"] = []
            for sent_sw_map in split_sents["offset_mapping"]:
                this_sent_sw_map = {}
                tok_counter = 0
                for wp_counter, wp in enumerate(sent_sw_map):
                    aux = list(range(tok_counter, tok_counter + len(wp)))
                    this_sent_sw_map[wp_counter] = aux
                    tok_counter += len(wp)
                split_sents["subwords_map"].append(this_sent_sw_map)

            # tokens
            split_sents["tokens"] = []
            tokens_no_special = [
                data["tokens"][o_sent_iter][t[0] : t[1]] for t in tok_s_e
            ]
            split_sents["tokens"] = [
                [start_sent_tok] + no_special + [end_sent_tok]
                for no_special in tokens_no_special
            ]

            # token_ids
            split_sents["token_ids"] = []
            token_ids_no_special = [
                data["token_ids"][o_sent_iter][t[0] : t[1]] for t in tok_s_e
            ]
            split_sents["token_ids"] = [
                [start_sent_tok_id] + no_special + [end_sent_tok_id]
                for no_special in token_ids_no_special
            ]

            # token_subword_type
            split_sents["token_subword_type"] = []
            sw_no_special = [
                data["token_subword_type"][o_sent_iter][t[0] : t[1]] for t in tok_s_e
            ]
            split_sents["token_subword_type"] = [
                [start_sent_sw_type] + no_special + [end_sent_sw_type]
                for no_special in sw_no_special
            ]

            if "char_idx_in_post" in data:
                # char_idx_in_post
                start_char_o_sent = data["char_idx_in_post"][o_sent_iter][0]
                split_sents["char_idx_in_post"] = [
                    (start_char_o_sent + x[0], start_char_o_sent + x[1])
                    for x in s_e_char_sents
                ]

            # ------------------------
            # Step 3: Split the span data, if it exists

            if gold_spans_mode is not None:

                # original_actions
                split_sents["original_actions"] = []
                if gold_spans_mode == "simple":
                    for sent_i in range(numb_split_sents):
                        s_w_id = wp_spans[sent_i][0]
                        e_w_id = wp_spans[sent_i][1]
                        relavant_ac = []
                        w_count = 0
                        for ac in data["original_actions"][o_sent_iter]:
                            if w_count in range(s_w_id, e_w_id):
                                relavant_ac.append(ac)

                            if ac == "OUT" or ac == "SHIFT":
                                w_count += 1

                        split_sents["original_actions"].append(relavant_ac)

                elif gold_spans_mode == "simple-char":
                    for sent_i in range(numb_split_sents):
                        s_char = s_e_char_sents[sent_i][0]
                        e_char = s_e_char_sents[sent_i][1]
                        relavant_ac = [
                            ac
                            for ac in data["original_actions"][o_sent_iter]
                            if s_char <= int(ac.split(",")[-1].split(")")[0]) < e_char
                        ]
                        shifted_ac = add_to_char_ids_in_simple_char_ac_sequence(
                            ac_sequence=relavant_ac,
                            to_add=-s_char,
                        )
                        split_sents["original_actions"].append(shifted_ac)

            # ------------------------
            # Step 4: Add everything to the `splitted_data`, reindexing
            assert set(split_sents.keys()) == keys

            for k in split_sents:
                for sent_iter in range(numb_split_sents):
                    splitted_data[k][new_sent_iter + sent_iter] = split_sents[k][
                        sent_iter
                    ]
            new_sent_iter += numb_split_sents

    # Sanity check
    assert max([len(x) for x in splitted_data["token_ids"].values()]) <= max_tokens

    return {
        "splitted_data": splitted_data,
        "num_splitted_sents": num_splitted_sents,
        "numb_deleted_sents": numb_deleted_sents,
    }
