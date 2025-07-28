import numpy as np
from src.modules.utils.action_utils import add_to_char_ids_in_simple_char_ac_sequence
import copy


def select_cells(matrix: np.array, target_value: int) -> list:
    # Create a mask for allowed cells
    allowed = np.ones(matrix.shape, dtype=bool)

    # Disallow cells with -1 initially
    allowed[matrix == -1] = False

    # Store selected cells (row, col) and their values
    selected_cells = []

    # Iterate while there are valid cells
    while allowed.any():
        # Mask the matrix
        masked_matrix = np.where(allowed, matrix, np.inf)

        # Compute the absolute difference with the mean
        diff = np.abs(masked_matrix - target_value)

        # Find the index of the cell with the smallest difference
        row, col = np.unravel_index(np.argmin(diff), matrix.shape)

        # Record the selection
        # This means a sentence will be created by merging sentences with indexes in
        # range(row, col + 1)
        selected_cells.append([int(row), int(col)])

        # Disallow any merges that contain the selected sentences
        allowed[row : (col + 1), :] = False
        allowed[:, row : (col + 1)] = False

    # Sort
    selected_cells.sort(key=lambda x: x[0])

    return selected_cells


def normalize_sent_lengths(data, max_numb_tokens: int, mode: str):
    """Given a dict
    {dataset_name:
        {'doc_id': {sent_id: doc_id},
         'token_ids': {sent_id: [2, 12906, 4253, ...]}
        }
    },

    returns a dict {dts_name: {doc_id: list of tuples}} where each entry in the list
    of tuples represents a merged sentence, and is
    (sent_id_start, sent_id_end), where sent_id_end is suppose to be included.

    The sentences are merged according to `mode`:
        'global': try to make all sentences in all documents of the training set
                  be of similar size
        'local': try to make the sentences of each document of each training dataset
                 be of similar size

    Note: the sent_id's in the above output are relative to the whole dataset,
    as indexed in data[dts_name][key].
    """

    # For each (dataset, doc_id), we get a matrix whose entries represent
    # the number of tokens when merging sentences of the training subset
    merged_sents_matrix = {}
    sents_ids_per_doc = {}
    for dts_name in data:
        merged_sents_matrix[dts_name] = {}
        sents_ids_per_doc[dts_name] = {}

        # Get the document ids
        doc_ids = sorted(set(data[dts_name]["doc_id"].values()))

        # Create the matrix for each document
        # E.g. if a document has 8 sents with lengths [26, 32, 17, 45, 12, 38, 45, 30]
        # and `max_numb_tokens`=150, the matrix looks like
        # array([[ 26.,  58.,  75., 120., 132.,  -1.,  -1.,  -1.],
        #       [ -1.,  32.,  49.,  94., 106., 144.,  -1.,  -1.],
        #       [ -1.,  -1.,  17.,  62.,  74., 112.,  -1.,  -1.],
        #       [ -1.,  -1.,  -1.,  45.,  57.,  95., 140.,  -1.],
        #       [ -1.,  -1.,  -1.,  -1.,  12.,  50.,  95., 125.],
        #       [ -1.,  -1.,  -1.,  -1.,  -1.,  38.,  83., 113.],
        #       [ -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  45.,  75.],
        #       [ -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  -1.,  30.]])
        # The entry (row, col) indicates the number of tokens we would get by merging
        # sentences[row: col + 1]. The -1 indicate impossible merges, either because
        # row > col, or because the number of tokens would exceed the maximum allowed

        for doc_id in doc_ids:
            sent_ids_this_doc = [
                si for si, di in data[dts_name]["doc_id"].items() if di == doc_id
            ]
            sents_ids_per_doc[dts_name][doc_id] = sent_ids_this_doc
            num_toks_per_sent = [
                len(data[dts_name]["token_ids"][si]) for si in sent_ids_this_doc
            ]

            numb_sents_in_doc = len(num_toks_per_sent)

            matrix = np.zeros(shape=(numb_sents_in_doc, numb_sents_in_doc))
            for start_sent in range(numb_sents_in_doc):
                for end_sent in range(numb_sents_in_doc):
                    total_toks = sum(num_toks_per_sent[start_sent : end_sent + 1])

                    # The total number of tokens in the possible merge should include
                    # the CLS at the begining, a SEP at the end, and SEP's in between
                    # the sentences. Since num_toks_per_sent already includes a CLS and
                    # a SEP, we can transform one of these tokens, gaining an extra
                    # token per merged sentence (except the last one)
                    add_to_max = (end_sent - start_sent) * ((end_sent - start_sent) > 0)
                    if total_toks > (max_numb_tokens + add_to_max):
                        matrix[start_sent, end_sent] = -1
                    else:
                        matrix[start_sent, end_sent] = total_toks

            matrix[matrix == 0] = -1
            merged_sents_matrix[dts_name][doc_id] = matrix

    target_value = {
        dts_name: {doc_id: 0 for doc_id in merged_sents_matrix[dts_name]}
        for dts_name in merged_sents_matrix
    }

    # Get the number of tokens in the merged sentences, per (dts_name, doc_id)
    numb_toks_merged = {
        dts_name: {doc_id: 0 for doc_id in merged_sents_matrix[dts_name]}
        for dts_name in merged_sents_matrix
    }
    numb_sents_merged = {
        dts_name: {doc_id: 0 for doc_id in merged_sents_matrix[dts_name]}
        for dts_name in merged_sents_matrix
    }

    for dts_name in merged_sents_matrix:
        for doc_id in merged_sents_matrix[dts_name]:
            aux_mat = merged_sents_matrix[dts_name][doc_id]
            mask = aux_mat != -1
            numb_toks_merged[dts_name][doc_id] = int(aux_mat[mask].sum())
            numb_sents_merged[dts_name][doc_id] = int(mask.sum())

    # To get the target value, we depend on the `mode`
    if mode == "global":
        # Get a global average of the number of tokens for allowed merges,
        # across all doc_ids and training datasets
        all_toks = sum(
            [
                sum(
                    [
                        numb_toks_merged[dts_name][doc_id]
                        for doc_id in numb_toks_merged[dts_name]
                    ]
                )
                for dts_name in numb_toks_merged
            ]
        )

        all_sents = sum(
            [
                sum(
                    [
                        numb_sents_merged[dts_name][doc_id]
                        for doc_id in numb_sents_merged[dts_name]
                    ]
                )
                for dts_name in numb_sents_merged
            ]
        )

        # All sentences have the same target
        global_average = all_toks // all_sents
        for dts_name in merged_sents_matrix:
            for doc_id in merged_sents_matrix[dts_name]:
                target_value[dts_name][doc_id] = global_average

    elif mode == "local":
        # Get the target based on the (dts_name, doc_id)

        for dts_name in merged_sents_matrix:
            for doc_id in merged_sents_matrix[dts_name]:
                target_value[dts_name][doc_id] = (
                    numb_toks_merged[dts_name][doc_id]
                    // numb_sents_merged[dts_name][doc_id]
                )

    # Go through each (dts_name, doc_id) and get the sent_ids that need to be merged
    sent_ids_to_merge = {dts_name: {} for dts_name in merged_sents_matrix}

    for dts_name in merged_sents_matrix:
        for doc_id in merged_sents_matrix[dts_name]:
            matrix = merged_sents_matrix[dts_name][doc_id]

            sent_ids_to_merge[dts_name][doc_id] = select_cells(
                matrix=matrix, target_value=target_value[dts_name][doc_id]
            )

            # Add the id of the first sentence in doc, in order to have sentences
            # refer to the same id as the original `data`
            sent_id_at_start_of_doc = sents_ids_per_doc[dts_name][doc_id][0]

            for i in range(len(sent_ids_to_merge[dts_name][doc_id])):
                rel_start = sent_ids_to_merge[dts_name][doc_id][i][0]
                rel_end = sent_ids_to_merge[dts_name][doc_id][i][1]
                sent_ids_to_merge[dts_name][doc_id][i] = (
                    rel_start + sent_id_at_start_of_doc,
                    rel_end + sent_id_at_start_of_doc,
                )

    return sent_ids_to_merge


def reindex_dts_for_merge(
    data: dict,
    merge_ids: dict,
    sentence_sep_token: str,
    sentence_sep_token_id: int,
) -> dict:
    """Assuming:

    a dict `data` with format {dts_name: dts_data}, where dts_data is a dict with:
        -> mandatory
        'mode': 'simple' or 'simple-char',
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
        'start_end_char_idx_of_words': {sent_id: list of tuples},
        
        -> and, optionally,
        'original_actions': {sent_id: list of actions},
    a dict `merge_ids` with structure
        {dts_name:
              {doc_id: [((start_sent_id, end_sent_id), numb words in merged sent)]}
        }
    a string `sentence_sep_token`, and its corresponding id `sentence_sep_token_id`,

    returns the data after merging sentences according to `merge_ids`."""

    merged_data = {}
    keys_to_merge = [
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
    if all(["original_actions" in x for x in data.values()]):
        keys_to_merge.append("original_actions")

    add_to_raw_sents = f" {sentence_sep_token} "

    for dts_name in data:
        merged_data[dts_name] = {key: {} for key in keys_to_merge + ["doc_id"]}
        mode = data[dts_name]["mode"]
        merged_data[dts_name]["mode"] = mode

        new_sent_counter = 0

        for doc_id, m_data in merge_ids[dts_name].items():
            for m_sent in m_data:
                start_sent_id, end_sent_id = m_sent
                number_of_sentences_in_merge = end_sent_id - start_sent_id + 1

                # Save the data of the collection of sentences that will be merged
                # in a dict thats list-valued
                data_this_merge = {key: [] for key in keys_to_merge}
                for key in data_this_merge:
                    for sent_id in range(start_sent_id, end_sent_id + 1):
                        data_this_merge[key].append(data[dts_name][key][sent_id])

                # For the 'doc_id'
                merged_data[dts_name]["doc_id"][new_sent_counter] = doc_id

                # If there a single sentence, we just copy the data
                if number_of_sentences_in_merge == 1:
                    for key in keys_to_merge:
                        merged_data[dts_name][key][new_sent_counter] = data_this_merge[
                            key
                        ][0]

                else:
                    # Join 'words_by_space' with the `sentence_sep_token`
                    m_words = []
                    for sent_iter, words in enumerate(
                        data_this_merge["words_by_space"]
                    ):
                        m_words.extend(words)
                        if sent_iter != number_of_sentences_in_merge - 1:
                            m_words.extend([sentence_sep_token])
                    merged_data[dts_name]["words_by_space"][new_sent_counter] = m_words

                    # Join 'raw_sentences' by adding ' {sentence_sep_token} ' in between
                    m_raw = ""
                    for sent_iter, raw_s in enumerate(data_this_merge["raw_sentences"]):
                        m_raw += raw_s
                        if sent_iter != number_of_sentences_in_merge - 1:
                            m_raw += add_to_raw_sents
                    merged_data[dts_name]["raw_sentences"][new_sent_counter] = m_raw

                    # Join 'original_actions' depending on `mode`
                    if "original_actions" in keys_to_merge:
                        m_actions = []
                        if mode == "simple":
                            for sent_iter, ac in enumerate(
                                data_this_merge["original_actions"]
                            ):
                                m_actions.extend(ac)
                                if sent_iter != number_of_sentences_in_merge - 1:
                                    m_actions.extend(["OUT"])
                        elif mode == "simple-char":
                            lens_of_raw_sents = [
                                len(x) + len(add_to_raw_sents)
                                for x in data_this_merge["raw_sentences"]
                            ]
                            start_idx_of_raw_sents_in_merge = [0] + [
                                sum(lens_of_raw_sents[: i + 1])
                                for i in range(len(lens_of_raw_sents))
                            ][:-1]

                            for sent_iter, ac in enumerate(
                                data_this_merge["original_actions"]
                            ):
                                shifted_ac = add_to_char_ids_in_simple_char_ac_sequence(
                                    ac_sequence=ac,
                                    to_add=start_idx_of_raw_sents_in_merge[sent_iter],
                                )
                                m_actions.extend(shifted_ac)
                        else:
                            raise ValueError(
                                "Mode should be 'simple' or 'simple-char'."
                            )
                        merged_data[dts_name]["original_actions"][
                            new_sent_counter
                        ] = m_actions

                    # Join 'tokens' by changing the CLS token in between sentences
                    # to the sentence_sep_token, and removing the SEP at the end of all
                    # sents that are not the last
                    m_tokens = []
                    for sent_iter, tok_s in enumerate(data_this_merge["tokens"]):
                        tok_list = copy.copy(tok_s)
                        if sent_iter != number_of_sentences_in_merge - 1:
                            tok_list.pop()
                        if sent_iter > 0:
                            tok_list[0] = sentence_sep_token

                        m_tokens.extend(tok_list)
                    merged_data[dts_name]["tokens"][new_sent_counter] = m_tokens

                    # Join the 'tokens_ids' by the same procedure, but with ids
                    m_tokens_id = []
                    for sent_iter, tok_id_s in enumerate(data_this_merge["token_ids"]):
                        tok_s_list = copy.copy(tok_id_s)
                        if sent_iter != number_of_sentences_in_merge - 1:
                            tok_s_list.pop()
                        if sent_iter > 0:
                            tok_s_list[0] = sentence_sep_token_id

                        m_tokens_id.extend(tok_s_list)
                    merged_data[dts_name]["token_ids"][new_sent_counter] = m_tokens_id

                    # Join the 'offset_mapping' by considering the sentence_sep_token
                    # as a word
                    m_offset_mapping = []
                    for sent_iter, offmap_s in enumerate(
                        data_this_merge["offset_mapping"]
                    ):
                        os_list = copy.copy(offmap_s)
                        if sent_iter != number_of_sentences_in_merge - 1:
                            os_list.append([(0, len(sentence_sep_token))])

                        m_offset_mapping.extend(os_list)
                    merged_data[dts_name]["offset_mapping"][
                        new_sent_counter
                    ] = m_offset_mapping

                    # Join 'offset_mapping_sentence' using the same idea
                    # Also update char indices
                    m_offset_mapping_sentence = []
                    rsent_lens_with_sent_sep_after = [
                        (
                            len(data_this_merge["raw_sentences"][id])
                            + len(add_to_raw_sents)
                        )
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
                    merged_data[dts_name]["offset_mapping_sentence"][
                        new_sent_counter
                    ] = m_offset_mapping_sentence

                    # Update 'start_end_char_idx_of_words' using
                    # updated 'offset_mapping_sentence'
                    merged_data[dts_name]["start_end_char_idx_of_words"][
                        new_sent_counter
                    ] = [
                        (x[0][0], x[-1][-1])
                        for x in merged_data[dts_name]["offset_mapping_sentence"][
                            new_sent_counter
                        ]
                    ]

                    # Update wordpieces using updated 'start_end_char_idx_of_words'
                    merged_data[dts_name]["wordpieces"][new_sent_counter] = [
                        merged_data[dts_name]["raw_sentences"][new_sent_counter][s:e]
                        for (s, e) in merged_data[dts_name][
                            "start_end_char_idx_of_words"
                        ][new_sent_counter]
                    ]

                    # Join 'subwords_map', by joining the original ones with the
                    # 'offset_mapping_sentence', and adding for the separator
                    word_offset = 0
                    tok_offset = 0
                    m_subwords_map = {}
                    merged_data[dts_name]["subwords_map"][new_sent_counter] = {}
                    for sent_iter, swm in enumerate(data_this_merge["subwords_map"]):
                        for original_word_id, original_swm in swm.items():
                            m_subwords_map[original_word_id + word_offset] = [
                                x + tok_offset for x in original_swm
                            ]

                        # Add the separator
                        if sent_iter != number_of_sentences_in_merge - 1:
                            sep_word_id = max(m_subwords_map.keys()) + 1
                            sep_tok_id = (
                                max([xs for x in m_subwords_map.values() for xs in x])
                                + 1
                            )
                            m_subwords_map[sep_word_id] = [sep_tok_id]

                        last_word_id = max(m_subwords_map.keys())
                        last_tok_id = max(
                            [xs for x in m_subwords_map.values() for xs in x]
                        )
                        word_offset = last_word_id + 1
                        tok_offset = last_tok_id + 1
                    merged_data[dts_name]["subwords_map"][
                        new_sent_counter
                    ] = m_subwords_map

                    # Join the 'token_subword_type' by dropping the last element if
                    # the sentence is not the last one
                    m_token_subword_type = []
                    for sent_iter, t_list in enumerate(
                        data_this_merge["token_subword_type"]
                    ):
                        if sent_iter != number_of_sentences_in_merge - 1:
                            m_token_subword_type.extend(t_list[:-1])
                        else:
                            m_token_subword_type.extend(t_list)
                    merged_data[dts_name]["token_subword_type"][
                        new_sent_counter
                    ] = m_token_subword_type

                # Increment the new sentence counter
                new_sent_counter += 1

    return merged_data


def greedy_partitioning(numb_tok_in_sents: list[int], threshold: int):
    partitions = []
    current_partition = [0]
    current_sum = numb_tok_in_sents[0]

    # A merged sentence should have a CLS at the start, a SEP at the end,
    # and SEPs between inner sentences. Since numb_tok_in_sents already has these
    # tokens accounted for, when merging sentences we can drop one token
    add_to_threshold = 1

    for i in range(1, len(numb_tok_in_sents)):
        if current_sum + numb_tok_in_sents[i] <= (threshold + add_to_threshold):
            current_partition.append(i)
            current_sum += numb_tok_in_sents[i] - add_to_threshold
        else:
            partitions.append([current_partition[0], current_partition[-1]])
            current_partition = [i]
            current_sum = numb_tok_in_sents[i]

    partitions.append([current_partition[0], current_partition[-1]])

    return partitions


def partition_variance(partitions):
    sums = [sum(partition) for partition in partitions]
    return sum((s - (sum(sums) / len(sums))) ** 2 for s in sums) / len(sums)


def swap_partitions(partitions, num_tok, threshold):
    part_id = 0
    if len(partitions) == 1:
        pass
    else:
        for part_id in range(len(partitions)):
            if part_id == 0:
                neighbors = [1]
            elif part_id == len(partitions) - 1:
                neighbors = [len(partitions) - 2]
            else:
                neighbors = [part_id - 1, part_id + 1]

            # We pass elements of the current partition to the neighbors
            for nei in neighbors:

                while True:
                    new_part = copy.deepcopy(partitions)

                    if nei < part_id:
                        # Left neighbor, place at the end of neighbor
                        new_part[nei][1] += 1
                        new_part[part_id][0] += 1
                    elif nei > part_id:
                        # Right neighbor, place at the start of neighbor
                        new_part[nei][0] -= 1
                        new_part[part_id][1] -= 1

                    # Check if the part_id is not empty
                    if (new_part[part_id][1] - new_part[part_id][0]) < 0:
                        new_part.pop(part_id)

                    # Get partition with token numbers
                    part_toks = [num_tok[s : e + 1] for (s, e) in partitions]
                    new_part_toks = [num_tok[s : e + 1] for (s, e) in new_part]

                    # The total number of tokens in the possible merge should include
                    # the CLS at the begining, a SEP at the end, and SEP's in between
                    # the sentences. Since num_tok already has these tokens, we can
                    # change one token for each sentence in a merge, that is not the
                    # last
                    add_to_threshold = [len(w) - 1 for w in new_part_toks]

                    # Update if better
                    if (
                        all(
                            [
                                sum(subset) <= (threshold + add_to_threshold[i])
                                for i, subset in enumerate(new_part_toks)
                            ]
                        )
                    ) and (
                        partition_variance(new_part_toks)
                        < partition_variance(part_toks)
                    ):
                        partitions = new_part
                    else:
                        break

    return partitions


def maximize_sent_lens(data: dict, max_numb_tokens: int, swap: bool):
    # Get sentence lengths per dataset and document
    sents_ids_per_doc = {}
    numb_tokens = {}
    for dts_name in data:
        sents_ids_per_doc[dts_name] = {}
        numb_tokens[dts_name] = {}
        doc_ids = sorted(set(data[dts_name]["doc_id"].values()))
        for doc_id in doc_ids:
            sent_ids_this_doc = [
                si for si, di in data[dts_name]["doc_id"].items() if di == doc_id
            ]
            sents_ids_per_doc[dts_name][doc_id] = sent_ids_this_doc
            numb_tokens[dts_name][doc_id] = [
                len(data[dts_name]["token_ids"][si]) for si in sent_ids_this_doc
            ]
            if not all(x <= max_numb_tokens for x in numb_tokens[dts_name][doc_id]):
                ValueError(
                    f"Sentence exceeds `max_numb_tokens` in dts {dts_name}, doc {doc_id}"
                )

    sent_ids_to_merge = {}
    for dts_name in numb_tokens:
        sent_ids_to_merge[dts_name] = {}
        for doc_id in numb_tokens[dts_name]:

            # Initial greedy partitioning
            partitions = greedy_partitioning(
                numb_tok_in_sents=numb_tokens[dts_name][doc_id],
                threshold=max_numb_tokens,
            )

            if swap:
                # Iterative improvement: swap elements
                partitions = swap_partitions(
                    partitions=partitions,
                    num_tok=numb_tokens[dts_name][doc_id],
                    threshold=max_numb_tokens,
                )

            # Add the id of the first sentence in doc, in order to have sentences
            # refer to the same id as the original `data`
            sent_id_at_start_of_doc = sents_ids_per_doc[dts_name][doc_id][0]

            sent_ids_to_merge[dts_name][doc_id] = []
            for i in range(len(partitions)):
                rel_start = partitions[i][0]
                rel_end = partitions[i][1]
                sent_ids_to_merge[dts_name][doc_id].append(
                    (
                        rel_start + sent_id_at_start_of_doc,
                        rel_end + sent_id_at_start_of_doc,
                    )
                )

    return sent_ids_to_merge
