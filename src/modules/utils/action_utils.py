from typing import OrderedDict, Dict, List, Tuple, Optional, NamedTuple, Union
import torch


def get_inner_action(ac: str):
    if ac.startswith("TRANSITION") or ac.startswith("REDUCE"):
        return ac.split("(")[1].split(")")[0]
    else:
        return None


def prepend_to_action_tag(text_to_add, ac_tag):
    inner_ac = get_inner_action(ac_tag)

    if inner_ac:
        if ac_tag.startswith("TRANSITION"):
            return f"TRANSITION({text_to_add}${inner_ac})"
        elif ac_tag.startswith("REDUCE"):
            return f"REDUCE({text_to_add}${inner_ac})"
    else:
        return ac_tag


class ActionsVocab:
    def __init__(self, id_to_tag: dict, use_out: bool):
        """Creates the action vocabulary.
        Given a dictionary `id_to_tag` mapping an integer to the name of a tag,
        builds the vocab.
        Allowed actions are EOA (index 0), SHIFT (index 1), OUT (index 2, if flagged),
        and one pair of TRANSITION(tag)/REDUCE(tag) per tag in `id_to_tag`.
        """

        # Check that the tag ids are consecutive 0-indexed integers
        assert sorted(id_to_tag.keys()) == [_ for _ in range(len(id_to_tag))]

        # Create w2i (word to index)
        self.w2i = {"EOA": 0, "SHIFT": 1}
        self.eoa_ix = self.w2i["EOA"]
        self.shift_ix = self.w2i["SHIFT"]

        if use_out:
            self.w2i["OUT"] = 2
            self.out_ix = self.w2i["OUT"]

        self.id_to_tag = id_to_tag
        self.tag_to_id = {v: k for (k, v) in self.id_to_tag.items()}

        # Create the w2i vocabulary {action_name: action_id}
        offset = len(self.w2i.keys())
        for tag_id in sorted(self.id_to_tag.keys()):
            tag_name = self.id_to_tag[tag_id]
            self.w2i[f"TRANSITION({tag_name})"] = offset
            self.w2i[f"REDUCE({tag_name})"] = offset + 1
            offset += 2

        # Create lists of the indices of transitions, and reduces
        self.tr_ixs = [
            id for ac_name, id in self.w2i.items() if ac_name.startswith("TRANSITION")
        ]
        self.re_ixs = [
            id for ac_name, id in self.w2i.items() if ac_name.startswith("REDUCE")
        ]

        # Associate to each tag, the index of its TR and of its RE
        self.tag_to_tr_re = {}
        for tag_name in self.tag_to_id:
            tag_tr = self.w2i[f"TRANSITION({tag_name})"]
            tag_re = self.w2i[f"REDUCE({tag_name})"]
            self.tag_to_tr_re[tag_name] = {"TR": tag_tr, "RE": tag_re}

        # Build opposite dictionary for vocabulary, i2w
        self.i2w = self._build_i2w()

        # Build map transition -> reduce
        self.tr_re_map = self._build_tr_re_map()

        # Number of actions
        self.max_v = len(self.w2i)

        print(f"Built action vocabulary ({self.max_v} actions)")
        print(self.i2w)
        print("Tags")
        print(self.tag_to_id)

    @classmethod
    def build_for_training(
        cls, id_to_tag: dict, use_out: bool, dataset_to_tag_names_dict: dict
    ):
        """Should be used when training/validating/testing the model.
        Adds aditional information on where the tags come from
        Arguments:

        `id_to_tag`: dict {tag_id: tag_name}
        `use_out`: whether to create an OUT action
        `dataset_to_tag_names_dict`: dict {name_dataset: set of tag names}
        """
        # Step 1: Build vocabulary
        actions_vocab = cls(id_to_tag=id_to_tag, use_out=use_out)

        # Step 2: Add aditional information for training/validation/testing
        # Step 2.1: Save the tag_ids per dataset
        actions_vocab.dataset_to_tag_ids = {
            dataset_name: sorted(
                actions_vocab.tag_to_id[tag]
                for tag in dataset_to_tag_names_dict[dataset_name]
            )
            for dataset_name in dataset_to_tag_names_dict
        }

        # Step 2.2: Record which tr/re are related to which dataset
        actions_vocab.tr_re_per_dataset = {
            dataset_name: [] for dataset_name in dataset_to_tag_names_dict
        }
        for dataset_name, tag_set in dataset_to_tag_names_dict.items():
            for tag in tag_set:
                actions_vocab.tr_re_per_dataset[dataset_name].append(
                    actions_vocab.w2i[f"TRANSITION({tag})"]
                )
                actions_vocab.tr_re_per_dataset[dataset_name].append(
                    actions_vocab.w2i[f"REDUCE({tag})"]
                )

        # Step 2.3: For training, a tensor informing which actions can be predicted
        # without incurring in loss
        actions_vocab.allowed_for_pred = {}
        for (
            dataset_name,
            tr_re_from_dataset,
        ) in actions_vocab.tr_re_per_dataset.items():
            actions_vocab.allowed_for_pred[dataset_name] = torch.ones(
                actions_vocab.max_v,
            )
            actions_vocab.allowed_for_pred[dataset_name][actions_vocab.eoa_ix] = 0
            actions_vocab.allowed_for_pred[dataset_name][actions_vocab.shift_ix] = 0
            if use_out:
                actions_vocab.allowed_for_pred[dataset_name][actions_vocab.out_ix] = 0
            actions_vocab.allowed_for_pred[dataset_name][tr_re_from_dataset] = 0

        print("Actions respective to each dataset")
        for dataset_name, allowed_ac in actions_vocab.tr_re_per_dataset.items():
            print(f"\t{dataset_name}\t{allowed_ac}")
        print("Tags respective to each dataset")
        for dataset_name, tag_id_set in actions_vocab.dataset_to_tag_ids.items():
            print(f"\t{dataset_name}\t{tag_id_set}")
        print()

        return actions_vocab

    @classmethod
    def build_for_prediction(cls, id_to_tag: dict, use_out: bool):
        """Given a dict `id_to_tag` mapping {tag_id: tag_name},
        builds the basic action vocabulary, dropping unnecessary information.
        This is to be used for prediction."""

        return cls(id_to_tag=id_to_tag, use_out=use_out)

    def _build_i2w(self):
        return {v: k for (k, v) in self.w2i.items()}

    def _build_tr_re_map(self):
        mp = {}

        for tag in self.tag_to_id:
            mp[self.w2i[f"TRANSITION({tag})"]] = self.w2i[f"REDUCE({tag})"]

        return mp

    def get_action_from_index(self, value):
        """Given an index, returns the action in string."""
        if value in self.i2w:
            return self.i2w[value]
        else:
            raise ValueError("The given index has no matching action.")

    def get_tag_from_ac_id(self, ac_id):
        if ac_id in self.tr_ixs or ac_id in self.re_ixs:
            return get_inner_action(self.i2w[ac_id])
        else:
            raise ValueError("There is no tag associated with this action.")

    def get_index_from_action(self, value):
        """Given an action in string form, returns the action index."""
        if value in self.w2i:
            return self.w2i[value]
        else:
            raise ValueError("The given action has no matching index.")

    def create_multilabel_action_tensor(self, singlelabel: list[str]):
        """Given a list `singlelabel` of size (ac_seq_len, ),
        each singlelabel[t] having the action at timestep t,
        this function returns a one-hot-like tensor (batch_size, reduced_seq_len)
        where several actions can take place at the same timestep.
        Details:
            - SHIFT/EOA cannot happen at the same time as any other action;
            - TR/RE can happen at the same time as long as they are different.
        """

        ac_seq_len = len(singlelabel)
        multilabel = torch.zeros((ac_seq_len, self.max_v), dtype=torch.float)

        num_ac = -1
        last_written_ac_was_sh = True
        for count in range(ac_seq_len):
            # If a SHIFT, we place it in the next timestep
            if singlelabel[count] == "SHIFT":
                num_ac += 1
                multilabel[num_ac, self.shift_ix] = 1
                last_written_ac_was_sh = True

            # If the EOA, we place it in the next timestep
            elif singlelabel[count] == "EOA":
                num_ac += 1
                multilabel[num_ac, self.eoa_ix] = 1

            # In the case of a TR or RE, we can place it in the same timestep
            # if the last action was a TR/RE of different index. Otherwise,
            # we move forward
            elif singlelabel[count].startswith("TRANSITION") or singlelabel[
                count
            ].startswith("REDUCE"):

                if (
                    multilabel[num_ac, self.w2i[singlelabel[count]]] == 1
                    or last_written_ac_was_sh
                ):
                    num_ac += 1

                multilabel[num_ac, self.w2i[singlelabel[count]]] = 1
                last_written_ac_was_sh = False
            else:
                raise ValueError("Could not create multilabel_action_tensor.")

        # Crop rows with only zeros
        multilabel = multilabel[multilabel.any(dim=-1)]

        return multilabel

    def decode_batch_action_tensor_predict(
        self, probs_tensor: torch.Tensor, raw_sents: List[str], offsets: List
    ):
        """Given a tensor (batch_size, seq_len, num_actions) with probabilities,
        returns the spans in the sentences."""

        highest_actions = probs_tensor.argmax(dim=-1)  # (batch_size, seq_len)
        batch_spans = []

        for sent_id, (ac_seq, high_acs, off, rs) in enumerate(
            zip(probs_tensor, highest_actions, offsets, raw_sents)
        ):
            word_count = 0
            # Spans which are open
            open_spans_per_tag = {tag_name: [] for tag_name in self.tag_to_tr_re}

            # Final list of spans
            spans = []

            for timestep_scores, high_ac in zip(ac_seq, high_acs):
                # If the action with the highest probability is the EOA we break
                if high_ac.item() == self.eoa_ix:
                    break
                # If it is the SHIFT or the OUT, we move forward
                elif (
                    hasattr(self, "out_ix") and high_ac.item() == self.out_ix
                ) or high_ac.item() == self.shift_ix:
                    word_count += 1

                # We handle TR/RE of each tag separately
                # We probe each action with probability above 0.5
                # We always RE first, and then TR
                else:
                    # Look at the RE's.
                    # We close a span if we have a TR of the same tag open
                    for ac in torch.argwhere(timestep_scores > 0.5):

                        ac_id = ac.item()

                        if ac_id in self.re_ixs:
                            corresp_tag = self.get_tag_from_ac_id(ac_id)

                            latest_span = None
                            if len(open_spans_per_tag[corresp_tag]) > 0:
                                latest_span = open_spans_per_tag[corresp_tag].pop()

                            if latest_span is not None:
                                if latest_span.start < word_count < len(off):
                                    latest_span.end = word_count

                                    start_char = off[latest_span.start][0][0]
                                    end_char = off[latest_span.end - 1][-1][-1]
                                    latest_span.text = rs[start_char:end_char]
                                    latest_span.score += timestep_scores[ac]

                                    if latest_span not in spans:
                                        spans.append(latest_span)

                    # Look at the TR's
                    # When we have a TR, we open a span
                    for ac in torch.argwhere(timestep_scores > 0.5):
                        ac_id = ac.item()
                        if ac_id in self.tr_ixs:
                            corresp_tag = self.get_tag_from_ac_id(ac_id)
                            span = LabeledSubstring(start=word_count, tag=corresp_tag)
                            span.score = timestep_scores[ac].unsqueeze(-1).item()
                            open_spans_per_tag[corresp_tag].append(span)

            batch_spans.append(sorted(spans, key=lambda x: (x.start, -x.end)))

        return batch_spans

    def decode_batch_action_tensor(
        self, probs_tensor: torch.Tensor, word_list: List, resolve_clash: bool = False
    ):
        """Given a tensor (batch_size, seq_len, num_actions) with probabilities,
        returns the spans in the sentences."""

        highest_actions = probs_tensor.argmax(dim=-1)  # (batch_size, seq_len)
        batch_spans = []

        for sent_id, (ac_seq, high_acs, words) in enumerate(
            zip(probs_tensor, highest_actions, word_list)
        ):
            word_count = 0
            # Spans which are open
            open_spans_per_tag = {tag_name: [] for tag_name in self.tag_to_tr_re}

            # Final list of spans
            spans = []

            for timestep_scores, high_ac in zip(ac_seq, high_acs):
                # If the action with the highest probability is the EOA we break
                if high_ac.item() == self.eoa_ix:
                    break
                # If it is the SHIFT or the OUT, we move forward
                elif (
                    hasattr(self, "out_ix") and high_ac.item() == self.out_ix
                ) or high_ac.item() == self.shift_ix:
                    word_count += 1

                # We handle TR/RE of each tag separately
                # We probe each action with probability above 0.5
                # We always RE first, and then TR
                else:
                    # Look at the RE's.
                    # We close a span if we have a TR of the same tag open
                    for ac in torch.argwhere(timestep_scores > 0.5):
                        ac_id = ac.item()

                        if ac_id in self.re_ixs:
                            corresp_tag = self.get_tag_from_ac_id(ac_id)

                            latest_span = None
                            if len(open_spans_per_tag[corresp_tag]) > 0:
                                latest_span = open_spans_per_tag[corresp_tag].pop()

                            if latest_span is not None:
                                if latest_span.start < word_count < len(words):
                                    latest_span.end = word_count
                                    latest_span.text = " ".join(
                                        words[latest_span.start : latest_span.end]
                                    )
                                    latest_span.score += timestep_scores[ac]

                                    if latest_span not in spans:
                                        spans.append(latest_span)

                    # Look at the TR's
                    # When we have a TR, we open a span
                    for ac in torch.argwhere(timestep_scores > 0.5):
                        ac_id = ac.item()
                        if ac_id in self.tr_ixs:
                            corresp_tag = self.get_tag_from_ac_id(ac_id)
                            span = LabeledSubstring(start=word_count, tag=corresp_tag)
                            span.score = timestep_scores[ac].unsqueeze(-1).item()
                            open_spans_per_tag[corresp_tag].append(span)

            if resolve_clash:
                spans = filter_clashed_by_priority(spans, allow_nested=True)

            batch_spans.append(sorted(spans, key=lambda x: (x.start, -x.end)))

        return batch_spans


class LabeledSubstring:
    """Class for modeling sentences or named entities"""

    def __init__(self, start=None, end=None, tag=None, text=None) -> None:
        self.start: int = start
        self.end: int = end
        self.text: Optional[Union[List[str], str]] = text
        self.tag: Optional[str] = tag

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, LabeledSubstring)
            and self.start == other.start
            and self.end == other.end
            and self.tag == other.tag
        )

    def __hash__(self) -> int:
        return hash((self.start, self.end, self.tag))

    def __str__(self) -> str:
        return f"'{self.text}'({self.start}, {self.end})//{self.tag}"

    def __repr__(self) -> str:
        return f"'{self.text}'({self.start}, {self.end})//{self.tag}"

    def same_spans(self, other) -> bool:
        return (self.start == other.start) and (self.end == other.end)

    def get_from_list_with_the_same_span(self, list_spans: list) -> list:
        """Returns a list of LabeledSubstring objects extracted from `list_spans`
        that have the same start + end index as `self`"""
        return [span for span in list_spans if span.same_spans(self)]

    def span_is_in_list(self, list_spans: list) -> bool:
        return len(self.get_from_list_with_the_same_span(list_spans)) > 0

    def prepend_to_tag(self, to_add) -> None:
        self.tag = to_add + "$" + self.tag

    def copy(self):
        return LabeledSubstring(
            start=self.start, end=self.end, tag=self.tag, text=self.text
        )

    def is_proper_subset(self, other) -> bool:
        return (other.start < self.start < self.end <= other.end) or (
            other.start <= self.start < self.end < other.end
        )

    def has_overlap(self, other) -> bool:
        return (
            len(
                set(range(self.start, self.end)).intersection(
                    set(range(other.start, other.end))
                )
            )
            > 0
        )


def is_overlapped(span1: LabeledSubstring, span2: LabeledSubstring):
    return span1.start < span2.end and span2.start < span1.end


def is_nested(span1: LabeledSubstring, span2: LabeledSubstring):

    return (span1.start <= span2.start and span2.end <= span1.end) or (
        span2.start <= span1.start and span1.end <= span2.end
    )


def is_clashed(
    chunk1: LabeledSubstring, chunk2: LabeledSubstring, allow_nested: bool = True
):
    if allow_nested:
        return is_overlapped(chunk1, chunk2) and not is_nested(chunk1, chunk2)
    else:
        return is_overlapped(chunk1, chunk2)


def filter_clashed_by_priority(spans, allow_nested: bool = True):
    spans = [ck for ck in sorted(spans, key=lambda x: x.score, reverse=True)]
    filtered_chunks = []
    for ck in spans:
        if all(
            not is_clashed(ck, ex_ck, allow_nested=allow_nested)
            for ex_ck in filtered_chunks
        ):
            filtered_chunks.append(ck)

    return filtered_chunks


def list_entities_char_idx_to_word_idx(entities_this_sent, s_e_token):
    """This is used to rewrite entities whose spans are given in character indexes to
    spans given in token indexes.

    Given a list `entities_this_sent` of LabeledSubtring, and a list `s_e_token` of
    tuples (ix_start_char, ix_end_char+1) that are the start/end characters of
    tokens in a sentence, returns two lists (`accepted`, `removed`),
    where each element in `accepted` is a LabeledSubstring with token-level indexing,
    and each element of `removed` is a LabeledSubstring with char-level indexing"""

    accepted = []
    removed = set()
    for ent in entities_this_sent:
        try:
            c = next(
                i for i, (start, end) in enumerate(s_e_token) if start == ent.start
            )
        except StopIteration:
            removed.add(ent.copy())
            continue

        try:
            d = next(i for i, (start, end) in enumerate(s_e_token) if end == ent.end)
        except StopIteration:
            removed.add(ent.copy())
            continue

        accepted.append(
            LabeledSubstring(start=c, end=d + 1, tag=ent.tag, text=ent.text)
        )
    return sorted(accepted, key=lambda x: (x.start, x.end)), sorted(
        removed, key=lambda x: (x.start, x.end)
    )


def list_entities_to_actions_encoding_with_OUT(words, labels):
    """
    Given:

    words = ['IL-2', 'gene', 'expression', 'and', 'NF-kappa', 'B', 'activation',
    'through', 'CD28', 'requires', 'reactive', 'oxygen', 'production', 'by',
    '5-lipoxygenase', '.']

    labels = 'IL-2 gene'(0, 2)//G#DNA, 'NF-kappa B'(4, 6)//G#protein,
    'CD28'(8, 9)//G#protein, '5-lipoxygenase'(14, 15)//G#protein]

    correspoonding to the labelling:
    'IL-2 gene' -> "G#DNA"
    'NF-kappa B' -> 'G#protein'
    'CD28' -> 'G#protein'
    '5-lipoxygenase' -> 'G#protein'

    this script returns the list of actions in the simple format with OUTs:

    ['TRANSITION(G#DNA)', 'SHIFT', 'SHIFT', 'REDUCE(G#DNA)', 'OUT', 'OUT',
    'TRANSITION(G#protein)', 'SHIFT', 'SHIFT', 'REDUCE(G#protein)', 'OUT', 'OUT',
    'TRANSITION(G#protein)', 'SHIFT', 'REDUCE(G#protein)', 'OUT', 'OUT', 'OUT',
    'OUT', 'OUT', 'TRANSITION(G#protein)', 'SHIFT', 'REDUCE(G#protein)', 'OUT']
    """

    sorted_labels = sorted(labels, key=lambda x: (x.end - x.start), reverse=True)
    transitions = list()

    # If a word has no label, then it's only action will be 'OUT'
    for i in range(len(words)):
        if any(i in range(label.start, label.end) for label in sorted_labels):
            transitions.append([])
        else:
            transitions.append(["OUT"])

    # We start of by writing the longest mentions to memory
    # This allows handling nesting without much worry
    for label in sorted_labels:
        transitions[label.start].append("TRANSITION(r)".replace("r", label.tag))

        if label.end == len(transitions):
            transitions.append(["REDUCE(r)".replace("r", label.tag)])
        else:
            transitions[label.end].insert(0, "REDUCE(r)".replace("r", label.tag))

    # At this point, if a subvector is empty, it should correspond to a SHIFT
    for i in range(len(transitions)):
        if len(transitions[i]) == 0:
            transitions[i].append("SHIFT")

        else:
            if (i != len(transitions) - 1) and ("OUT" not in transitions[i]):
                transitions[i].append("SHIFT")

    # We concatenate the 'transitions' vector into a string
    for i in range(len(transitions)):
        transitions[i] = " ".join(transitions[i])

    transitions = " ".join(transitions).split(" ")

    return transitions


def from_simple_char_ac_list_to_list_entities_with_char_idx(
    list_actions: List[str], raw_sentence: str
) -> List[str]:
    """Given a list of actions in simple-char format, returns a list
    of entities, each represented by a LabeledSubstring with char-level indexing.

    E.g. given

    list_actions = ['TRANSITION(Gene,0)', 'REDUCE(Gene,27)', 'TRANSITION(Disease,74)',
    'REDUCE(Disease,90)', 'TRANSITION(Gene,140)', 'REDUCE(Gene,147)']

    raw_sentence = 'Hepatocyte nuclear factor-6: associations between genetic
    variability and type II diabetes and between genetic variability and estimates of
    insulin secretion. '

    it outputs

    ['Hepatocyte nuclear factor-6'(0, 27)//Gene,
    'type II diabetes'(74, 90)//Disease, 'insulin'(140, 147)//Gene]

    Note: not to be used at inference; it is assumed that the
    action sequence is correct."""

    list_entities = []
    tr_list = []
    original_tr_order = []
    resulting_order = []
    tr_count = 0
    for ac in list_actions:
        ac_type = ac.split("(")[-1].split(",")[0]
        index = int(ac.split("(")[-1].split(",")[1].split(")")[0])

        if ac.startswith("TRANSITION"):
            tr_count += 1
            original_tr_order.append(tr_count)
            tr_list.append(LabeledSubstring(start=index, tag=ac_type))
        elif ac.startswith("REDUCE"):
            to_add = tr_list.pop()
            to_add.end = index
            to_add.text = raw_sentence[to_add.start : to_add.end]
            list_entities.append(to_add)
            order = original_tr_order.pop()
            resulting_order.append(order)

    list_entities = [item for _, item in sorted(zip(resulting_order, list_entities))]

    return list_entities


def from_simple_ac_list_to_list_entities_with_word_idx(
    list_actions: List[str], word_list: str
) -> List[str]:
    """Given a list of actions in simple format, returns a list
    of entities, each represented by a LabeledSubstring with word-level indexing.

    E.g. given

    list_actions = ['TRANSITION(G#DNA)', 'SHIFT', 'SHIFT', 'REDUCE(G#DNA)', 'OUT',
    'OUT', 'TRANSITION(G#protein)', 'SHIFT', 'SHIFT', 'REDUCE(G#protein)', 'OUT',
    'OUT', 'TRANSITION(G#protein)', 'SHIFT', 'REDUCE(G#protein)', 'OUT',
    'OUT', 'OUT', 'OUT', 'OUT', 'TRANSITION(G#protein)', 'SHIFT', 'REDUCE(G#protein)',
    'OUT']

    word_list = ['IL-2', 'gene', 'expression', 'and', 'NF-kappa', 'B', 'activation',
    'through', 'CD28', 'requires', 'reactive', 'oxygen', 'production', 'by',
    '5-lipoxygenase', '.']

    it outputs

    ['IL-2 gene'(0, 2)//G#DNA, 'NF-kappa B'(4, 6)//G#protein,
    'CD28'(8, 9)//G#protein, '5-lipoxygenase'(14, 15)//G#protein]

    Note: not to be used at inference; it is assumed that the action
    sequence is correct."""

    sent_idx = 0
    transitions = []
    transitions_stack = []
    spans = []

    for ac in list_actions:
        if ac == "SHIFT" or ac == "OUT":
            sent_idx += 1

        elif ac.startswith("TRANSITION"):
            transitions.append(ac)
            transitions_stack.append(sent_idx)

        elif ac.startswith("REDUCE") and len(transitions):
            tr = transitions.pop()
            tr_idx = transitions_stack.pop()
            red_label = get_inner_action(ac)
            tr_label = get_inner_action(tr)

            if red_label != tr_label:
                raise ValueError(
                    """ Wrong action sequence. This function should not be used
                    at inference time """
                )

            span = LabeledSubstring(
                start=tr_idx,
                end=sent_idx,
                text=" ".join(word_list[tr_idx:sent_idx]),
                tag=tr_label,
            )

            if span not in spans:
                spans.append(span)

    return spans


def add_to_char_ids_in_simple_char_ac_sequence(ac_sequence: list, to_add: int):
    """Adds a fixed integer to the char ids in a simple-char sequence. Example:

    Given a list `ac_sequence`
    ['TRANSITION(Gene,59)', 'REDUCE(Gene,64)', 'TRANSITION(Disease,100)',
    'REDUCE(Disease,149)']

    and `to_add` = 100

    returns
    ['TRANSITION(Gene,159)', 'REDUCE(Gene,164)', 'TRANSITION(Disease,200)',
     'REDUCE(Disease,249)']
    """

    position_comma = [ac.index(",") for ac in ac_sequence]
    position_right_parenthesis = [ac.index(")") for ac in ac_sequence]

    new_ac_seq = []
    for ac, pos_c, pos_p in zip(
        ac_sequence, position_comma, position_right_parenthesis
    ):
        number = int(ac[pos_c + 1 : pos_p])
        new_number = number + to_add
        new_ac_seq.append(ac[:pos_c] + "," + str(new_number) + ")")

    return new_ac_seq


# def from_tr_re_to_simple_char_ac_seq(transitions, reduces):
#     """Given two sorted lists of tuples like (start/end index, tag), returns
#     a list of actions in simple-char format."""

#     i, j = 0, 0
#     ac_list = []
#     while i < len(transitions) and j < len(reduces):
#         if transitions[i][0] < reduces[j][0]:
#             ac_list.append(f"TRANSITION({transitions[i][1]},{transitions[i][0]})")
#             i += 1
#         else:
#             ac_list.append(f"REDUCE({reduces[j][1]},{reduces[j][0]})")
#             j += 1
#     while i < len(transitions):
#         ac_list.append(f"TRANSITION({transitions[i][1]},{transitions[i][0]})")
#         i += 1
#     while j < len(reduces):
#         ac_list.append(f"REDUCE({reduces[j][1]},{reduces[j][0]})")
#         j += 1

#     return ac_list


def from_ent_list_to_simple_char_ac_seq(ent_list: List[LabeledSubstring]):
    ent_list_sorted = sorted(ent_list, key=lambda x: (x.start, -x.end, x.tag))
    timesteps = sorted(set([xs for x in ent_list_sorted for xs in [x.start, x.end]]))

    ac_simple = []

    for t in timesteps:
        start_ent_idx = [
            ent_idx for ent_idx, ent in enumerate(ent_list_sorted) if ent.start == t
        ]
        end_ent_idx = [
            ent_idx for ent_idx, ent in enumerate(ent_list_sorted) if ent.end == t
        ]
        if len(end_ent_idx):
            aux = [
                f"REDUCE({ent_list_sorted[id].tag},{ent_list_sorted[id].end})"
                for id in reversed(end_ent_idx)
            ]
            ac_simple.extend(aux)

        if len(start_ent_idx):
            aux = [
                f"TRANSITION({ent_list_sorted[id].tag},{ent_list_sorted[id].start})"
                for id in start_ent_idx
            ]
            ac_simple.extend(aux)

    return ac_simple
