from transformers import AutoTokenizer
import os
import shutil
from collections import Counter
from typing import Union, List, Dict


def nest_list(flat_list, nested_structure):
    """
    Nest flat_list according to the structure of nested_structure.

    :param flat_list: The list that needs to be nested.
    :param nested_structure: The structure to follow when nesting.
    :return: A nested list that follows the structure of nested_structure.
    """
    iterator = iter(flat_list)

    def nest(structure):
        if isinstance(structure, list):
            return [nest(item) for item in structure]
        else:
            return next(iterator)

    return nest(nested_structure)


class Tokenizer:
    def __init__(self, model_name, cache_dir: str = ".transformers_models/"):
        model_cache_folder = os.path.join(cache_dir, model_name)
        try:
            # If cache folder already exists, just load locally
            tokenizer = AutoTokenizer.from_pretrained(
                model_cache_folder, local_files_only=True
            )
        except:
            # Otherwise, download the model hashed files into a temporary folder
            tmp_folder = os.path.join(cache_dir, model_name, "tmp")
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=tmp_folder)
            shutil.rmtree(tmp_folder)
            # save model files with proper formats and names to be reused later
            tokenizer.save_pretrained(model_cache_folder)

        print(f"Loaded pretrained tokenizer {model_name} with {len(tokenizer)} words")
        self.tokenizer = tokenizer

        # Find the tokenizer separator
        self._set_tokenizer_separator()

        # Define all special ids
        self._set_all_special_ids()

    def rebuild_words(self, tokens):
        words = []

        if self.tokenizer_separator == "▁":
            # For sentence piece type tokenizers
            for token in tokens:
                if token.startswith(self.tokenizer_separator):
                    # If the token starts with '_', create a new wordappend it to the last word in the list
                    words.append(token[len(self.tokenizer_separator) :])
                else:
                    # Otherwise, add the token to the previous word
                    words[-1] += token
        else:
            for token in tokens:
                if token.startswith(self.tokenizer_separator):
                    # If the token starts with '##', append it to the last word in the list
                    words[-1] += token[len(self.tokenizer_separator) :]
                else:
                    # Otherwise, add the token as a new word
                    words.append(token)

        return words

    def tokenize_sent(
        self, list_pretok_words_or_str: Union[str, List[str]], pre_tokenized: bool
    ) -> Dict:
        """Given `list_pretok_words_or_str` (string or pretokenized sentence presented
        as a list of strings) and the flag `pre_tokenized`, returns a dictionary with
        `tokens` (including special ones), `token_ids`, `offset_mapping`,
        `offset_mapping_sentence` and `token_subword_type`.

        Warning: assumes that the tokenizer used has [CLS] as first token,
        and [SEP] as last.

        E.g.

        """

        # If we already have the sentence pretokenized, we flatten the
        # list of 'input_ids' from the encoder

        enc = self.tokenizer(
            list_pretok_words_or_str,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )

        if pre_tokenized:
            # Tokenization
            input_ids = enc["input_ids"]
            token_ids = [self.tokenizer.cls_token_id]
            for t in input_ids:
                token_ids.extend(t[1:-1])
            token_ids.append(self.tokenizer.sep_token_id)
            offset_mapping = [off[1:-1] for off in enc["offset_mapping"]]
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

            # Offset mapping in char index from the start of the sentence
            offset_mapping_char = []
            offset_char = 0
            for word_idx, word in enumerate(list_pretok_words_or_str):
                offset_mapping_char.append(
                    [
                        (offset_char + tok_off[0], offset_char + tok_off[1])
                        for tok_off in offset_mapping[word_idx]
                    ]
                )
                offset_char += len(word) + 1

            # Map index of original_word -> list of indices of subwords
            subwords_map = {}
            offset = 0
            for i, om in enumerate(offset_mapping):
                subwords_map[i] = [offset + j for j in range(len(om))]
                offset += len(om)

            # Mask for tokens
            # 3: the token corresponds to a single word
            # 1: the token corresponds to the first sub-word
            # 2: the token corresponds to the last sub-word
            # 0: the token corresponds to one of the middle sub-words
            # -1: the token is a special token
            offset_start = 1  # Since we have [CLS] at the begining
            token_subword_type = [0] * len(token_ids)
            token_subword_type[0] = token_subword_type[-1] = -1
            for sw in subwords_map.values():
                if len(sw):
                    token_subword_type[offset_start + sw[0]] = 1
                    token_subword_type[offset_start + sw[-1]] |= 2
        else:
            token_ids_sent = enc["input_ids"]
            tokens_sent = self.tokenizer.convert_ids_to_tokens(token_ids_sent)
            rebuilt_words = self.rebuild_words(tokens=tokens_sent[1:-1])

            new_enc = self.tokenize_sent(rebuilt_words, pre_tokenized=True)

            offset_mapping_char = nest_list(
                enc["offset_mapping"][1:-1],
                new_enc["offset_mapping"],
            )
            offset_mapping = new_enc["offset_mapping"]

            tokens = new_enc["tokens"]
            token_ids = new_enc["token_ids"]
            subwords_map = new_enc["subwords_map"]
            token_subword_type = new_enc["token_subword_type"]

        return {
            "tokens": tokens,
            "token_ids": token_ids,
            "offset_mapping": offset_mapping,
            "offset_mapping_sentence": offset_mapping_char,
            "subwords_map": subwords_map,
            "token_subword_type": token_subword_type,
        }

    def _set_tokenizer_separator(self):
        tmp_sentence = "XYZ AAA BBB CCC DDD EEE FFF GGG HHH"
        tmp_enc = self.tokenize_sent(tmp_sentence.split(), pre_tokenized=True)["tokens"]
        unigram = Counter([x[:1] for x in tmp_enc]).most_common(1)
        bigram = Counter([x[:2] for x in tmp_enc]).most_common(1)
        trigram = Counter(x[:3] for x in tmp_enc).most_common(1)

        sep = ""
        cnt = -1
        for count in [unigram, bigram, trigram]:
            aux_sep, aux_cnt = count[0]
            # >= because the larger n-gram values have priority
            if aux_cnt >= cnt:
                sep = aux_sep
                cnt = aux_cnt

        self.tokenizer_separator = sep

    def _set_all_special_ids(self):
        self.all_special_ids = self.tokenizer.all_special_ids
