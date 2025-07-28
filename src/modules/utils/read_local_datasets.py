from unidecode import unidecode
from typing import Dict, List


def ingest_only_text_from_file(
    file_path: str,
    lower_case: bool = False,
    normalize: bool = False,
    encoding: str = "utf-8",
    ignore_ac: bool = False,
) -> Dict:

    words = {}
    raw_sentences = {}
    doc_ids = {}

    # Used to index in the dictionaries and to print the number of parsed examples
    sent_ix = 0

    # Counter of the number of empty lines
    nr_empty_lines = 0

    # Counter of documents
    doc_ix = -1
    sentence_ids_new_doc = []

    for line in open(file_path, "r", encoding=encoding):
        line = line.strip()

        # The corpus might have empty lines
        if not line:
            nr_empty_lines += 1
            continue

        # Use doc-sep to separate by documents
        if "doc-sep" in line:
            sentence_ids_new_doc.append(sent_ix)
            doc_ix += 1
            continue

        if ignore_ac:
            line = line.split("|||")[0].strip()

        # Record the raw sentence and words tokenized by space
        words[sent_ix] = line.split()
        raw_sentences[sent_ix] = line.strip()

        # Record the document id
        doc_ids[sent_ix] = doc_ix

        if lower_case:
            words[sent_ix] = [w.lower() for w in words[sent_ix]]

        if normalize:
            words[sent_ix] = [unidecode(w) for w in words[sent_ix]]

        sent_ix += 1

    if nr_empty_lines:
        print(f"Empty lines in {file_path}:", nr_empty_lines)

    # If there are no document separators, we consider all corpus as one big document
    if len(sentence_ids_new_doc) == 0:
        sentence_ids_new_doc = [0]
        for sent_ix in doc_ids.keys():
            doc_ids[sent_ix] = 0

    return {
        "words_by_space": words,
        "raw_sentences": raw_sentences,
        "doc_id": doc_ids,
    }


def ingest_dataset_from_file(
    file_path: str,
    mode: str,
    lower_case: bool = False,
    normalize: bool = False,
    encoding: str = "utf-8",
) -> Dict:

    words = {}
    actions = {}
    raw_sentences = {}
    doc_ids = {}

    # Used to index in the dictionaries and to print the number of parsed examples
    sent_ix = 0

    # Counter of the number of empty lines
    nr_empty_lines = 0

    # Counter of documents
    doc_ix = -1
    sentence_ids_new_doc = []

    for line in open(file_path, "r", encoding=encoding):
        line = line.strip()

        # The corpus might have empty lines
        if not line:
            nr_empty_lines += 1
            continue

        # Use doc-sep to separate by documents
        if "doc-sep" in line:
            sentence_ids_new_doc.append(sent_ix)
            doc_ix += 1
            continue

        words_str, actions_str = line.split("|||")
        # Record the raw sentence and words tokenized by space
        words[sent_ix] = words_str.split()
        raw_sentences[sent_ix] = words_str.strip()
        # Get the action sequence. We also check if the sequence is valid.
        ac_seq = check_and_extract_action_sequence_from_file(
            ac_list=actions_str.split(), words=words[sent_ix], mode=mode
        )
        if ac_seq == False:
            raise ValueError(line)
        else:
            actions[sent_ix] = ac_seq
        # Record the document id
        doc_ids[sent_ix] = doc_ix

        if lower_case:
            words[sent_ix] = [w.lower() for w in words[sent_ix]]

        if normalize:
            words[sent_ix] = [unidecode(w) for w in words[sent_ix]]

        sent_ix += 1

    if nr_empty_lines:
        print(f"Empty lines in {file_path}:", nr_empty_lines)

    # If there are no document separators, we consider all corpus as one big document
    if len(sentence_ids_new_doc) == 0:
        sentence_ids_new_doc = [0]
        for sent_ix in doc_ids.keys():
            doc_ids[sent_ix] = 0

    return {
        "words_by_space": words,
        "raw_sentences": raw_sentences,
        "original_actions": actions,
        "doc_id": doc_ids,
    }


def check_and_extract_action_sequence_from_file(
    ac_list: List[str], words: List[str], mode: str
) -> List[str]:
    # STEP 0: Check the action sequence is valid
    # 0.1: Assert that the number of transitions and reduces match
    nr_tr = sum([1 for x in ac_list if "TRANSITION" in x])
    nr_rd = sum([1 for x in ac_list if "REDUCE" in x])
    if nr_tr != nr_rd:
        print("ERROR: Number of transitions and reduces must match")
        return False

    # 0.2: mode specific checks
    if mode == "simple":
        # Assert that the number of words match the number of OUT+SHIFT
        nr_outs = ac_list.count("OUT")
        nr_shifts = ac_list.count("SHIFT")
        if nr_outs + nr_shifts != len(words):
            print(
                "ERROR (simple mode): Number of OUT+SHIFT doesn't match number of words"
            )
            return False
        # Assert that there are no transitions followed by reduces without shifts in
        # the middle
        flag = False
        for ix, action in enumerate(ac_list):
            if ix > 0 and "REDUCE" in action:
                if not (ac_list[ix - 1] == "SHIFT" or "REDUCE" in ac_list[ix - 1]):
                    flag = True
                    break
        if flag:
            print(
                "ERROR (simple mode): REDUCE can only be preceeded by SHIFT or REDUCE"
            )
            return False

    elif mode == "simple-char":
        # Assert that the char ids are in non-decreasing order
        char_ids = [int(ac[ac.index(",") + 1 : ac.index(")")]) for ac in ac_list]
        if not all(char_ids[i] <= char_ids[i + 1] for i in range(len(char_ids) - 1)):
            print("ERROR (simple-char mode): wrong character information")
        # Assert that the TR/RE match
        simple_ac_seq = [
            (
                (ac[ac.index("(") + 1 : ac.index(",")], "T")
                if "TRANSITION" in ac
                else (ac[ac.index("(") + 1 : ac.index(",")], "R")
            )
            for ac in ac_list
        ]
        tr_list = []
        flag = False
        for ac in simple_ac_seq:
            tag = ac[0]
            type = ac[1]
            if type == "T":
                tr_list.append(tag)
            elif type == "R":
                if len(tr_list) == 0:
                    flag = True
                    break
                else:
                    latest_tr_tag = tr_list.pop()
                    if tag != latest_tr_tag:
                        flag = True
                        break
        if flag:
            print("ERROR (simple-char mode): TR/RE pairs are incorrect")
            return False

    # STEP 1: Get the action sequence
    return ac_list
