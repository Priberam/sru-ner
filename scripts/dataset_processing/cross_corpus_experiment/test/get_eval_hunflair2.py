import autorootcwd
import argparse
from bioc import pubtator
from typing import Dict, List, Tuple
from omegaconf import OmegaConf
from tqdm import tqdm
import os
from collections import Counter
from src.modules.utils.action_utils import (
    LabeledSubstring,
    from_ent_list_to_simple_char_ac_seq,
)
import spacy

ALLOW_NESTING = True

# ------------------------------------------------------------------------
# Taken from https://github.com/hu-ner/hunflair2-experiments/blob/main/predict_hunflair2.py#L39 and
# https://github.com/hu-ner/hunflair2-experiments/blob/main/evaluate.py#L19


def load_documents(path: str) -> List[pubtator.PubTator]:
    documents = []
    with open(path) as fp:
        for d in pubtator.load(fp):
            documents.append(d)
    return documents


def split_on_newlines(text: str):
    start = 0
    sents = []
    for char_idx, char in enumerate(text):
        if char == "\n":
            end = char_idx
            sents.append(LabeledSubstring(start=start, end=end, text=text[start:end]))
            start = end + 1
        elif char_idx == len(text) - 1:
            end = len(text)
            sents.append(LabeledSubstring(start=start, end=end, text=text[start:end]))

    return sents


def split_into_sents(text: str, splitter):

    document = splitter(text)
    sents = []

    for doc in document.sents:
        sents.append(
            LabeledSubstring(start=doc.start_char, end=doc.end_char, text=doc.text)
        )

    return sents

# ------------------------------------------------------------------------

MAX_DOC = 0

def main(paths, output_folder_path):
    os.makedirs(str(output_folder_path), exist_ok=True)
    splitter = spacy.load("en_core_sci_sm")

    for dataset_name, dts_data in paths.items():
        print("#" * 20)
        print("Working on dataset:", dataset_name)

        dataset = load_documents(path=dts_data["file"])
        numb_ents = Counter([a.type for doc in dataset for a in doc.annotations])
        print("Numb ents in original dataset")
        print(numb_ents)
        doc_and_anot = {}

        doc_iter = 0
        for doc in tqdm(dataset, desc="Annotating"):
            text = doc.text
            sents = []
            
            # anots = [
            #     LabeledSubstring(start=a.start, end=a.end, text=a.text, tag=a.type)
            #     for a in doc.annotations
            # ]

            # Filter annots based on entity_types in `paths`
            anots = []
            for a in doc.annotations:
                if a.type not in dts_data["entity_types"]:
                    raise ValueError("incomplete entity_types")
                elif dts_data["entity_types"][a.type] is None:
                    continue
                else:
                    new_type = dts_data["entity_types"][a.type]
                    mention = LabeledSubstring(
                        start=a.start, end=a.end, text=a.text, tag=new_type
                    )
                    anots.append(mention)

            split_nl = split_on_newlines(text=text)

            for sent in split_nl:
                split_sents = split_into_sents(sent.text, splitter=splitter)
                start_char = sent.start
                for s in split_sents:
                    sents.append(
                        LabeledSubstring(
                            start=s.start + start_char,
                            end=s.end + start_char,
                            text=s.text,
                        )
                    )

            # Sanity check
            assert all(s.text == text[s.start : s.end] for s in sents)

            this_doc = []

            # Aggregate sents and annots
            for s in sents:
                annot_in_sent = [
                    a for a in anots if s.start <= a.start < a.end <= s.end
                ]

                # Correct indices
                annot_in_sent = [
                    LabeledSubstring(
                        start=a.start - s.start,
                        end=a.end - s.start,
                        text=a.text,
                        tag=a.tag,
                    )
                    for a in annot_in_sent
                ]

                # Sanity check
                assert all(a.text == s.text[a.start : a.end] for a in annot_in_sent)

                if len(s.text):
                    this_doc.append({"text": s.text, "ents": annot_in_sent})

            doc_and_anot[doc_iter] = this_doc
            doc_iter += 1
            if bool(MAX_DOC) and doc_iter > MAX_DOC:
                break

        # Print statistics
        nr_docs = len(doc_and_anot)
        nr_sents = sum(len(x) for x in doc_and_anot.values())
        nr_ents = Counter()
        for doc in doc_and_anot.values():
            for sent in doc:
                nr_ents += Counter([x.tag for x in sent["ents"]])
        print("Nr docs", nr_docs)
        print("Nr sents", nr_sents)
        print("Nr ents in converted dataset", nr_ents)

        # Transform into our format
        with open(os.path.join(output_folder_path, dataset_name + ".txt"), "w") as f:
            for doc in doc_and_anot.values():
                f.write("-- doc-sep --\n")
                for sent in doc:
                    text = sent["text"]
                    ac_seq = from_ent_list_to_simple_char_ac_seq(ent_list=sent["ents"])
                    line = text + " ||| " + " ".join(ac_seq) + "\n"
                    f.write(line)

    print("#" * 20)
    print("DONE !")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Transform Hunflair2 evaluation datasets from PubTator to simple-char format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Yaml of paths to text and annotations",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Folder of output files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    paths = OmegaConf.load(args.input)
    main(paths, args.output)
