import autorootcwd
import os
import argparse
import yaml
from src.modules.utils.action_utils import (
    LabeledSubstring,
    list_entities_to_actions_encoding_with_OUT,
)
from collections import Counter


def parse_BIO(dataset_path):
    doc_id = -1
    sent_id = 0
    data = {}
    with open(dataset_path, "r", encoding="utf-8") as f:
        word_counter = 0
        words = []
        ents = []
        current_ent = None
        for line in f:
            line = line.strip()

            if len(line) == 0:
                if len(words):
                    data.setdefault(doc_id, [])
                    data[doc_id].append({"words": words, "ents": ents})
                    sent_id += 1

                word_counter = 0
                words = []
                ents = []
                current_ent = None
            elif line.startswith("-DOCSTART-"):
                doc_id += 1
            else:
                separated = line.split("\t")
                word = separated[0]
                label = separated[1]

                if label == "O":
                    if current_ent is not None:
                        assert [c[0] == current_ent[0][0] for c in current_ent]
                        tag = current_ent[0][0]
                        start = current_ent[0][-1]
                        end = current_ent[-1][-1] + 1
                        words_in_ent = " ".join(c[1] for c in current_ent)

                        ents.append(
                            LabeledSubstring(
                                start=start, end=end, tag=tag, text=words_in_ent
                            )
                        )
                        current_ent = None

                    words.append(word)
                    word_counter += 1
                elif label.startswith("B"):
                    tag = label.split("B-")[-1]
                    current_ent = [(tag, word, word_counter)]
                    words.append(word)
                    word_counter += 1
                elif label.startswith("I"):
                    tag = label.split("I-")[-1]
                    if current_ent is None:
                        raise ValueError
                    else:
                        current_ent.append((tag, word, word_counter))

                    words.append(word)
                    word_counter += 1

    if len(data) == 1:
        data[0] = data.pop(-1)

    # processed = {}
    # for doc_id in data:
    #     processed[doc_id] = []
    #     for sent in data[doc_id]:
    #         processed[doc_id].append(
    #             {
    #                 "words": sent["words"],
    #                 "ac_seq_list":
    #                     list_entities_to_actions_encoding_with_OUT(
    #                         sent["words"], sent["ents"]
    #                     ),
    #             }
    #         )

    return data


def main(dts_paths_yaml, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    with open(dts_paths_yaml) as f:
        data_info = yaml.safe_load(f)

    for dts_name, dts_data in data_info.items():
        print()
        print(f'----- WORKING ON DTS {dts_name} -----')
        dts_dir = os.path.join(output_folder, dts_name)
        os.makedirs(dts_dir, exist_ok=True)

        for split_name, split_path in dts_data.items():
            data = parse_BIO(split_path)

            num_docs = len(data)
            num_words = 0
            num_sents = 0
            ent_counter = Counter()

            processed_split_path = os.path.join(dts_dir, f"{split_name}.txt")

            with open(processed_split_path, "w", encoding="utf-8") as f:
                if num_docs == 1:
                    for sent in data[0]:
                        words = sent["words"]
                        num_words += len(words)
                        num_sents += 1
                        ents = sent["ents"]
                        ent_c = Counter(e.tag for e in ents)
                        ent_counter += ent_c
                        ac_seq = " ".join(
                            list_entities_to_actions_encoding_with_OUT(words, ents)
                        )
                        f.write(" ".join(words) + " ||| " + ac_seq + "\n")
                else:
                    for doc_id in data:
                        f.write("-- doc-sep --\n")
                        for sent in data[doc_id]:
                            words = sent["words"]
                            num_words += len(words)
                            num_sents += 1
                            ents = sent["ents"]
                            ent_c = Counter(e.tag for e in ents)
                            ent_counter += ent_c
                            ac_seq = " ".join(
                                list_entities_to_actions_encoding_with_OUT(words, ents)
                            )
                            f.write(" ".join(words) + " ||| " + ac_seq + "\n")

            print(f"Wrote file {processed_split_path}")
            print(f"Num docs {num_docs}")
            print(f"Num sents {num_sents}")
            print(f"Num words {num_words}")
            print(f"Ent count {dict(ent_counter)}")
            print()
    print()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Converter from BIO to simple.")
    parser.add_argument(
        "--dts_paths_yaml",
        type=str,
        required=True,
        help="Path to yaml with dataset information.",
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path of output folder",
    )
    args = parser.parse_args()

    main(args.dts_paths_yaml, args.output_folder)
