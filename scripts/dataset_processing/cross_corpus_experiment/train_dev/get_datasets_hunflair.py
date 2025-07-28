import autorootcwd
import os
from collections import Counter
from torch.utils.data.dataset import Subset
from flair.datasets.biomedical import (
    HUNER_ALL_BIORED,
    HUNER_GENE_NLM_GENE,
    HUNER_GENE_GNORMPLUS,
    HUNER_CHEMICAL_SCAI,
    HUNER_DISEASE_SCAI,
    HUNER_CHEMICAL_NLM_CHEM,
    HUNER_SPECIES_LINNEAUS,
    HUNER_SPECIES_S800,
    HUNER_DISEASE_NCBI,
)
from src.modules.utils.action_utils import (
    from_ent_list_to_simple_char_ac_seq,
    LabeledSubstring,
)
from src.modules.utils.read_local_datasets import (
    check_and_extract_action_sequence_from_file,
)
import argparse


""" Based on https://github.com/flairNLP/flair/blob/master/resources/docs/HUNFLAIR2_TUTORIAL_3_TRAINING_NER.md """

def extract_data(sent_dataset):
    struct_data = []
    for sentence in sent_dataset:
        sent = LabeledSubstring(
            start=sentence.start_position, end=sentence.end_position, text=sentence.text
        )

        all_mentions = []
        for label in sentence.labels:
            start = label.data_point.start_position
            end = label.data_point.end_position
            tag = label.data_point.tag
            text = label.data_point.text
            all_mentions.append(
                LabeledSubstring(start=start, end=end, tag=tag, text=text)
            )

        struct_data.append({"text": sent, "ents": all_mentions})

    return struct_data


def main(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    corpora = (
        HUNER_ALL_BIORED(),
        HUNER_GENE_NLM_GENE(),  # (random_seed=RANDOM_SEED),
        HUNER_GENE_GNORMPLUS(),  # (random_seed=RANDOM_SEED),
        HUNER_CHEMICAL_SCAI(),
        HUNER_DISEASE_SCAI(),
        HUNER_CHEMICAL_NLM_CHEM(),
        HUNER_SPECIES_LINNEAUS(),
        HUNER_SPECIES_S800(),
        HUNER_DISEASE_NCBI(),
    )

    for dataset in corpora:
        name_dataset = os.path.split(dataset.name)[-1]
        path_this_dataset = os.path.join(output_dir, name_dataset)
        os.makedirs(path_this_dataset, exist_ok=True)

        print(f"#### DATASET {name_dataset} ####")

        for split_name, split in zip(
            ["train", "dev", "test"], [dataset.train, dataset.dev, dataset.test]
        ):

            if isinstance(split, Subset):
                sent_dataset = [split.dataset[ix] for ix in split.indices]
            else:
                sent_dataset = split.datasets[0].sentences

            data = extract_data(sent_dataset)
            output_file = (
                os.path.join(path_this_dataset, name_dataset) + f".{split_name}"
            )

            num_sents = 0
            num_entities = Counter()

            with open(output_file, "w") as f:
                for sent_data in data:
                    text = sent_data["text"]

                    if not len(text.text):
                        continue

                    ents = sent_data["ents"]

                    ents_this_sent = [x.tag for x in ents]
                    num_entities += Counter(ents_this_sent)
                    num_sents += 1

                    ac_seq_list = from_ent_list_to_simple_char_ac_seq(ents)

                    if (
                        check_and_extract_action_sequence_from_file(
                            ac_seq_list, text.text.split(), "simple-char"
                        )
                        == False
                    ):
                        raise ValueError()

                    # Sanity check of spans
                    assert all([text.text[e.start : e.end] == e.text for e in ents])

                    full_data = text.text + " ||| " + " ".join(ac_seq_list)
                    f.write(full_data + "\n")

            print("-")
            print(f"Wrote file: {output_file}")
            print(f"Num_sents: {num_sents}")
            print(f"Entity counts {num_entities}\n\n")

    print("-" * 20)
    print("DONE!")
    print("-" * 20)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Converter from Flair datasets.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output folder where the converted datasets will be stored.",
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir)