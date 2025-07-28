# Dataset preprocessing

First of all, create a folder named `data` inside the root directory of the repository. Inside it, create two additional folders, one named `original`, and another named `converted`.

Follow the steps below, guranteeing that you adhere to the save directories specified, in order to replicate our experiments with the provided Hydra configuration files in the repository.

## Single-task experiments

### CoNLL dataset

Download the original dataset by following the instructions in https://www.clips.uantwerpen.be/conll2003/ner/.

Place the `eng.train`, `eng.testa` and `eng.testb` files in a folder `data/original/CoNLL`.

Then, run the conversion script below:

    scripts/dataset_processing/conll/convert_conll.sh

The converted dataset will be stored in `data/converted/CoNLL`.

### GENIA dataset

We adapt the conversion script found here: https://github.com/yhcc/CNN_Nested_NER/tree/master/preprocess. Follow the steps below.

Download the GENIA dataset in xml format from https://gitlab.com/sutd_nlp/overlapping_mentions/-/blob/master/data/GENIA/scripts/GENIAcorpus3.02.merged.fixed.xml. Save the file in `data/original/GENIA`.

Download the splits used by Yan et. al (2022) from https://github.com/yhcc/CNN_Nested_NER/tree/master/preprocess/splits/genia. Save the folder as `data/original/GENIA/splits`.

Run the conversion script below:

    python scripts/dataset_processing/process_genia.py --xml_path data/original/GENIA/GENIAcorpus3.02.merged.fixed.xml --output_folder data/converted/GENIA --genia_split_folder data/original/GENIA/splits

The processed files will be stored at `data/converted/GENIA`.

## Multi-task experiment
First, download each of the folders below. Save each folder (keeping its original name) in `data/original/Crichton`.

|Dataset name  | Link |
|--|--|
|BC2GM   | https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/BC2GM-IOB |
| BC4CHEMD | https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/BC4CHEMD |
| BC5CDR | https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/BC5CDR-IOB |
|JNLPBA | https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/JNLPBA |
| Linnaeus | https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/linnaeus-IOB |
| NCBIDisease |  https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data/NCBI-disease-IOB |

Then, run the conversion script below:

    scripts/dataset_processing/multi_task_experiment/convert_datasets.sh


The converted datasets will be stored in `data/converted/multi-task`.


## Cross-corpus evaluation experiment

### Training corpora

The train/dev datasets come from https://github.com/flairnlp/flair. Our preprocessing file builds upon flair's framework. In order to get the processed files, first install additional dependencies by doing:

    pip install -r scripts/dataset_processing/cross_corpus_experiment/train_dev/flair_requirements.txt

Then, run the following:

    python scripts/dataset_processing/cross_corpus_experiment/train_dev/get_datasets_hunflair.py --output_dir data/converted/flair

The converted datasets will be stored at `data/converted/flair`.

*Note the following*: for the NLM Gene and GNormPlus corpora there is no predefined dev split in the flair framework. Therefore, a dev split is created by flair by randomly partitioning the train split. Hence, for these corpora, it is expected that the entity counts are different to the ones we report in the paper.

### Evaluation corpora

The evaluation datasets are extracted from https://github.com/hu-ner/hunflair2-experiments. In order to get the converted datasets, follow the steps below.

First, download the following files, and save them in `data/original/cross_corpus_eval`:

|Dataset name| Link |
|--|--|
| BioID  | https://github.com/hu-ner/hunflair2-experiments/blob/main/annotations/goldstandard/bioid.txt |
| MedMentions| https://github.com/hu-ner/hunflair2-experiments/blob/main/annotations/goldstandard/medmentions_ctd_only_mappable.txt |
|tmVar3 | https://github.com/hu-ner/hunflair2-experiments/blob/main/annotations/goldstandard/tmvar_v3.txt |

Then, run the following script:

    python scripts/dataset_processing/cross_corpus_experiment/test/get_eval_hunflair2.py --input scripts/dataset_processing/cross_corpus_experiment/test/paths.yaml --output data/converted/hunflair2_eval


## Evaluation of global predictions using synthetic ensemble

First, make sure you have followed the instructions in the [Multi-task experiment](#multi-task-experiment) section above, so that the original BC5CDR corpus is saved in `data/original/Crichton/BC5CDR-IOB`.

Then, run the script that creates the disjoint subsets:

    python scripts/synthetic_ensemble/create_splits.py --BC5_train_path data/original/Crichton/BC5CDR-IOB/train.tsv --BC5_dev_path data/original/Crichton/BC5CDR-IOB/devel.tsv --save_dir data/original/BC5_splits

Now, convert the datasets into simple format by running:

    python scripts/dataset_processing/BIO_to_simple.py --dts_paths_yaml scripts/synthetic_ensemble/paths.yaml --output_folder data/converted/BC5_splits


## Evaluation of out-of-domain predictions by human evaluation

This experiment uses the CoNLL dataset as processed in the [Single-task experiments](#conll-dataset) above, and the BC5CDR dataset as processed in the section [Multi-task experiment](#multi-task-experiment) above.
