  

# SRU-NER
Source code for the paper **Effective Multi-Task Learning for Biomedical Named Entity Recognition** ([arXiv link](https://arxiv.org/abs/2507.18542)), by João Ruano, Gonçalo M. Correia, Leonor Barreiros and Afonso Mendes. Presented at the  [24th BioNLP workshop](https://aclweb.org/aclwiki/BioNLP_Workshop), co-located with **ACL 2025**.

If you wish to use the code, please read the attached  **LICENSE.md** and cite our paper:

    @inproceedings{ruano-etal-2025-effective,
    title = "Effective Multi-Task Learning for Biomedical Named Entity Recognition",
    author = "Ruano, Jo{\~a}o  and
      Correia, Gon{\c{c}}alo  and
      Barreiros, Leonor  and
      Mendes, Afonso",
    booktitle = "Proceedings of the 24th Workshop on Biomedical Language Processing",
    year = "2025",
    address = "Viena, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.bionlp-1.20/",
    ISBN = "979-8-89176-275-6"}
  

## Installation

- Python version: 3.9.21

- CUDA version 12.2

- Install PyTorch with CUDA with: `pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121`

- Install torch-scatter with: `pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu121.html`

- Install other requirements with: `pip install -r requirements.txt`

## Training and testing models
The training and evaluation scripts use CUDA backends, so make sure that in your terminal session you specify:

    export CUDA_VISIBLE_DEVICES={x}

replacing `{x}` with the ID of the GPU you want to use. The code is prepared to use a single GPU.

Experiment configs are organized hierarchically using **Hydra** (https://hydra.cc/). Hydra is used to compose several YAML files that configure parts of the pipeline in order to build a unified YAML.
The main config file used for training is `configs/train.yaml`. 
The `configs/experiment` folder contains a file for each experiment to perform. Each entry in these experiment files overwrites the default values.

### Training a model
**First**, define the output directory. This is the place where everything about a model is saved (checkpoints, test results, etc.). In the file `configs/train.yaml`, change the entry:

    hydra:
	    run:
		    dir: logs/${task_name}/${run_name}

by replacing `logs` with the desired directory path. Keep the `${task_name}/${run_name}` string as is.

**Secondly**, create an experiment config file in the `configs/experiment` folder. There are three headers you *need to overwrite*. Make sure you:
- Overwrite the `seed` parameter.
- Overwrite the `task_name` parameter. This needs to match the name of the experiment file.
- Overwrite the `datasets` parameter.

Then, you can modify any other default parameter. For example, an experiment config for training a single-task model on the CoNLL dataset could be like this:

    # @package _global_     # NOTE: leave this as is

	# Mandatory overwrites
    seed: 42
    task_name: CoNLL        # NOTE: name of the experiment
    datasets:
	    CONLLen:    # NOTE: name of the dataset
		    source: local
		    encoding: utf-8
		    mode: simple    # NOTE: action encoding mode: simple or simple-char
		    split_paths:    # NOTE: paths to the datasets
			    train: data/converted/CoNLL/train.txt
			    dev: data/converted/CoNLL/dev.txt
			    test: data/converted/CoNLL/test.txt
		    tag_dict:   # NOTE: option to rename or remove entity types
			    LOC: Location
			    MISC: null
			    ORG: Organization
			    PER: People
    
    # Optional overwrites
    models:
	    embed_model:
		    pretrained_name: xlm-roberta-large

The last block overwrites a part of the model (the pretrained embedding model). Parameters which are not overwritten are set to their default value.
Note how datasets are specified. For each dataset that we train with, there should be a block like the one above for the CoNLL dataset. We accept two formats ('modes') for a corpus: *simple* or *simple-char*. For example, given the sentence *"Barack Obama was a president ."*, the sentence could be encoded in:

 - *Simple* mode, as `Barack Obama was a president . ||| TRANSITION(PER) SHIFT SHIFT REDUCE(PER) OUT OUT OUT OUT`
 - *Simple-char* mode, as `Barack Obama was a president . ||| TRANSITION(PER,0) REDUCE(PER,12)`

The *simple* mode assumes a pre-tokenization, whereas the *simple-char* mode uses character level indexing for mention specification. The `tag_dict` dictionary is used to rename the entity types in the original datasets. In this case, the model would be trained to predict the entity types `Location, Organization, People` (the `MISC` mentions are removed due to the value `null`).
Check out the section [Replicating our experiments](#replicating-our-experiments) for more example configs.

If you want to use MLFlow to track experiment metrics, change the default configuration file `configs/MLFlowLogger/defaults.yaml`by setting `use: True` and filling in the `tracking_uri` parameter.

**Finally**, run the model as follows. Inside the root folder of the project, do:

    python src/train.py experiment={experiment_name}

by replacing `{experiment_name}` with the name of the experiment file you created as above. 

During training:

-  **checkpoints** are saved in the folder `{directory_path}/{task_name}/{run_name}/checkpoints`

- the **final composed config** yaml is saved in `{directory_path}/{task_name}/{run_name}/resolved_cfg.yaml`

*Note*: the subfolder structure inside `{directory_path}/{task_name}` is slighly different if you don't enable MLFlow tracking.

### Testing a model
The file `src/test.py` takes in the final composed config saved after training and performs a **validation on the test sets** of the datasets used to train the model.
Run it by:

    python src/test.py --cfg_path {path_to_the_final_composed_config}

By default, the `src/train.py` script tests the model when training is finished, by calling the test script and using the best performant checkpoint on the validation data. 

## Replicating our experiments
### Dataset processing
Follow the instructions in [this guide](scripts/dataset_processing/README.md#dataset-preprocessing) in order to download and preprocess the datasets as we did for our experiments. Make sure you adhere to path specifications so that the experiment configs referenced below work correctly.

### Single-task experiments
For the CoNLL experiment, run:

    python src/train.py experiment=benchmarks/CoNLL

For the GENIA experiment, run:

    python src/train.py experiment=benchmarks/GENIA

### Multi-task experiment
To train the multi-task model on the six datasets that we used, do:

    python src/train.py experiment=multi-task/MT

To train single-task models on each of the datasets of the multi-task ensemble, do:

    python src/train.py experiment=multi-task/{exp}

with `{exp}` replaced by each of the experiment names who find in the folder `configs/experiment/multi-task`, starting in 'ST'.

### Cross-corpus evaluation experiment
First, train a multi-task model on the ensemble of nine datasets by doing:

    python src/train.py experiment=cross-corpus

Then, evaluate that model in the independent test ensemble by doing

    python scripts/cross_corpus_evaluation/predict_on_cross_corpus.py --model_cfg_path {path_to_the_resolved_config} --inference_corpus_path scripts/cross_corpus_evaluation/inference_data_paths.yaml

### Evaluation of global predictions on synthetic ensemble
Train the two single-task models by running:

    python src/train.py experiment=synthetic/sChemical

and

    python src/train.py experiment=synthetic/sDisease

Now, train the multi-task model by running:

    python src/train.py experiment=synthetic/sBoth

The test step that is automatically called in the above scripts will purposely fail. So, for testing the models on the full test split of the BC5CDR corpus, run the following script for each of the three models:

    python scripts/cross_corpus_evaluation/predict_on_cross_corpus.py --model_cfg_path {path_to_the_resolved_config} --inference_corpus_path scripts/synthetic_ensemble/inference_data_paths.yaml

replacing `{path_to_the_resolved_config}` with the corresponding path for each model.

### Evaluation of out-of-domain predictions by human assessment
**ST model trained on BC5CDR and evaluated on CoNLL**
The expert model trained on biomedical entity types was the single-task model trained on the BC5CDR corpus of section "Multi-task experiment" above.
Run the following script to retrieve the predictions of this model on the CoNLL corpus:

    python scripts/cross_corpus_evaluation/predict_on_cross_corpus.py --model_cfg_path {path_to_BC5_model_resolved_config} --inference_corpus_path scripts/out_of_domain_evaluation/conll_paths.yaml --pickle_save_file scripts/out_of_domain_evaluation/results_BC5model_on_CoNLL.pkl

Ignore the metrics displayed on the console. The predictions are saved in pickle format in `scripts/out_of_domain_evaluation/results_BC5model_on_CoNLL.pkl`.

**ST model trained on CoNLL and evaluated on BC5CDR**
In order to have an expert model trained on general domain entity types that can be compared with the multi-task model, we trained a model on the CoNLL dataset by using the BioLink encoder. To train this model run the following:

    python src/train.py experiment=human_eval/CoNLLBioLink

Then, run the following script to retrieve the predictions of this model on the BC5CDR corpus:

    python scripts/cross_corpus_evaluation/predict_on_cross_corpus.py --model_cfg_path {path_to_CoNLL_model_resolved_config} --inference_corpus_path scripts/out_of_domain_evaluation/bc5_paths.yaml --pickle_save_file scripts/out_of_domain_evaluation/results_CoNLLmodel_on_BC5.pkl

Again, ignore the metrics displayed on the console. The predictions are saved in pickle format in `scripts/out_of_domain_evaluation/results_CoNLLmodel_on_BC5.pkl`.

**MT model**

Train the multi-task model by running:

    python src/train.py experiment=human_eval/humanMT

The predictions of the model can be found in a pickle named `test_results.pkl` inside the parent directory of the checkpoints directory of the model.

**Evaluation of the spans**

The pickle files returned by the above scripts contain the predictions of the models in a Pandas DataFrame, for all entity types.
The script below filters the frames and outputs a txt file that contains solely the predictions of out-of-domain entity types.

    python scripts/out_of_domain_evaluation/predict_OOD.py --tag_specification_yaml {yaml_path} --prediction_pickle {pickle_path} --output_path {output_txt_path}

Run the script above by replacing the argument values in curly brackets, for each of the three models. For the `tag_specification_yaml`, use the files:

 - For the model trained on BC5 and evaluated on the CoNLL: `scripts/out_of_domain_evaluation/tag_spec_biomed_model.yaml`
 - For the model trained on CoNLL and evaluated on the BC5: `scripts/out_of_domain_evaluation/tag_spec_general_model.yaml`
 - For the multi-task model: `scripts/out_of_domain_evaluation/tag_spec_MT_model.yaml`

The `prediction_pickle` argument should be the paths of the pickles defined in the last subsections.

The evaluation of the spans in the output txt files was conducted by two human annotators. You can find more details in the paper.
