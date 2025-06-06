# Should We Fine-Tune or RAG? Evaluating Different Techniques to Adapt LLMs for Dialogue
Repository containing the code for the paper *[Should We Fine-Tune or RAG? Evaluating Different Techniques to Adapt LLMs for Dialogue](https://arxiv.org/abs/2406.06399)* accepted at INLG2024.

- [Should We Fine-Tune or RAG? Evaluating Different Techniques to Adapt LLMs for Dialogue](#should-we-fine-tune-or-rag-evaluating-different-techniques-to-adapt-llms-for-dialogue)
  - [Introduction](#introduction)
  - [Usage](#usage)
    - [Installation](#installation)
    - [Data](#data)
      - [DailyDialog - Open Domain Dialogues (ODD)](#dailydialog---open-domain-dialogues-odd)
      - [WizardOfWikipedia - Knowledge Grounded Dialogues (KGD)](#wizardofwikipedia---knowledge-grounded-dialogues-kgd)
      - [DSTC9 (Track 1) - Task-Oriented Dialogues (TOD)](#dstc9-track-1---task-oriented-dialogues-tod)
      - [NarrativeQA - Question Answering (QA)](#narrativeqa---question-answering-qa)
    - [Download Fine-Tuned Models and Results](#download-fine-tuned-models-and-results)
    - [Run the code](#run-the-code)
      - [fine-tune](#fine-tune)
      - [prompting](#prompting)
      - [generate](#generate)
      - [int-grad](#int-grad)
    - [Analysis](#analysis)
    - [Human Evalution Protocol](#human-evalution-protocol)
  - [License](#license)
  - [How To Cite](#how-to-cite)


## Introduction
We study the limitations of Large Language Models (LLMs) for the task of response generation in human-machine dialogue. Several techniques have been proposed in the literature for different dialogue types (e.g., Open-Domain). However, the evaluations of these techniques have been limited in terms of base LLMs, dialogue types and evaluation metrics. In this work, we extensively analyze different LLM adaptation techniques when applied to different dialogue types. We have selected two base LLMs, Llama-2 and Mistral, and four dialogue types Open-Domain, Knowledge-Grounded, Task-Oriented, and Question Answering. We evaluate the performance of in-context learning and fine-tuning techniques across datasets selected for each dialogue type. We assess the impact of incorporating external knowledge to ground the generation in both scenarios of Retrieval-Augmented Generation (RAG) and gold knowledge. We adopt consistent evaluation and explainability criteria for automatic metrics and human evaluation protocols. Our analysis shows that there is no universal best-technique for adapting large language models as the efficacy of each technique depends on both the base LLM and the specific type of dialogue. Last but not least, the assessment of the best adaptation technique should include human evaluation to avoid false expectations and outcomes derived from automatic metrics.

## Usage

### Installation
Use the following command to clone the repository and the submodules.

```bash
git clone --recurse-submodules https://github.com/sislab-unitn/Fine-Tune-or-Rag.git
```

The original conda environment is provided for reproducibility.
```bash
conda env create -f environment.yaml
```

If you prefer to create the environment yourself, use python 3.10.14 and then run:
```bash
pip install -r requirements.txt
```

> [!IMPORTANT]  
> ParlAI is required to compute the intrinsic evaluation for some of the proposed tasks. However, because of its incompatibility with other Python packages, the user needs to perform the installation through pip.


### Data
The following set of commands will download and process the data to train/evaluate a model.


#### DailyDialog - Open Domain Dialogues (ODD)
To download DailyDialog to train/evaluate the model on the ODD setting, run the following commands:

```bash
pushd original_data
wget http://yanran.li/files/ijcnlp_dailydialog.zip
unzip ijcnlp_dailydialog.zip
rm ijcnlp_dailydialog.zip
pushd ijcnlp_dailydialog
unzip train.zip
unzip validation.zip
unzip test.zip
popd
popd
python ./utils/daily_dialogue_prepare.py
```

#### WizardOfWikipedia - Knowledge Grounded Dialogues (KGD)
To download WizardOfWikipedia (WoW) to train/evaluate the model on the KGD setting, run the following commands:

```bash
pushd original_data
wget http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz
mkdir -p WizardOfWikipedia
tar -xzvf wizard_of_wikipedia.tgz -C WizardOfWikipedia
popd
python ./utils/wizard_of_wikipedia_prepare.py
```


#### DSTC9 (Track 1) - Task-Oriented Dialogues (TOD)
To train/evaluate the model on the TOD setting first you have to download MultiWOZ 2.1 dataset using the following commands:

```bash
pushd original_data
wget "https://github.com/budzianowski/multiwoz/raw/master/data/MultiWOZ_2.1.zip"
unzip MultiWOZ_2.1.zip
rm MultiWOZ_2.1.zip
rm -rf __MACOSX
popd
```

Then, you can download DSTC9 (Track 1) dataset and preprocess the data:
```bash
mkdir -p original_data/DSTC9
cp -r alexa_with_dstc9_track1_dataset/data* original_data/DSTC9/
python ./utils/dstc9_prepare.py
```


#### NarrativeQA - Question Answering (QA)
To download NarrativeQA to train/evaluate the model on the QA setting, run the following commands:
```bash
pushd original_data
git clone https://github.com/google-deepmind/narrativeqa.git
mkdir -p NarrativeQA
mv narrativeqa/qaps.csv NarrativeQA
mv narrativeqa/documents.csv NarrativeQA
mv narrativeqa/third_party/wikipedia/summaries.csv NarrativeQA
rm -rf narrativeqa
popd
python ./utils/narrative_qa_prepare.py
```

### Download Fine-Tuned Models and Results
You can access the fine-tuned models and the results obtained by downloading the additional resources attached to the [release](https://github.com/sislab-unitn/Fine-Tune-or-Rag/releases/tag/v1.0).

You can place the `*.tar.xz` in the `output` folder and unzip them using the following command:

```bash
cat *.tar.gz | tar -xzvf - -i
```

Each folder contains the results for both Llama-2 Chat and Mistral Instruct 7B. For each model, you have access to a set of files and folders containing the weights (`best_model`), the perplexity, the optimization results, and the responses generated using:

- no additional knowldge (`none`)
- gold knowledge (`gold`)
- retrieved knowledge (`top-1` and `top-3`)

Below, the stucture for the ODD setting:
```bash
├── llama # folder containing the weights/results for Llama-2 Chat 7B
│
...
│
└── mistral # folder containing the weights/results for Mistral Instruct 7B
    ├── none_0 # fine-tuning results with no external knowledge (seed 0)
    ├── none_1 # fine-tuning results with no external knowledge (seed 1)
    ├── none_2 # fine-tuning results with no external knowledge (seed 2)
    │   ├── best_model # folder containing the weights of the fine-tuned model
    │   ├── fine_tune.txt # fine-tuning results (PPL)
    │   ├── optimization_results_none.json # optimization results with no external knowledge (Fine-Tuning)
    │   ├── training_params.json
    │   └── train_stats.json
    ├── optimization_results_none.json # optimization results with no external knowledge (In-Context Learning)
    ├── prompting_dev_Instr1_none.txt # In-Context Learning optimization (Instruction 1)
    ├── prompting_dev_Instr2_none.txt # In-Context Learning optimization (Instruction 2)
    ├── prompting_dev_NoInstr_none.txt # In-Context Learning optimization (no Instruction)
    └── prompting_NoInstr_none.txt # In-Context Learning results
```

> [!IMPORTANT]  
> 1. Optimization always refers to the validation set. 
> 2. Other folders may contain the results for gold and retrieved (top-1 and top-3) knowledge


### Run the code
`main.py` is the main module. For all the executions, you'll have to run the command specifying the model, the data, and the subparser, i.e.

```bash
python -m main MODEL_NAME DATA_FOLDER SUBPARSER
```
where:
- MODEL_NAME can be llama or mistral
- DATA_FOLDER is one of the sub-folder contained in data/ (e.g. data/DailyDialog)
- SUBPARSER is one of the following submodules

> [!IMPORTANT]  
> 1. You can use the `--unstr-know` argument to specify whether to use additional knowledge (i.e. `none`, `gold`, `retrieved`) 
> 2. When selecting `--unstr-know retrieved`, you can use `--top-k` to specify the number of retrieved documents to include in the input.

Use `--help` to get additional insights on the optional parameters.


#### fine-tune
This subparser fine-tunes the specified model. For example, to fine-tune llama on DailyDialog you can run the following command:

```bash
python -m main llama data/DailyDialog fine-tune your_experiment
```

Use `--help` to get additional insights on the optional parameters.


#### prompting
This subparser selects the optimal instruction from a fixed list (e.g. `data/DailyDialog/prompts.json`) and then (based on the validation results) prompts the specified model. For example, to prompt llama on DailyDialog, you can run the following command:

```bash
python -m main llama data/DailyDialog prompting
```

Use `--help` to get additional insights on the optional parameters.


#### generate
This subparser generates (using batch generation) the responses for the specified model. A MODEL_KEY specifying the set of hyper-parameters (e.g. top-p, top-k, ...) for the generation is required. You can find for each dialogue setting a `generation.json` file (e.g. `data/DailyDialog/generation.json`) containing the best set of hyper-parameters for each model (llama and mistral), technique (fine-tuning and in-context lerning), and knowledge (no, gold, top-1, and top-3).

For example, to generate the responses for DailyDialog using the in-context version of llama, you can run the following command:

```bash
python -m main llama data/DailyDialog generate llama_prompt_none
```

Instead, to generate the responses for DailyDialog using the fine-tuned version of llama (seed 0), you can run the following command:

```bash
python -m main llama data/DailyDialog generate llama_ft_none_0 --experiment-path output/DailyDialog/llama/none_0
```

> [!IMPORTANT]  
> Use the `--experiment-path` to point to the fine-tuned version (i.e. weights) of the model. If not specified, the model downloaded from HuggingFace will be instantiated.

Use `--help` to get additional insights on the optional parameters.


#### int-grad
This subparser returns the integrated gradient results for the specified model and set of answers.

For example, to compute the integrated gradients for DailyDialog using llama, you can run the following command:

```bash
python -m main llama data/DailyDialog int-grad output/DailyDialog/llama/generation_results_none.json
```

> [!IMPORTANT]  
> 1. Use the `--experiment-path` to point to the fine-tuned version (i.e. weights) of the model. If not specified, the model downloaded from HuggingFace will be instantiated (See [generate](#generate) for an example).
> 2. At least **two A100 GPUs with 80GB** are required to run integrated gradients.

Use `--help` to get additional insights on the optional parameters.


### Analysis
Please take a look at the jupyter notebooks available in the [stats](./stats/) folder to perform analysis on the data/results.


### Human Evalution Protocol
We extended the Human Evalution Protocol proposed by Mousavi et al. (2022) in the paper *[Evaluation of Response Generation Models: Shouldn’t It Be Shareable and Replicable?](https://aclanthology.org/2022.gem-1.12/)*

The extended version of the protocol (v1.1) is available [here](https://github.com/sislab-unitn/Human-Evaluation-Protocol/tree/v1.1).


## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This work is licensed under a [MIT License](https://opensource.org/licenses/MIT).


## How To Cite
```
@inproceedings{alghisi-etal-2024-fine-tune,
    title = "Should We Fine-Tune or RAG? Evaluating Different Techniques to Adapt LLMs for Dialogue",
    author = "Alghisi, Simone  and
      Rizzoli, Massimo  and
      Roccabruna, Gabriel  and
      Mousavi, Seyed Mahed  and
      Riccardi, Giuseppe",
    editor = "Mahamood, Saad  and
      Minh, Nguyen Le  and
      Ippolito, Daphne",
    booktitle = "Proceedings of the 17th International Natural Language Generation Conference",
    month = sep,
    year = "2024",
    address = "Tokyo, Japan",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.inlg-main.15/",
    pages = "180--197",
    abstract = "We study the limitations of Large Language Models (LLMs) for the task of response generation in human-machine dialogue. Several techniques have been proposed in the literature for different dialogue types (e.g., Open-Domain). However, the evaluations of these techniques have been limited in terms of base LLMs, dialogue types and evaluation metrics. In this work, we extensively analyze different LLM adaptation techniques when applied to different dialogue types. We have selected two base LLMs, Llama-2 and Mistral, and four dialogue types Open-Domain, Knowledge-Grounded, Task-Oriented, and Question Answering. We evaluate the performance of in-context learning and fine-tuning techniques across datasets selected for each dialogue type. We assess the impact of incorporating external knowledge to ground the generation in both scenarios of Retrieval-Augmented Generation (RAG) and gold knowledge. We adopt consistent evaluation and explainability criteria for automatic metrics and human evaluation protocols. Our analysis shows that there is no universal best-technique for adapting large language models as the efficacy of each technique depends on both the base LLM and the specific type of dialogue. Last but not least, the assessment of the best adaptation technique should include human evaluation to avoid false expectations and outcomes derived from automatic metrics."
}
```
