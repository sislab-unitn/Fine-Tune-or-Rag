import argparse
import json
import os
import random
from itertools import product
from pathlib import Path


import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from peft.peft_model import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


from utils import daily_dialogue, dstc9, narrative_qa, wizard_of_wikipedia
from utils.utils import generate, MAP_MODELS


def configure_subparsers(subparsers: argparse._SubParsersAction):
    """Configure a new subparser ."""
    parser = subparsers.add_parser(
        "optimize",
        help="Optimize the generation of a model on a specific dev set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--batch-size",
        metavar="BATCH_SIZE",
        type=int,
        default=4,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--top-p",
        metavar="TOP_P",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 0.99],
        help="Top-p values for generation.",
    )
    parser.add_argument(
        "--temperature",
        metavar="TEMPERATURE",
        type=float,
        nargs="+",
        default=[0.5, 0.7, 0.9],
        help="Temperature values for generation.",
    )
    parser.add_argument(
        "--top-k-gen",
        metavar="TOP_K_GEN",
        type=int,
        nargs="+",
        default=[5, 20, 50],
        help="Top-k values for generation.",
    )
    parser.add_argument(
        "--experiment-path",
        metavar="EXPERIMENT_PATH",
        type=str,
        default=None,
        help="Path to the experiment folder of the fine-tuned model (without best_model).",
    )
    parser.add_argument(
        "--prompt",
        metavar="PROMPT",
        type=str,
        default="NoInstr",
        choices=["NoInstr", "Instr1", "Instr2"],
        help="Prompt to use for the generation.",
    )

    parser.set_defaults(func=main)


def main(args):
    dataset_name = Path(args.data_folder).name
    if dataset_name == "DailyDialog":
        BaselineDataset = daily_dialogue.BaselineDataset
        Collator = daily_dialogue.GenerationCollator
    elif dataset_name == "DSTC9":
        BaselineDataset = dstc9.OptimizationDataset
        if args.unstr_know == "none":
            Collator = dstc9.NoKnowledgeGenerationCollator
        elif args.unstr_know == "gold":
            Collator = dstc9.UnstrKnowGenerationCollator
        elif args.unstr_know == "retrieved":
            Collator = dstc9.RetrievedUnstrGenerationKnowCollator
        else:
            raise NotImplementedError(f"Unsupported setting: '{args.unstr_know}'")
    elif dataset_name == "NarrativeQA":
        BaselineDataset = narrative_qa.GenerationDataset
        Collator = narrative_qa.GenerationCollator
    elif dataset_name == "WizardOfWikipedia":
        BaselineDataset = wizard_of_wikipedia.KGDDataset
        Collator = wizard_of_wikipedia.GenerationCollator
    else:
        raise NotImplementedError(f"Unsupported dataset: '{dataset_name}'")

    output_folder = os.path.join(args.out_dir, Path(args.data_folder).name, args.model)

    try:
        with open(os.path.join(args.data_folder, "generation.json"), "r") as f:
            generation_params = json.load(f)["optimization"]
    except FileNotFoundError:
        raise FileNotFoundError(
            "The 'generation.json' file is missing. Please create the file in the specified data folder and provide the 'max_length' parameter."
        )

    try:
        with open(os.path.join(args.data_folder, "prompts.json"), "r") as f:
            system_prompt = json.load(f)[args.prompt]
    except FileNotFoundError:
        raise FileNotFoundError(
            "The 'prompts.json' file is missing. Please create the file in the specified data folder with keys 'NoInstr', 'Instr1', and 'Instr2'."
        )

    model_name = MAP_MODELS[args.model]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if args.parallel else args.device,
        torch_dtype=torch.bfloat16,
    )

    if args.experiment_path is not None:
        output_folder = os.path.join(output_folder, Path(args.experiment_path).name)
        model = PeftModel.from_pretrained(
            model, os.path.join(args.experiment_path, "best_model"), is_trainable=False  # type: ignore
        )
    os.makedirs(output_folder, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_special_tokens=False)
    # Tok config
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    with open(os.path.join(args.data_folder, "valid_optimization.json"), "r") as f:
        valid_data = json.load(f)
    valid_ds = BaselineDataset(
        valid_data,
        split="valid",
        unstr_kb_path="./original_data/DSTC9/data/",
        args=args,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=Collator(
            tokenizer,  # type: ignore
            args,
            system_prompt,
            max_length=4096 - generation_params["max_new_tokens"],
        ),
    )

    if args.unstr_know == "retrieved":
        file_name = f"optimization_results_{args.unstr_know}_top-{args.top_k}.json"
    else:
        file_name = f"optimization_results_{args.unstr_know}.json"

    if os.path.exists(os.path.join(output_folder, file_name)):
        with open(
            os.path.join(output_folder, file_name),
            "r",
        ) as f:
            optimization_results = json.load(f)
    else:
        optimization_results = {}

    # set the seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    for top_p, temperature, top_k in tqdm(
        product(args.top_p, args.temperature, args.top_k_gen),
        desc="Trying different configs",
        total=len(args.top_p) * len(args.temperature) * len(args.top_k_gen),
    ):
        if f"{top_p}_{temperature}_{top_k}" in optimization_results:
            print(
                f"Skipping {top_p}_{temperature}_{top_k} as it already exists in the results."
            )
            continue

        results = {}
        bleu_scores = []
        for input_ids, attention_mask, targets, sample_ids in tqdm(
            valid_loader, desc=f"Generating with P={top_p}, T={temperature}, K={top_k}"
        ):
            input_ids = input_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)

            output = generate(
                model,
                tokenizer,  # type: ignore
                input_ids,
                attention_mask,
                max_new_tokens=generation_params["max_new_tokens"],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

            # get only the generated tokens
            output = output[:, input_ids.size(1) :]

            output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
            for text, target, sample_id in zip(output_text, targets, sample_ids):
                bleu_scores.append(
                    sentence_bleu(
                        [target.split()],
                        text.split(),
                        weights=(0.25, 0.25, 0.25, 0.25),
                    )
                )
                results[sample_id] = {
                    "target": target,
                    "output": text,
                    "bleu": bleu_scores[-1],
                }

        optimization_results[f"{top_p}_{temperature}_{top_k}"] = {
            "bleu": np.mean(bleu_scores),
            "results": results,
        }

        with open(
            os.path.join(output_folder, file_name),
            "w",
        ) as f:
            json.dump(optimization_results, f, indent=4)
