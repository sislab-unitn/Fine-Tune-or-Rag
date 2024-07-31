import argparse
import json
import os
import pickle
import random
from pathlib import Path


import torch
import numpy as np
from inseq import load_model
from peft.peft_model import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.utils import attribute, InseqDataset, MAP_MODELS


def configure_subparsers(subparsers: argparse._SubParsersAction):
    """Configure a new subparser ."""
    parser = subparsers.add_parser(
        "int-grad",
        help="Calculate integrated gradients for the generated replies of a model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "generation_file",
        metavar="GENERATION_FILE",
        type=str,
        help="Path of the .json file containing the model replies generated using 'generate.py'.",
    )
    parser.add_argument(
        "--experiment-path",
        metavar="EXPERIMENT_PATH",
        type=str,
        default=None,
        help="Path to the experiment folder of the fine-tuned model (without best_model).",
    )
    parser.add_argument(
        "--whitelist",
        metavar="WHITELIST",
        type=str,
        default=None,
        help="Path to the whitelist file.",
    )

    parser.set_defaults(func=main)


def main(args):
    dataset_name = Path(args.data_folder).name
    if dataset_name not in ["DailyDialog", "DSTC9", "NarrativeQA", "WizardOfWikipedia"]:
        raise NotImplementedError(f"Unsupported dataset: '{dataset_name}'")

    model_name = MAP_MODELS[args.model]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if args.parallel else args.device,
        torch_dtype=torch.bfloat16,
    )

    if args.experiment_path is not None:
        model = PeftModel.from_pretrained(
            model, os.path.join(args.experiment_path, "best_model"), is_trainable=False  # type: ignore
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_special_tokens=False)
    # Tok config
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = load_model(
        model,  # type: ignore
        "integrated_gradients",
        tokenizer=tokenizer,
    )

    with open(os.path.join(args.generation_file), "r") as f:
        generated_data = json.load(f)

    if args.whitelist is not None:
        with open(args.whitelist, "r") as f:
            whitelist = json.load(f)
            generated_data["results"] = {
                sample_id: sample_data
                for sample_id, sample_data in generated_data["results"].items()
                if sample_id in whitelist
            }
    generated_ds = InseqDataset(generated_data)
    generated_loader = DataLoader(
        generated_ds,
        batch_size=1,  # batch size set to 1 for Inseq
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # set the seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    attribution_results = {}

    for input_texts, generated_texts, sample_ids in tqdm(
        generated_loader,
        desc=f"Calculating Integrated Gradients for {Path(args.generation_file).name}",
    ):
        attribution = attribute(model, input_texts, generated_texts)
        aggregation = attribution.aggregate()

        for input_text, generated_text, sample_id, sequence_attribution in zip(
            aggregation.info["input_texts"],
            aggregation.info["generated_texts"],
            sample_ids,
            aggregation.sequence_attributions,
        ):
            attribution_results[sample_id] = {
                "input_text": input_text,
                "generated_text": generated_text,
                "attribution": sequence_attribution,
            }

    file_name = (
        Path(args.generation_file).parents[0]
        / f"integrated_gradients_{Path(args.generation_file).stem.split('generation_results_')[-1]}.pkl"
    )

    with open(
        file_name,
        "wb",
    ) as f:
        pickle.dump(attribution_results, f)
