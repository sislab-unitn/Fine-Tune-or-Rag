import argparse
import json
import os
import random
from pathlib import Path
from pprint import pprint


import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer


from utils import daily_dialogue, dstc9, narrative_qa, wizard_of_wikipedia
from utils.utils import evaluate, MAP_MODELS


def configure_subparsers(subparsers: argparse._SubParsersAction):
    """Configure a new subparser ."""
    parser = subparsers.add_parser(
        "baseline",
        help="Baseline evaluation of a model on a specific task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--batch-size",
        metavar="BATCH_SIZE",
        type=int,
        default=16,
        help="Batch size for evaluation.",
    )

    parser.set_defaults(func=main)


def main(args):
    dataset_name = Path(args.data_folder).name
    if dataset_name == "DailyDialog":
        BaselineDataset = daily_dialogue.BaselineDataset
        Collator = daily_dialogue.BaselineCollator
    elif dataset_name == "DSTC9":
        BaselineDataset = dstc9.Dataset
        if args.unstr_know == "none":
            Collator = dstc9.NoKnowledgeCollator
        elif args.unstr_know == "gold":
            Collator = dstc9.UnstrKnowCollator
        elif args.unstr_know == "retrieved":
            Collator = dstc9.RetrievedUnstrKnowCollator
        else:
            raise NotImplementedError(f"Unsupported setting: '{args.unstr_know}'")
    elif dataset_name == "NarrativeQA":
        BaselineDataset = narrative_qa.QADataset
        Collator = narrative_qa.BaselineCollator
    elif dataset_name == "WizardOfWikipedia":
        BaselineDataset = wizard_of_wikipedia.KGDDataset
        Collator = wizard_of_wikipedia.BaselineCollator
    else:
        raise NotImplementedError(f"Unsupported dataset: '{dataset_name}'")

    output_folder = os.path.join(args.out_dir, Path(args.data_folder).name, args.model)
    os.makedirs(output_folder, exist_ok=True)

    model_name = MAP_MODELS[args.model]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if args.parallel else args.device,
        torch_dtype=torch.bfloat16,
    )

    criterion = CrossEntropyLoss(ignore_index=-100, reduction="sum")
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_special_tokens=False)
    # Tok config
    tokenizer.pad_token_id = tokenizer.eos_token_id

    with open(os.path.join(args.data_folder, "test.json"), "r") as f:
        test_data = json.load(f)

    test_ds = BaselineDataset(test_data, split="test", unstr_kb_path="./original_data/DSTC9/data_eval/", args=args)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=Collator(tokenizer, args),  # type: ignore
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    nll, ppl = evaluate(args, model, test_loader, criterion, is_test=True)

    results = {
        "Test NLL": nll,
        "Test PPL": ppl,
    }
    pprint(results, indent=4)

    with open(os.path.join(output_folder, f"baseline.txt"), "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")