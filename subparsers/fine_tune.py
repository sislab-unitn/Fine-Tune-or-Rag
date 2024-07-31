import argparse
import gc
import json
import os
import random
from pathlib import Path
from pprint import pprint


import torch
import numpy as np
from peft import LoraConfig, PeftModel, get_peft_model  # type: ignore
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer


from utils import daily_dialogue, dstc9, narrative_qa, wizard_of_wikipedia
from utils.utils import (
    Checkpointer,
    evaluate,
    load_training_params,
    resume_training,
    save_training_params,
    train,
    seed_worker,
    MAP_MODELS,
)


def configure_subparsers(subparsers: argparse._SubParsersAction):
    """Configure a new subparser ."""
    parser = subparsers.add_parser(
        "fine-tune",
        help="Fine-Tune a model on a specific dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "experiment_name",
        metavar="EXPERIMENT_NAME",
        type=str,
        help="Name of the experiment.",
    )
    parser.add_argument(
        "--epochs",
        metavar="EPOCHS",
        type=int,
        default=10,
        help="Number of epochs for training.",
    )
    parser.add_argument(
        "--save-every",
        metavar="SAVE_EVERY",
        type=int,
        default=100,
        help="Save model every SAVE_EVERY steps.",
    )
    parser.add_argument(
        "--lr",
        metavar="LR",
        type=float,
        default=1e-4,
        help="Learning rate for training.",
    )
    parser.add_argument(
        "--max-patience",
        metavar="MAX_PATIENCE",
        type=int,
        default=2,
        help="Maximum patience for early stopping.",
    )
    parser.add_argument(
        "--r",
        metavar="R",
        type=int,
        default=32,
        help="Rank for LoRA.",
    )
    parser.add_argument(
        "--lora-alpha",
        metavar="LORA_ALPHA",
        type=int,
        default=64,
        help="Alpha value for LoRA.",
    )
    parser.add_argument(
        "--batch-size",
        metavar="BATCH_SIZE",
        type=int,
        default=4,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--val-batch-size",
        metavar="VAL_BATCH_SIZE",
        type=int,
        default=16,
        help="Batch size for validation.",
    )

    parser.set_defaults(func=main)


def main(args):
    # removes the function from the args for serialization
    del args.func

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
        Collator = narrative_qa.KnowledgeCollator
    elif dataset_name == "WizardOfWikipedia":
        BaselineDataset = wizard_of_wikipedia.KGDDataset
        Collator = wizard_of_wikipedia.KnowledgeCollator
    else:
        raise NotImplementedError(f"Unsupported dataset: '{dataset_name}'")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_folder = os.path.join(
        args.out_dir, Path(args.data_folder).name, args.model, args.experiment_name
    )
    if not os.path.exists(output_folder):
        # create output folder
        os.makedirs(output_folder)
        save_training_params(args, output_folder)
    else:
        # load training parameters from existing folder
        print(f"Experiment '{args.experiment_name}' already exists.")
        print(f"Loading training parameters from {output_folder}.")
        args = load_training_params(output_folder)

    model_name = MAP_MODELS[args.model]

    # Loss function
    criterion = CrossEntropyLoss(ignore_index=-100, reduction="sum")

    # Tok config
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_special_tokens=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if args.parallel else args.device,
        torch_dtype=torch.bfloat16,
    )

    checkpointer = Checkpointer(args)
    early_stopping = checkpointer.checkpoint.early_stopping
    if os.path.exists(os.path.join(output_folder, "checkpoint")):
        checkpoint, model = checkpointer.load_checkpoint(model, output_folder)
        early_stopping = checkpoint.early_stopping
        start_epoch = checkpoint.epoch
    else:
        # Initialize training for the first time
        start_epoch = 0

        # default configuration from https://huggingface.co/docs/peft/en/developer_guides/quantization
        config = LoraConfig(
            r=args.r,
            lora_alpha=args.lora_alpha,
            init_lora_weights="gaussian",
        )

        model = get_peft_model(model, config)

    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if checkpointer.checkpoint.optimizer is not None:
        optimizer.load_state_dict(checkpointer.checkpoint.optimizer)

    with open(os.path.join(args.data_folder, "train.json"), "r") as f:
        train_data = json.load(f)
    with open(os.path.join(args.data_folder, "valid.json"), "r") as f:
        valid_data = json.load(f)
    with open(os.path.join(args.data_folder, "test.json"), "r") as f:
        test_data = json.load(f)
    train_ds = BaselineDataset(
        train_data,
        split="train",
        unstr_kb_path="./original_data/DSTC9/data",
        args=args,
    )
    valid_ds = BaselineDataset(
        valid_data,
        split="valid",
        unstr_kb_path="./original_data/DSTC9/data/",
        args=args,
    )
    test_ds = BaselineDataset(
        test_data,
        split="test",
        unstr_kb_path="./original_data/DSTC9/data_eval/",
        args=args,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=Collator(tokenizer, args),  # type: ignore
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=Collator(tokenizer, args),  # type: ignore
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=Collator(tokenizer, args),  # type: ignore
    )

    if resume_training(start_epoch, args, early_stopping):
        train(
            args,
            model,  # type: ignore
            train_loader,  # type: ignore
            valid_loader,
            criterion,
            optimizer,
            output_folder,
            checkpointer,
        )

    # Free memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Load best model and evaluate on test set
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if args.parallel else args.device,
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(
        model, os.path.join(output_folder, "best_model"), is_trainable=False  # type: ignore
    )
    nll, ppl = evaluate(args, model, test_loader, criterion, is_test=True)  # type: ignore

    results = {
        "Test NLL": nll,
        "Test PPL": ppl,
    }
    pprint(results, indent=4)

    with open(os.path.join(output_folder, f"fine_tune.txt"), "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
