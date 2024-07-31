import json
import math
import os
import random
from argparse import Namespace
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from inseq.models.attribution_model import AttributionModel, FeatureAttributionOutput
from peft.peft_model import PeftModel
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


MAP_MODELS = {
    "llama": "meta-llama/Llama-2-7b-chat-hf",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
    "gpt2": "gpt2",
}


def build_input(
    input_segments: List[str],
    args: Namespace,
    tokenizer: AutoTokenizer,
    system_prompt: str = "",
) -> str:
    conversation = []
    if args.model == "llama":
        conversation = [
            f"{tokenizer.bos_token}[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"  # type: ignore
        ]
        for i, in_seg in enumerate(input_segments):
            if i % 2 == 0:  # user
                conversation.append(f"{in_seg} [/INST] ")
            else:  # system
                conversation.append(
                    f"{in_seg} {tokenizer.eos_token}{tokenizer.bos_token}[INST] "  # type: ignore
                )
    elif args.model == "mistral":
        conversation = [f"{tokenizer.bos_token}[INST] {system_prompt} "]  # type: ignore
        for i, in_seg in enumerate(input_segments):
            if i % 2 == 0:  # user
                conversation.append(f"{in_seg} [/INST] ")
            else:  # system
                conversation.append(f"{in_seg}{tokenizer.eos_token} [INST] ")  # type: ignore
    else:
        conversation = [f" {system_prompt} "]  # type: ignore
        for i, in_seg in enumerate(input_segments):
            conversation.append(f"{in_seg} \n\n ")

    return "".join(conversation)


def compute_nll_and_ppl(
    losses: List[float], unmasked_tokens: int
) -> Tuple[float, float]:
    nll = sum(losses) / unmasked_tokens
    ppl = math.exp(nll)
    return nll, ppl


def evaluate(
    args,
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    criterion: CrossEntropyLoss,
    is_test: bool = False,
) -> Tuple[float, float]:

    losses = []
    model.eval()  #  type: ignore
    unmasked_tokens = 0
    with torch.no_grad():
        for input_ids, labels, _ in (pbar := tqdm(dataloader, desc="Evaluating")):
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)

            with torch.no_grad():
                outputs = model(input_ids).logits.permute(0, 2, 1)  # type: ignore
                losses.append(criterion(outputs, labels).item())
                unmasked_tokens += (labels != -100).sum().item()
                _, ppl = compute_nll_and_ppl(losses, unmasked_tokens)

            pbar.set_postfix(
                {
                    f"{'Test' if is_test else 'Valid'} PPL": ppl,
                }
            )
    return compute_nll_and_ppl(losses, unmasked_tokens)


def save_training_params(args: Namespace, output_folder: str) -> None:
    with open(os.path.join(output_folder, "training_params.json"), "w") as f:
        json.dump(vars(args), f, indent=4)


def load_training_params(output_folder: str) -> Namespace:
    with open(os.path.join(output_folder, "training_params.json"), "r") as f:
        return Namespace(**json.load(f))


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.counter = 0
        self.best_ppl = float("inf")
        self.stopped = False

    def should_stop(self, current_ppl: float) -> bool:
        if current_ppl < self.best_ppl:
            self.best_ppl = current_ppl
            self.counter = 0
        else:
            self.counter += 1

        # save the counter condition for the checkpointer
        self.stopped = self.counter >= self.patience
        return self.stopped


class Checkpoint:
    def __init__(self, args: Namespace):
        self.epoch = 0
        self.step = 0
        self.optimizer: Optional[dict] = None
        self.early_stopping = EarlyStopping(args.max_patience)
        self.train_stats = []
        self.losses = []
        self.unmasked_tokens = 0


class Checkpointer:
    def __init__(self, args: Namespace):
        self.checkpoint = Checkpoint(args)

    def update_checkpoint(
        self,
        model: PeftModel,
        optimizer: torch.optim.Optimizer,
        step: int,
        losses: List[float],
        unmasked_tokens: int,
        output_folder: str,
        train_stats: Optional[List[dict]] = None,
        early_stopping: Optional[EarlyStopping] = None,
        epoch: Optional[int] = None,
    ):
        model.save_pretrained(os.path.join(output_folder, "checkpoint"))
        self.checkpoint.optimizer = optimizer.state_dict()
        if epoch is not None:
            self.checkpoint.epoch = epoch
        self.checkpoint.step = step
        self.checkpoint.losses = losses
        self.checkpoint.unmasked_tokens = unmasked_tokens
        if train_stats is not None:
            self.checkpoint.train_stats = train_stats
        if early_stopping is not None:
            self.checkpoint.early_stopping = early_stopping
        torch.save(
            self.checkpoint, os.path.join(output_folder, "checkpoint", "checkpoint.pt")
        )

    def load_checkpoint(
        self, model: AutoModelForCausalLM, output_folder: str
    ) -> Tuple[Checkpoint, PeftModel]:
        model = PeftModel.from_pretrained(
            model, os.path.join(output_folder, "checkpoint"), is_trainable=True  # type: ignore
        )
        self.checkpoint = torch.load(
            os.path.join(output_folder, "checkpoint", "checkpoint.pt")
        )
        return self.checkpoint, model  # type: ignore


def resume_training(epoch: int, args: Namespace, early_stopping: EarlyStopping) -> bool:
    if epoch >= args.epochs:
        print(f"Reached maximum number of epochs {epoch}.")
        return False
    if early_stopping.stopped:
        print(f"Early stopping at epoch {epoch}")
        return False
    return True


def train_one_epoch(
    args,
    model: PeftModel,
    optimizer: torch.optim.Optimizer,
    train_iterator: Iterator,
    dataloader: DataLoader,
    criterion: CrossEntropyLoss,
    start_step: int,
    steps_so_far: int,
    checkpointer: Checkpointer,
    output_folder: str,
) -> Tuple[Tuple[float, float], int]:

    model.train()  # type: ignore
    losses = checkpointer.checkpoint.losses
    unmasked_tokens = checkpointer.checkpoint.unmasked_tokens

    # Resume training for the current step
    for _ in range(start_step):
        next(train_iterator)

    with tqdm(dataloader, desc="Training") as pbar:
        pbar.total = len(dataloader)
        pbar.n = start_step
        pbar.refresh()
        for step, (input_ids, labels, _) in enumerate(train_iterator, start=start_step):
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)

            optimizer.zero_grad()
            outputs = model(input_ids).logits.permute(0, 2, 1)  # type: ignore

            # compute the loss as sum
            loss = criterion(outputs, labels)
            # store the original loss
            losses.append(loss.item())
            # compute the number of unmasked tokens for this batch
            batch_unmasked_tokens = (labels != -100).sum().item()
            # normalize the loss
            loss /= batch_unmasked_tokens
            loss.backward()
            optimizer.step()

            unmasked_tokens += batch_unmasked_tokens
            _, ppl = compute_nll_and_ppl(losses, unmasked_tokens)
            pbar.set_postfix(
                {
                    "Train PPL": ppl,
                }
            )
            pbar.update(1)
            steps_so_far += 1

            if steps_so_far % args.save_every == 0:
                checkpointer.update_checkpoint(
                    model,
                    optimizer,
                    step + 1,  # We finished the current step, so we increment the step
                    losses,
                    unmasked_tokens,
                    output_folder,
                )

    return compute_nll_and_ppl(losses, unmasked_tokens), steps_so_far


def train(
    args: Namespace,
    model: PeftModel,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    criterion: CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    output_folder: str,
    checkpointer: Checkpointer,
) -> None:

    start_epoch = checkpointer.checkpoint.epoch
    start_step = checkpointer.checkpoint.step
    train_stats = checkpointer.checkpoint.train_stats
    early_stopping = checkpointer.checkpoint.early_stopping

    steps_so_far = len(train_loader) * start_epoch + start_step

    # Resume training for the current epoch
    for _ in range(start_epoch):
        iter(train_loader)

    best_ppl = early_stopping.best_ppl
    for epoch in trange(
        start_epoch,
        args.epochs,
        desc="Epochs",
        initial=start_epoch,
        total=args.epochs,
    ):
        train_iterator = iter(train_loader)
        (train_nnl, train_ppl), steps_so_far = train_one_epoch(
            args,
            model,
            optimizer,
            train_iterator,
            train_loader,
            criterion,
            start_step,
            steps_so_far,
            checkpointer,
            output_folder,
        )
        valid_nll, valid_ppl = evaluate(args, model, valid_loader, criterion)  # type: ignore

        # Save best model
        if valid_ppl < best_ppl:
            best_ppl = valid_ppl
            model.save_pretrained(os.path.join(output_folder, "best_model"))

        # Save the results for the checkpointer
        should_stop = early_stopping.should_stop(valid_ppl)

        train_stats.append(
            {
                "Epoch": epoch,
                "Train NLL": train_nnl,
                "Train PPL": train_ppl,
                "Valid NLL": valid_nll,
                "Valid PPL": valid_ppl,
                "Patience": early_stopping.patience - early_stopping.counter,
            }
        )

        # Update the checkpointer
        checkpointer.update_checkpoint(
            model=model,
            optimizer=optimizer,
            step=0,  # We finished the epoch, so we reset the step
            losses=[],  # We finished the epoch, so we reset the losses
            unmasked_tokens=0,  # We finished the epoch, so we reset the unmasked tokens
            output_folder=output_folder,
            train_stats=train_stats,
            early_stopping=early_stopping,
            epoch=epoch + 1,  # We finished the current epoch, so we increment the epoch
        )

        start_step = 0

        with open(os.path.join(output_folder, "train_stats.json"), "w") as f:
            json.dump(train_stats, f, indent=4)

        # Early stopping
        if should_stop:
            break


def generate(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    model.eval()  # type: ignore
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,  # type: ignore
            do_sample=True,
        )
        return outputs


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def attribute(
    model: AttributionModel,
    input_texts: List[str],
    generated_texts: List[str],
) -> FeatureAttributionOutput:

    with torch.no_grad():
        attribution = model.attribute(
            input_texts=input_texts,
            generated_texts=generated_texts,
            show_progress=False,
        )
    return attribution


class InseqDataset(Dataset):
    def __init__(
        self,
        data: dict,
        **kwargs,
    ):
        self.input_texts = []
        self.generated_texts = []
        self.sample_ids = []

        for sample_id, sample in data["results"].items():
            self.input_texts.append(sample["input_text"])
            self.generated_texts.append(sample["generated_text"])
            self.sample_ids.append(sample_id)

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx) -> Tuple[str, str, str]:
        return self.input_texts[idx], self.generated_texts[idx], self.sample_ids[idx]
