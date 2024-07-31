from argparse import Namespace
from typing import List, Tuple

import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils.utils import build_input


class BaselineDataset(Dataset):
    def __init__(
        self,
        data: dict,
        **kwargs,
    ):
        self.inputs = []
        self.targets = []
        self.sample_ids = []

        for sample_id, sample in data.items():
            self.inputs.append(sample["dial_history"])
            self.targets.append(sample["text"])
            self.sample_ids.append(sample_id)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx) -> Tuple[List[str], str, str]:
        return self.inputs[idx], self.targets[idx], self.sample_ids[idx]


class BaselineCollator:
    def __init__(
        self, tokenizer: AutoTokenizer, args: Namespace, system_prompt: str = ""
    ):
        self.tokenizer = tokenizer
        self.args = args
        self.system_prompt = system_prompt

    def __call__(self, batch) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        input_ids = []
        labels = []
        sample_ids = []

        for i, t, sample_id in batch:
            sample_ids.append(sample_id)

            i = build_input(i, self.args, self.tokenizer, self.system_prompt)
            i_ids = torch.tensor(
                self.tokenizer.encode(i + t, add_special_tokens=False), dtype=torch.long  # type: ignore
            )
            tokens_to_mask = self.tokenizer.encode(i, add_special_tokens=False)[  # type: ignore
                1:
            ]  # skip the bos token

            label = torch.cat(
                (i_ids[1:], torch.tensor([self.tokenizer.eos_token_id]))  # type: ignore
            )  # skip bos and add eos
            label[: len(tokens_to_mask) - 1] = -100  # mask the input tokens

            assert len(label) == len(i_ids), "Input and label length mismatch"
            mask = label[:-1] != -100
            assert torch.equal(
                label[:-1][mask], i_ids[1:][mask]
            ), "Input and label mismatch"

            input_ids.append(i_ids)
            labels.append(label)

        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id  # type: ignore
        )
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return input_ids, labels, sample_ids


class GenerationCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        args: Namespace,
        system_prompt: str = "",
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.args = args
        self.system_prompt = system_prompt

    def __call__(
        self, batch
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
        input_ids = []
        labels = []
        sample_ids = []

        for i, t, sample_id in batch:
            sample_ids.append(sample_id)

            i = build_input(i, self.args, self.tokenizer, self.system_prompt)
            i_ids = torch.tensor(
                self.tokenizer.encode(i, add_special_tokens=False), dtype=torch.long  # type: ignore
            )

            input_ids.append(i_ids)
            labels.append(t)

        reversed_input_ids = [i.flip([0]) for i in input_ids]

        attention_mask = pad_sequence(
            reversed_input_ids, batch_first=True, padding_value=-100  # type: ignore
        )
        attention_mask = (attention_mask != -100).long()
        attention_mask = attention_mask.flip([1])
        reversed_input_ids = pad_sequence(
            reversed_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id  # type: ignore
        )
        input_ids = reversed_input_ids.flip([1])

        return input_ids, attention_mask, labels, sample_ids
