from argparse import Namespace
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils.utils import build_input


class KGDDataset(Dataset):
    def __init__(
        self,
        data: dict,
        args: Namespace,
        **kwargs,
    ):
        self.inputs = []
        self.targets = []
        self.sample_ids = []
        self.gold_knowledge = []
        self.retrieved_knowledge = []

        for sample_id, sample in data.items():
            self.inputs.append(sample["dialogue_history"])
            self.targets.append(sample["text"])
            self.sample_ids.append(sample_id)
            self.gold_knowledge.append(sample["gold_sentence"])

            retrieved_knowledge = []
            for k in sample["retrieved_sentences"]:
                # skip sentences which are too long
                if len(k["sentence"].split()) > 512:
                    continue
                retrieved_knowledge.append(k)
                if len(retrieved_knowledge) == args.top_k:
                    break
            self.retrieved_knowledge.append(retrieved_knowledge)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx) -> Tuple[List[str], str, str, str, List[str]]:
        return (
            self.inputs[idx],
            self.targets[idx],
            self.sample_ids[idx],
            self.gold_knowledge[idx],
            self.retrieved_knowledge[idx],
        )


class BaselineCollator:
    def __init__(
        self, tokenizer: AutoTokenizer, args: Namespace, system_prompt: str = ""
    ):
        self.tokenizer = tokenizer
        self.args = args
        self.system_prompt = system_prompt

    def __call__(
        self, batch: List[Tuple[List[str], str, str, str, List[str]]]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        input_ids = []
        labels = []
        sample_ids = []

        for i, t, sample_id, *_ in batch:
            sample_ids.append(sample_id)

            topic, dialogue_history = i[0], i[1:]

            # The first turn is the one of the apprentice
            if len(dialogue_history) % 2 == 1:
                first_turn, dialogue_history = dialogue_history[0], dialogue_history[1:]
                i = [f"Topic: {topic}\nDialogue: {first_turn}"] + dialogue_history
            else:
                i = [f"Topic: {topic}\nDialogue:"] + dialogue_history

            i = i[:-1] + [i[-1] + " Knowledge: 'None'"]

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


class KnowledgeCollator:
    def __init__(
        self, tokenizer: AutoTokenizer, args: Namespace, system_prompt: str = ""
    ):
        self.tokenizer = tokenizer
        self.args = args
        self.system_prompt = system_prompt

    def add_knowledge(
        self,
        i: List[str],
        gold_knowledge: Dict[str, str],
        retrieved_knowledge: List[Dict[str, str]],
        args,
    ):
        if args.unstr_know == "none":
            knowledge = " Knowledge: 'Not Available'"
        elif args.unstr_know == "gold":
            if gold_knowledge["sentence"] is not None:
                knowledge = f" Knowledge:\n-Topic: {gold_knowledge['topic']}, Info: {gold_knowledge['sentence']}"
            else:
                knowledge = " Knowledge: 'Not Available'"
        elif args.unstr_know == "retrieved":
            if gold_knowledge["sentence"] is None:
                knowledge = " Knowledge: 'Not Available'"
            else:
                knowledge = " Knowledge:"
                for k in retrieved_knowledge:
                    knowledge += f"\n-Topic: {k['topic']}, Info: {k['sentence']}"
        else:
            raise NotImplementedError(f"Unsupported setting: '{args.unstr_know}'")

        i = i[:-1] + [i[-1] + knowledge]

        return i

    def __call__(
        self,
        batch: List[Tuple[List[str], str, str, Dict[str, str], List[Dict[str, str]]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        input_ids = []
        labels = []
        sample_ids = []

        for i, t, sample_id, gold_knowledge, retrieved_knowledge in batch:
            sample_ids.append(sample_id)

            topic, dialogue_history = i[0], i[1:]

            # The first turn is the one of the apprentice
            if len(dialogue_history) % 2 == 1:
                first_turn, dialogue_history = dialogue_history[0], dialogue_history[1:]
                i = [f"Topic: {topic}\nDialogue: {first_turn}"] + dialogue_history
            else:
                i = [f"Topic: {topic}\nDialogue:"] + dialogue_history

            # add the knowledge to the last turn
            i = self.add_knowledge(i, gold_knowledge, retrieved_knowledge, self.args)

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

    def add_knowledge(
        self,
        i: List[str],
        gold_knowledge: Dict[str, str],
        retrieved_knowledge: List[Dict[str, str]],
        args,
    ):
        if args.unstr_know == "none":
            knowledge = " Knowledge: 'Not Available'"
        elif args.unstr_know == "gold":
            if gold_knowledge["sentence"] is not None:
                knowledge = f" Knowledge:\n-Topic: {gold_knowledge['topic']}, Info: {gold_knowledge['sentence']}"
            else:
                knowledge = " Knowledge: 'Not Available'"
        elif args.unstr_know == "retrieved":
            if gold_knowledge["sentence"] is None:
                knowledge = " Knowledge: 'Not Available'"
            else:
                knowledge = " Knowledge:"
                for k in retrieved_knowledge:
                    knowledge += f"\n-Topic: {k['topic']}, Info: {k['sentence']}"
        else:
            raise NotImplementedError(f"Unsupported setting: '{args.unstr_know}'")

        i = i[:-1] + [i[-1] + knowledge]

        return i

    def __call__(
        self,
        batch: List[Tuple[List[str], str, str, Dict[str, str], List[Dict[str, str]]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
        input_ids = []
        labels = []
        sample_ids = []

        for i, t, sample_id, gold_knowledge, retrieved_knowledge in batch:
            sample_ids.append(sample_id)

            topic, dialogue_history = i[0], i[1:]

            # The first turn is the one of the apprentice
            if len(dialogue_history) % 2 == 1:
                first_turn, dialogue_history = dialogue_history[0], dialogue_history[1:]
                i = [f"Topic: {topic}\nDialogue: {first_turn}"] + dialogue_history
            else:
                i = [f"Topic: {topic}\nDialogue:"] + dialogue_history

            # add the knowledge to the last turn
            i = self.add_knowledge(i, gold_knowledge, retrieved_knowledge, self.args)

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
