import json
import os
from argparse import Namespace
from typing import List, Tuple

import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils.utils import build_input


class QADataset(Dataset):
    def __init__(
        self,
        data: dict,
        split: str,
        args: Namespace,
        **kwargs,
    ):
        self.questions = []
        self.answers = []
        self.sample_ids = []
        self.gold_summaries = []
        self.retrieved_summaries = []

        with open(os.path.join(args.data_folder, f"{split}_kb.json"), "r") as f:
            kb = json.load(f)

        for sample in data["questions"]:
            question = sample["question"]
            document_id = sample["document_id"]
            question_id = sample["question_id"]
            sample_id = f"{document_id}_{question_id}"
            gold_summary = kb[document_id]
            retrieved_summaries = [
                kb[summary["summary_id"]]
                for summary in sample["retrieved_summaries"][: args.top_k]
            ]

            assert len(retrieved_summaries) == min(
                args.top_k, len(sample["retrieved_summaries"])
            ), "Retrieved summaries length mismatch"

            for answer in sample["answers"]:
                self.questions.append(question)
                self.answers.append(answer)
                self.sample_ids.append(sample_id)
                self.gold_summaries.append(gold_summary)
                self.retrieved_summaries.append(retrieved_summaries)

    def __len__(self):
        return len(self.questions)

    def __getitem__(
        self, idx
    ) -> Tuple[str, str, str, Tuple[str, str], List[Tuple[str, str]]]:
        return (
            self.questions[idx],
            self.answers[idx],
            self.sample_ids[idx],
            self.gold_summaries[idx],
            self.retrieved_summaries[idx],
        )


class BaselineCollator:
    def __init__(
        self, tokenizer: AutoTokenizer, args: Namespace, system_prompt: str = ""
    ):
        self.tokenizer = tokenizer
        self.args = args
        self.system_prompt = system_prompt

    def __call__(
        self, batch: Tuple[str, str, str, Tuple[str, str], List[Tuple[str, str]]]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        input_ids = []
        labels = []
        sample_ids = []

        for i, t, sample_id, *_ in batch:  # type: ignore
            sample_ids.append(sample_id)

            i = build_input([i], self.args, self.tokenizer, self.system_prompt)  # type: ignore
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
        self,
        tokenizer: AutoTokenizer,
        args: Namespace,
        system_prompt: str = "",
        max_length: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.args = args
        self.system_prompt = system_prompt
        self.max_length = max_length

    def is_more_than_max_length(self, input_template: str, answer: str) -> bool:
        i = build_input([input_template], self.args, self.tokenizer, self.system_prompt)
        return (
            len(self.tokenizer.encode(i + answer, add_special_tokens=False))  # type: ignore
            > self.max_length
        )

    def prepare_knowledge(
        self,
        args: Namespace,
        question: str,
        gold_summary: Tuple[str, str],
        retrieved_summaries: List[Tuple[str, str]],
        answer: str,
    ) -> str:
        template = "Question: {}\n\nContext: {}\n\nAnswer:"
        if args.unstr_know == "none":
            knowledge = "NONE"
        elif args.unstr_know == "full":
            NotImplementedError("Full knowledge not supported for NarrativeQA")
        elif args.unstr_know == "gold":
            title, summary = gold_summary["title"], gold_summary["summary"]  # type: ignore
            knowledge = "\n- Title: {}, Summary: {}".format(title, summary)
            if self.is_more_than_max_length(
                template.format(question, knowledge), answer
            ):
                knowledge = ""
        elif args.unstr_know == "retrieved":
            knowledge = ""
            for retrieved_summary in retrieved_summaries:
                title, summary = (
                    retrieved_summary["title"],  # type: ignore
                    retrieved_summary["summary"],  # type: ignore
                )
                # try adding the knowledge
                knowledge_to_add = knowledge + "\n- Title: {}, Summary: {}".format(
                    title, summary
                )
                # check if the length is within the limit
                if self.is_more_than_max_length(
                    template.format(question, knowledge_to_add), answer
                ):
                    break
                # if it is, update the knowledge
                knowledge = knowledge_to_add
        else:
            raise NotImplementedError(f"Unsupported setting: '{args.unstr_know}'")

        return template.format(question, knowledge)

    def __call__(
        self, batch: Tuple[str, str, str, Tuple[str, str], List[Tuple[str, str]]]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        input_ids = []
        labels = []
        sample_ids = []

        for question, answer, sample_id, gold_summary, retrieved_summaries in batch:  # type: ignore
            sample_ids.append(sample_id)

            i = self.prepare_knowledge(
                self.args, question, gold_summary, retrieved_summaries, answer  # type: ignore
            )

            i = build_input([i], self.args, self.tokenizer, self.system_prompt)
            i_ids = torch.tensor(
                self.tokenizer.encode(i + answer, add_special_tokens=False),  # type: ignore
                dtype=torch.long,
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
            assert len(i_ids) <= self.max_length, "Input length exceeds max length"

            input_ids.append(i_ids)
            labels.append(label)

        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id  # type: ignore
        )
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return input_ids, labels, sample_ids


class GenerationDataset(Dataset):
    def __init__(
        self,
        data: dict,
        split: str,
        args: Namespace,
        **kwargs,
    ):
        self.questions = []
        self.answers = []
        self.sample_ids = []
        self.gold_summaries = []
        self.retrieved_summaries = []

        with open(os.path.join(args.data_folder, f"{split}_kb.json"), "r") as f:
            kb = json.load(f)

        for sample in data["questions"]:
            question = sample["question"]
            document_id = sample["document_id"]
            question_id = sample["question_id"]
            sample_id = f"{document_id}_{question_id}"
            gold_summary = kb[document_id]
            retrieved_summaries = [
                kb[summary["summary_id"]]
                for summary in sample["retrieved_summaries"][: args.top_k]
            ]
            answer = sample["answers"][0]

            assert len(retrieved_summaries) == min(
                args.top_k, len(sample["retrieved_summaries"])
            ), "Retrieved summaries length mismatch"

            self.questions.append(question)
            self.answers.append(answer)
            self.sample_ids.append(sample_id)
            self.gold_summaries.append(gold_summary)
            self.retrieved_summaries.append(retrieved_summaries)

    def __len__(self):
        return len(self.questions)

    def __getitem__(
        self, idx
    ) -> Tuple[str, str, str, Tuple[str, str], List[Tuple[str, str]]]:
        return (
            self.questions[idx],
            self.answers[idx],
            self.sample_ids[idx],
            self.gold_summaries[idx],
            self.retrieved_summaries[idx],
        )


class GenerationCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        args: Namespace,
        system_prompt: str = "",
        max_length: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.args = args
        self.system_prompt = system_prompt
        self.max_length = max_length

    def is_more_than_max_length(self, input_template: str) -> bool:
        i = build_input([input_template], self.args, self.tokenizer, self.system_prompt)
        return (
            len(self.tokenizer.encode(i, add_special_tokens=False))  # type: ignore
            > self.max_length
        )

    def prepare_knowledge(
        self,
        args: Namespace,
        question: str,
        gold_summary: Tuple[str, str],
        retrieved_summaries: List[Tuple[str, str]],
    ) -> str:
        template = "Question: {}\n\nContext: {}\n\nAnswer:"
        if args.unstr_know == "none":
            knowledge = "NONE"
        elif args.unstr_know == "full":
            NotImplementedError("Full knowledge not supported for NarrativeQA")
        elif args.unstr_know == "gold":
            title, summary = gold_summary["title"], gold_summary["summary"]  # type: ignore
            knowledge = "\n- Title: {}, Summary: {}".format(title, summary)
            if self.is_more_than_max_length(template.format(question, knowledge)):
                knowledge = ""
        elif args.unstr_know == "retrieved":
            knowledge = ""
            for retrieved_summary in retrieved_summaries:
                title, summary = (
                    retrieved_summary["title"],  # type: ignore
                    retrieved_summary["summary"],  # type: ignore
                )
                # try adding the knowledge
                knowledge_to_add = knowledge + "\n- Title: {}, Summary: {}".format(
                    title, summary
                )
                # check if the length is within the limit
                if self.is_more_than_max_length(
                    template.format(question, knowledge_to_add)
                ):
                    break
                # if it is, update the knowledge
                knowledge = knowledge_to_add
        else:
            raise NotImplementedError(f"Unsupported setting: '{args.unstr_know}'")

        return template.format(question, knowledge)

    def __call__(
        self, batch: Tuple[str, str, str, Tuple[str, str], List[Tuple[str, str]]]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
        input_ids = []
        labels = []
        sample_ids = []

        for question, answer, sample_id, gold_summary, retrieved_summaries in batch:  # type: ignore
            sample_ids.append(sample_id)

            i = self.prepare_knowledge(
                self.args, question, gold_summary, retrieved_summaries  # type: ignore
            )

            i = build_input([i], self.args, self.tokenizer, self.system_prompt)
            i_ids = torch.tensor(
                self.tokenizer.encode(i, add_special_tokens=False),  # type: ignore
                dtype=torch.long,
            )
            assert len(i_ids) <= self.max_length, "Input length exceeds max length"

            input_ids.append(i_ids)
            labels.append(answer)

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
        assert input_ids.shape == attention_mask.shape, "Input and mask shape mismatch"

        return input_ids, attention_mask, labels, sample_ids
