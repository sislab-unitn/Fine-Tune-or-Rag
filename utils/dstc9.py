import torch
from argparse import Namespace
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data
from transformers import AutoTokenizer
from utils.db import MwozDataBase
from utils.utils import build_input
from tqdm import tqdm
from alexa_with_dstc9_track1_dataset.scripts.knowledge_reader import KnowledgeReader


class Dataset(data.Dataset):
    def __init__(
        self,
        data: Dict[int, Dict[str, Union[Any, List[Dict[str, Any]]]]],
        split: str,
        db_path: Optional[str] = None,
        unstr_kb_path: Optional[str] = None,
        **kwargs,
    ):
        self.hist: List[List[str]] = []
        self.dialogue_state: List[Dict[str, str]] = []
        self.dialogue_act: List[Dict[str, str]] = []
        self.query_results: List[Dict[str, List[Dict[str, Any]]]] = []
        self.gold_knowledge: List[List[Dict[str, Any]]] = []
        self.retrieved_knowledge: List[List[Dict[str, Any]]] = []
        self.targets: List[str] = []
        self.sample_ids: List[str] = []
        self.db: Optional[MwozDataBase] = None
        if db_path is not None:
            self.db = MwozDataBase(db_path, accept_dontcare=True)
        self.unstr_kb: Optional[KnowledgeReader] = None
        if unstr_kb_path is not None:
            self.unstr_kb = KnowledgeReader(unstr_kb_path, "knowledge.json")
            # with open(os.path.join(unstr_kb_path, "knowledge.json")) as f:
            #     kb = json.load(f)

        for dial_id, dial in tqdm(
            data.items(), desc="Preparing dataset", unit="dialogues"
        ):
            # skip test dialogues that are not from multiwoz 2.1 (no dialogue state annotation)
            if split == "test" and (
                "source" not in dial or dial["source"] != "multiwoz"
            ):
                continue
            for t_id, turn in enumerate(dial["turns"]):
                # skip user turns
                if t_id % 2 == 0:
                    continue
                history = [t["text"] for t in dial["turns"][:t_id]]
                self.hist.append(history)
                self.dialogue_state.append(turn["dialogue_state"])
                if "dialogue_act" in turn:
                    self.dialogue_act.append(turn["dialogue_act"])
                else:
                    self.dialogue_act.append({})
                self.query_results.append(self._get_query_results(turn))
                self.gold_knowledge.append(self._get_knowledge(turn, "knowledge"))
                self.retrieved_knowledge.append(self._get_knowledge(turn, "top_k"))
                self.targets.append(turn["text"])
                self.sample_ids.append(f"{split}_{dial_id}_{t_id}")

    def _get_query_results(
        self, turn: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        results = {}
        if self.db:
            results = self.db.query(turn["dialogue_state"], exclude_long_fields=True)
            # summarised = summarise_query_result(results, 3)
        return results

    def _get_knowledge(
        self, turn: Dict[str, Any], k_field: str = "knowledge"
    ) -> List[Dict[str, Any]]:
        results = []
        if self.unstr_kb:
            if k_field in turn and turn[k_field] is not None:
                for entry in turn[k_field]:
                    results.append(
                        self.unstr_kb.get_doc(
                            entry["domain"], entry["entity_id"], entry["doc_id"]
                        )
                    )
        return results

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx) -> Tuple[
        str,
        Dict[str, str],
        Dict[str, str],
        Dict[str, List[Dict[str, Any]]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        str,
        str,
    ]:
        return (
            self.hist[idx],
            self.dialogue_state[idx],
            self.dialogue_act[idx],
            self.query_results[idx],
            self.gold_knowledge[idx],
            self.retrieved_knowledge[idx],
            self.targets[idx],
            self.sample_ids[idx],
        )


class HistOnlyCollator:
    def __init__(
        self, tokenizer: AutoTokenizer, args: Namespace, system_prompt: str = ""
    ):
        self.tokenizer: AutoTokenizer = tokenizer
        self.args = args
        self.system_prompt = system_prompt

    def __call__(self, batch) -> Any:
        input_ids = []
        labels = []
        sample_ids = []

        for (
            history,
            dialogue_state,
            dialogue_act,
            query_results,
            gold_knowledge,
            retrieved_knowledge,
            target,
            sample_id,
        ) in batch:
            sample_ids.append(sample_id)

            history = build_input(
                history, self.args, self.tokenizer, self.system_prompt
            )
            i_ids = torch.tensor(
                self.tokenizer.encode(history + target, add_special_tokens=False), dtype=torch.long  # type: ignore
            )
            history_ids = self.tokenizer.encode(history, add_special_tokens=False)  # type: ignore
            assert self.tokenizer.decode(history_ids[0]) == self.tokenizer.bos_token  # type: ignore
            tokens_to_mask = history_ids[1:]  # type: ignore  # skip the bos token

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


def get_ds_str(dialogue_state: Dict[str, str]) -> str:
    if len(dialogue_state) == 0:
        ds_str = "none"
    else:
        ds_list = []
        ds_str_list = []
        for slot, value in dialogue_state.items():
            domain, slot = slot.split("-")
            ds_list.append((domain, slot, value))
        ds_list = sorted(ds_list)
        for domain, slot, value in ds_list:
            # ds_str_list.append(f"[domain] {domain} [slot] {slot} [value] {value}")
            ds_str_list.append(f"{domain} {slot} {value},")
        assert ds_str_list[-1][-1] == ","
        ds_str_list[-1] = ds_str_list[-1][:-1]
        ds_str = " ".join(ds_str_list)
    return ds_str


def get_da_str(dialogue_act: Dict[str, str]) -> str:
    if len(dialogue_act) == 0:
        da_str = "none"
    da_str_list = []
    for act, value in dialogue_act.items():
        try:
            assert len(value) == 1
        except:
            # TODO manage more values
            # print("dial act with more values")
            pass
            # import pdb; pdb.set_trace()
        slot, value = value[0]
        if value not in ["none", "?"]:
            da_str_list.append(f"[dialogue_act] {act} [slot] {slot} [value] {value}")
    da_str = " ".join(da_str_list)
    return da_str


def get_qr_str(query_results: Dict[str, List[Dict[str, Any]]]) -> str:
    if len(query_results) == 0:
        qr_str = "none"
    qr_str_list = []
    for table, results in query_results.items():
        for res in results:
            qr_str_list.append("[result]")
            for field, value in res.items():
                qr_str_list.append(f"[table] {table} [field] {field} [value] {value}")
    qr_str = " ".join(qr_str_list)
    return qr_str


class NoKnowledgeCollator:
    def __init__(
        self, tokenizer: AutoTokenizer, args: Namespace, system_prompt: str = ""
    ):
        self.tokenizer: AutoTokenizer = tokenizer
        self.args = args
        self.system_prompt = system_prompt
        print("NoKnowledgeCollator")

    def __call__(self, batch) -> Any:
        input_ids = []
        labels = []
        sample_ids = []

        for (
            history,
            dialogue_state,
            dialogue_act,
            query_results,
            gold_knowledge,
            retrieved_knowledge,
            target,
            sample_id,
        ) in batch:
            sample_ids.append(sample_id)

            ds_str = get_ds_str(dialogue_state)
            # append to last turn of dialogue history
            history[-1] = f"{history[-1]} Dialogue state: {ds_str}"

            history = build_input(
                history, self.args, self.tokenizer, self.system_prompt
            )

            i_ids = torch.tensor(
                self.tokenizer.encode(history + target, add_special_tokens=False), dtype=torch.long  # type: ignore
            )
            history_ids = self.tokenizer.encode(history, add_special_tokens=False)  # type: ignore
            assert self.tokenizer.decode(history_ids[0]) == self.tokenizer.bos_token  # type: ignore
            tokens_to_mask = history_ids[1:]  # type: ignore  # skip the bos token

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


def get_uk_str(
    unstr_kn: List[Dict[str, Any]], top_k: Optional[int] = None, use_name: bool = False
) -> str:
    if len(unstr_kn) == 0:
        uk_str = "none"
    else:
        uk_str_list = []
        if top_k is None:
            top_k = len(unstr_kn)
        for entry in unstr_kn[: min(len(unstr_kn), top_k)]:
            question = entry["doc"]["title"]
            answer = entry["doc"]["body"]
            if not use_name:
                uk_str_list.append(f"{question} {answer}\n")
            else:
                name = entry["entity_name"]
                if entry["domain"] in ["taxi", "train"]:
                    name = entry["domain"].upper()
                uk_str_list.append(f"{name}, {question} {answer}\n")
        assert uk_str_list[-1][-1] == "\n"
        uk_str_list[-1] = uk_str_list[-1][:-1]
        uk_str = " ".join(uk_str_list)
    return uk_str


class UnstrKnowCollator:
    def __init__(
        self, tokenizer: AutoTokenizer, args: Namespace, system_prompt: str = ""
    ):
        self.tokenizer: AutoTokenizer = tokenizer
        self.args = args
        self.system_prompt = system_prompt
        print("UnstrKnowCollator")

    def __call__(self, batch) -> Any:
        input_ids = []
        labels = []
        sample_ids = []

        for (
            history,
            dialogue_state,
            dialogue_act,
            query_results,
            gold_knowledge,
            retrieved_knowledge,
            target,
            sample_id,
        ) in batch:
            sample_ids.append(sample_id)

            ds_str = get_ds_str(dialogue_state)
            uk_str = get_uk_str(gold_knowledge)
            # append to last turn of dialogue history
            history[-1] = f"{history[-1]} Dialogue state: {ds_str} Knowledge: {uk_str}"

            history = build_input(
                history, self.args, self.tokenizer, self.system_prompt
            )

            i_ids = torch.tensor(
                self.tokenizer.encode(history + target, add_special_tokens=False), dtype=torch.long  # type: ignore
            )
            history_ids = self.tokenizer.encode(history, add_special_tokens=False)  # type: ignore
            assert self.tokenizer.decode(history_ids[0]) == self.tokenizer.bos_token  # type: ignore
            tokens_to_mask = history_ids[1:]  # type: ignore  # skip the bos token

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


class RetrievedUnstrKnowCollator:
    def __init__(
        self, tokenizer: AutoTokenizer, args: Namespace, system_prompt: str = ""
    ):
        self.tokenizer: AutoTokenizer = tokenizer
        self.args = args
        self.system_prompt = system_prompt
        print("RetrievedUnstrKnowCollator")

    def __call__(self, batch) -> Any:
        input_ids = []
        labels = []
        sample_ids = []

        for (
            history,
            dialogue_state,
            dialogue_act,
            query_results,
            gold_knowledge,
            retrieved_knowledge,
            target,
            sample_id,
        ) in batch:
            sample_ids.append(sample_id)

            ds_str = get_ds_str(dialogue_state)
            uk_str = get_uk_str(
                retrieved_knowledge, top_k=self.args.top_k, use_name=True
            )
            # append to last turn of dialogue history
            history[-1] = f"{history[-1]} Dialogue state: {ds_str} Knowledge: {uk_str}"

            history = build_input(
                history, self.args, self.tokenizer, self.system_prompt
            )

            i_ids = torch.tensor(
                self.tokenizer.encode(history + target, add_special_tokens=False), dtype=torch.long  # type: ignore
            )
            history_ids = self.tokenizer.encode(history, add_special_tokens=False)  # type: ignore
            # assert self.tokenizer.decode(history_ids[0]) == self.tokenizer.bos_token
            tokens_to_mask = history_ids[1:]  # type: ignore  # skip the bos token

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


class OptimizationDataset(data.Dataset):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        **kwargs,
    ):
        self.hist: List[List[str]] = []
        self.dialogue_state: List[Dict[str, str]] = []
        self.dialogue_act: List[Dict[str, str]] = []
        self.query_results: List[Dict[str, List[Dict[str, Any]]]] = []
        self.gold_knowledge: List[List[Dict[str, Any]]] = []
        self.retrieved_knowledge: List[List[Dict[str, Any]]] = []
        self.targets: List[str] = []
        self.sample_ids: List[str] = []

        for sample in data:
            self.hist.append(sample["history"])
            self.dialogue_state.append(sample["dialogue_state"])
            self.dialogue_act.append(sample["dialogue_act"])
            self.query_results.append(sample["query_results"])
            self.gold_knowledge.append(sample["gold_knowledge"])
            self.retrieved_knowledge.append(sample["retrieved_knowledge"])
            self.targets.append(sample["target"])
            self.sample_ids.append(sample["sample_id"])

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx) -> Tuple[
        List[str],
        Dict[str, str],
        Dict[str, str],
        Dict[str, List[Dict[str, Any]]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        str,
        str,
    ]:
        return (
            self.hist[idx],
            self.dialogue_state[idx],
            self.dialogue_act[idx],
            self.query_results[idx],
            self.gold_knowledge[idx],
            self.retrieved_knowledge[idx],
            self.targets[idx],
            self.sample_ids[idx],
        )


class NoKnowledgeGenerationCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        args: Namespace,
        system_prompt: str = "",
        **kwargs,
    ):
        self.tokenizer: AutoTokenizer = tokenizer
        self.args = args
        self.system_prompt = system_prompt
        print("NoKnowledgeCollator")

    def __call__(self, batch) -> Any:
        input_ids = []
        labels = []
        sample_ids = []

        for (
            history,
            dialogue_state,
            dialogue_act,
            query_results,
            gold_knowledge,
            retrieved_knowledge,
            target,
            sample_id,
        ) in batch:
            sample_ids.append(sample_id)

            ds_str = get_ds_str(dialogue_state)
            # append to last turn of dialogue history
            history[-1] = f"{history[-1]} Dialogue state: {ds_str}"

            history = build_input(
                history, self.args, self.tokenizer, self.system_prompt
            )

            i_ids = torch.tensor(
                self.tokenizer.encode(history, add_special_tokens=False), dtype=torch.long  # type: ignore
            )

            input_ids.append(i_ids)
            labels.append(target)

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


class UnstrKnowGenerationCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        args: Namespace,
        system_prompt: str = "",
        **kwargs,
    ):
        self.tokenizer: AutoTokenizer = tokenizer
        self.args = args
        self.system_prompt = system_prompt
        print("UnstrKnowCollator")

    def __call__(self, batch) -> Any:
        input_ids = []
        labels = []
        sample_ids = []

        for (
            history,
            dialogue_state,
            dialogue_act,
            query_results,
            gold_knowledge,
            retrieved_knowledge,
            target,
            sample_id,
        ) in batch:
            sample_ids.append(sample_id)

            ds_str = get_ds_str(dialogue_state)
            uk_str = get_uk_str(gold_knowledge)
            # append to last turn of dialogue history
            history[-1] = f"{history[-1]} Dialogue state: {ds_str} Knowledge: {uk_str}"

            history = build_input(
                history, self.args, self.tokenizer, self.system_prompt
            )

            i_ids = torch.tensor(
                self.tokenizer.encode(history, add_special_tokens=False), dtype=torch.long  # type: ignore
            )

            input_ids.append(i_ids)
            labels.append(target)

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


class RetrievedUnstrGenerationKnowCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        args: Namespace,
        system_prompt: str = "",
        **kwargs,
    ):
        self.tokenizer: AutoTokenizer = tokenizer
        self.args = args
        self.system_prompt = system_prompt
        print("RetrievedUnstrKnowCollator")

    def __call__(self, batch) -> Any:
        input_ids = []
        labels = []
        sample_ids = []

        for (
            history,
            dialogue_state,
            dialogue_act,
            query_results,
            gold_knowledge,
            retrieved_knowledge,
            target,
            sample_id,
        ) in batch:
            sample_ids.append(sample_id)

            ds_str = get_ds_str(dialogue_state)
            uk_str = get_uk_str(
                retrieved_knowledge, top_k=self.args.top_k, use_name=True
            )
            # append to last turn of dialogue history
            history[-1] = f"{history[-1]} Dialogue state: {ds_str} Knowledge: {uk_str}"

            history = build_input(
                history, self.args, self.tokenizer, self.system_prompt
            )

            i_ids = torch.tensor(
                self.tokenizer.encode(history, add_special_tokens=False), dtype=torch.long  # type: ignore
            )

            input_ids.append(i_ids)
            labels.append(target)

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
