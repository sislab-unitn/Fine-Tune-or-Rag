import json
import re
import os
from copy import deepcopy
from nltk.tokenize import word_tokenize
from nltk.metrics.distance import edit_distance
from typing import Any, Dict, List, Set, Tuple, Union
from tqdm import tqdm
from collections import Counter
from pprint import pprint
from langchain_community.vectorstores import FAISS

import sys

sys.path.append("../llms4nlg")
from alexa_with_dstc9_track1_dataset.scripts.dataset_walker import DatasetWalker
from alexa_with_dstc9_track1_dataset.scripts.knowledge_reader import KnowledgeReader
from retriever import get_vector_store, query_vector_store


def is_match(dial: List[Dict[str, Any]], match: List[Dict[str, Any]]) -> bool:
    matched = True
    assert len(dial) >= len(match)
    for t_id, turn in enumerate(match):
        if turn["text"] != dial[t_id]["text"]:
            matched = False
            break
    return matched


def unify(
    dial: Dict[str, Union[Any, List[Dict[str, Any]]]],
    match: Dict[str, Union[Any, List[Dict[str, Any]]]],
) -> Dict[str, Union[Any, List[Dict[str, Any]]]]:
    for t_id, turn in enumerate(match["turns"]):
        assert turn["text"] == dial["turns"][t_id]["text"]
        if "knowledge" in turn:
            if "knowledge" in dial["turns"][t_id]:
                assert str(turn["knowledge"]) == str(dial["turns"][t_id]["knowledge"])
            else:
                # add knowledge if not already present
                print("Unifying")
                dial["turns"][t_id]["knowledge"] = turn["knowledge"]
    return dial


def unify_dials(
    dials: Dict[int, Dict[str, Union[Any, List[Dict[str, Any]]]]]
) -> Dict[int, Dict[str, Union[Any, List[Dict[str, Any]]]]]:
    unified: Dict[int, Dict[str, Union[Any, List[Dict[str, Any]]]]] = {}
    to_remove = set()
    for dial_id, dial in tqdm(dials.items(), desc="Unifying", unit="dialogue"):
        if dial_id in to_remove:
            continue
        unified[dial_id] = dial
        for match_id, match in dials.items():
            if match_id == dial_id:
                continue
            if match_id in to_remove:
                continue
            if len(dial["turns"]) < len(match["turns"]):
                continue
            if is_match(dial["turns"], match["turns"]):
                to_remove.add(match_id)
                unified[dial_id] = unify(dial, match)
    for dial_id in to_remove:
        if dial_id in unified:
            del unified[dial_id]
    return unified


def format_dials(
    dials: Dict[int, Tuple[Any, Any]]
) -> Dict[int, Dict[str, Union[Any, List[Dict[str, Any]]]]]:
    # dict of dialogues
    # each dialogue is a dictionary containing a list of turns "turns"
    formatted: Dict[int, Dict[str, Union[Any, List[Dict[str, Any]]]]] = {}
    c = 0
    sources = Counter()
    for dial_id, dial in dials.items():
        if dial_id not in formatted:
            formatted[dial_id] = {"turns": []}
        target_avail = False
        target_label = {}
        target_turn = -1
        for log, label in dial:
            # check that all saved turns are in log of next turn
            for t_id, turn in enumerate(formatted[dial_id]["turns"]):
                assert log[t_id]["text"] == turn["text"]
            # append all turns after the already saved ones
            start_turn = len(formatted[dial_id]["turns"])
            formatted[dial_id]["turns"].extend(log[start_turn:])
            if target_avail:
                target_avail = False
                # in turn after finding target
                formatted[dial_id]["turns"][target_turn]["knowledge"] = target_label[
                    "knowledge"
                ]
                # check that the response is the same
                assert formatted[dial_id]["turns"][target_turn]["speaker"] == "S"
                assert (
                    formatted[dial_id]["turns"][target_turn]["text"]
                    == target_label["response"]
                )
            if label["target"]:
                assert log[-1]["speaker"] == "U"
                target_avail = True
                target_label = label
                target_turn = len(formatted[dial_id]["turns"])
                if "source" in label:
                    sources.update([label["source"]])
                    formatted[dial_id]["source"] = label["source"]
                else:
                    sources.update(["no_source"])
        # check if dialogues end without using the knowledge and response provided at the previous turn
        if target_avail:
            c += 1
            # dialogue ended without turn with system response
            formatted[dial_id]["turns"].append(
                {
                    "speaker": "S",
                    "text": target_label["response"],
                    "knowledge": target_label["knowledge"],
                }
            )
    print("added sys turns:", c)
    print("sources:", sources)
    return formatted


def load_dstc9_train(split_folder: str, split_root: str) -> Dict[int, Tuple[Any, Any]]:
    ds = DatasetWalker(split_folder, split_root, labels=True)
    ds = [(log, label) for log, label in ds]
    dials = {}
    prev_len = 0
    dial_id = -1
    for i, (log, label) in enumerate(ds):
        curr_len = len(log)
        new_dial = (
            (curr_len < prev_len and curr_len == 1)
            or i == 0
            or (prev_len == 1 and curr_len == 1)
        )
        assert curr_len > prev_len or new_dial
        prev_len = curr_len
        if curr_len == 1:
            dial_id += 1
            dials[dial_id] = []
        dials[dial_id].append((log, label))
    return dials


def load_dstc9_test(split_folder: str, split_root: str) -> Dict[int, Tuple[Any, Any]]:
    ds = DatasetWalker(split_folder, split_root, labels=True)
    ds = [(log, label) for log, label in ds]
    dials = {}
    prev_len = float("Inf")
    dial_id = -1
    for log, label in ds:
        curr_len = len(log)
        new_dial = curr_len < prev_len
        if new_dial:
            dial_id += 1
            dials[dial_id] = []
        dials[dial_id].append((log, label))
    return dials


def load_dstc9() -> Dict[str, Dict[int, Dict[str, Union[Any, List[Dict[str, Any]]]]]]:
    split_folders = {
        "train": ("./original_data/DSTC9/data/", "train"),
        "dev": ("./original_data/DSTC9/data/", "val"),
        "test": ("./original_data/DSTC9/data_eval/", "test"),
    }
    dialogues = {}
    for split in split_folders.keys():
        print(split)
        split_root, split_folder = split_folders[split]
        if split != "test":
            dials = load_dstc9_train(split_folder, split_root)
        else:
            dials = load_dstc9_test(split_folder, split_root)

        dials = format_dials(dials)
        # unify partial dialogues
        dials = unify_dials(dials)
        print(f"Dials: {len(dials)}")
        dialogues[split] = dials
        print("----")
    return dialogues


def load_mwoz21_split_ids(ids_path: str) -> Set[str]:
    ids = set()
    with open(ids_path, "r") as f:
        ids.update([l.strip("\n") for l in f.readlines()])
    return ids


def format_mwoz21_ds(ds: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    formatted: Dict[str, str] = {}
    for domain, dom_state in ds.items():
        domain = domain.lower().strip()
        for slot_type in ["book", "semi"]:
            book = "book" if slot_type == "book" else ""
            for slot, value in dom_state[slot_type].items():
                slot = slot.lower().strip()
                if slot == "booked":
                    continue
                if value.lower().strip() not in ["", "not mentioned", "none"]:
                    slot_name = f"{domain}-{book}{slot}"
                    formatted[slot_name] = value
    return formatted


def load_mwoz21() -> Dict[str, Dict[str, Dict[str, Union[Any, List[Dict[str, Any]]]]]]:
    dialogues = {split: {} for split in ["train", "dev", "test"]}
    data_path = "./original_data/MultiWOZ_2.1/data.json"
    dev_ids_path = "./original_data/MultiWOZ_2.1/valListFile.txt"
    test_ids_path = "./original_data/MultiWOZ_2.1/testListFile.txt"
    with open(data_path, "r") as f:
        raw_data = json.load(f)
    dev_ids = load_mwoz21_split_ids(dev_ids_path)
    test_ids = load_mwoz21_split_ids(test_ids_path)

    for dial_id, raw_dial in raw_data.items():
        split = "train"
        if dial_id in dev_ids:
            split = "dev"
        elif dial_id in test_ids:
            split = "test"
        dial = {"turns": []}
        for raw_turn in raw_dial["log"]:
            turn = {
                "text": raw_turn["text"],
                "dialogue_state": format_mwoz21_ds(raw_turn["metadata"]),
                "dialogue_act": {},
            }
            if "dialog_act" in raw_turn:
                turn["dialogue_act"] = raw_turn["dialog_act"]
            dial["turns"].append(turn)
        dialogues[split][dial_id] = dial

    return dialogues


def print_turns(turns, start: int = 0):
    for t_id, turn in enumerate(turns[start:]):
        print(f"{start + t_id} | {turn['speaker']}: '{turn['text']}'")


def matches_with_mwoz(
    turns: List[Dict[str, Any]],
    mw_turns: List[Dict[str, Any]],
    test_split: bool = False,
    special: bool = False,
) -> Tuple[bool, List[str]]:
    matched = True
    t_id = 0
    tot_skip = 0
    edits = ["I"] * len(turns)
    partial = False
    for mw_t_id, mw_turn in enumerate(mw_turns):
        while t_id + 1 < len(turns) and "knowledge" in turns[t_id + 1]:
            t_id += 2
            tot_skip += 2
        if t_id >= len(turns):
            break

        # skip non-matching turns (might be extra from dstc9 but missing knowledge)
        after_match = False
        while True:
            if not test_split:
                cont = (
                    t_id < len(turns)
                    and turns[t_id]["text"].lower().strip()
                    != mw_turn["text"].lower().strip()
                )
            elif not special:
                cont = t_id < len(turns) and re.sub(
                    r"\s\s", r" ", turns[t_id]["text"].lower().strip()
                ) != re.sub(r"\s\s", r" ", mw_turn["text"].lower().strip())
            else:
                cont = t_id < len(turns) and re.sub(
                    r"\s*", r" ", turns[t_id]["text"].lower().strip()
                ) != re.sub(r"\s*", r" ", mw_turn["text"].lower().strip())
            if cont:
                t_id += 1
            else:
                break
        if t_id < len(turns):
            partial = True
            after_match = True
            edits[t_id] = "="

        if not after_match:
            matched = False
            break
        t_id += 1

    edits_counter = Counter(edits)
    if not test_split:
        assert "D" not in edits_counter
    # TODO check with > 2
    if not matched and partial and edits_counter["="] > 1:
        matched = True
        edits.extend(["D"] * (len(mw_turns) - mw_t_id))
        assert edits_counter["D"] < edits_counter["="] + edits_counter["I"]
    else:
        assert len(turns) == len(edits)
    return matched, edits


def patch_dial(
    dial: Dict[str, Union[Any, List[Dict[str, Any]]]],
    mw_split: str,
    mw_dial: Dict[str, Union[Any, List[Dict[str, Any]]]],
    mw_dial_id: str,
    edits: List[str],
    test_split: bool = False,
    special: bool = False,
) -> Dict[str, Union[Any, List[Dict[str, Any]]]]:
    patched = deepcopy(dial)
    patched["mwoz21_split"] = mw_split
    mw_t_id = 0
    for (t_id, turn), edit in zip(enumerate(dial["turns"]), edits):
        if edit == "I":
            # dstc9 added turn
            if turn["speaker"] == "S":
                if "knowledge" not in turn:
                    # mark turns where knowledge is missing
                    patched["turns"][t_id]["knowledge"] = None
                # propagate dialogue state from previous turns
                if t_id - 2 > 0:
                    assert patched["turns"][t_id - 2]["speaker"] == "S"
                    patched["turns"][t_id]["dialogue_state"] = deepcopy(
                        patched["turns"][t_id - 2]["dialogue_state"]
                    )
                else:
                    # no previous turns
                    patched["turns"][t_id]["dialogue_state"] = {}
            else:  # user turn
                patched["turns"][t_id]["dialogue_state"] = {}
        elif edit == "=":
            if not test_split:
                assert (
                    turn["text"].lower().strip()
                    == mw_dial["turns"][mw_t_id]["text"].lower().strip()
                )
            elif not special:
                assert re.sub(r"\s\s", r" ", turn["text"].lower().strip()) == re.sub(
                    r"\s\s", r" ", mw_dial["turns"][mw_t_id]["text"].lower().strip()
                )
            else:
                assert re.sub(r"\s*", r" ", turn["text"].lower().strip()) == re.sub(
                    r"\s*", r" ", mw_dial["turns"][mw_t_id]["text"].lower().strip()
                )
            patched["turns"][t_id]["dialogue_state"] = deepcopy(
                mw_dial["turns"][mw_t_id]["dialogue_state"]
            )
            patched["turns"][t_id]["dialogue_act"] = deepcopy(
                mw_dial["turns"][mw_t_id]["dialogue_act"]
            )
            patched["turns"][t_id]["mw_dial_id"] = mw_dial_id
            mw_t_id += 1
        else:
            assert edit == "D"
    return patched


def add_ds_to_dstc9(
    dials: Dict[str, Dict[int, Dict[str, Union[Any, List[Dict[str, Any]]]]]],
    mwoz_dials: Dict[str, Dict[str, Dict[str, Union[Any, List[Dict[str, Any]]]]]],
) -> Dict[str, Dict[int, Dict[str, Union[Any, List[Dict[str, Any]]]]]]:
    patched: Dict[str, Dict[int, Dict[str, Union[Any, List[Dict[str, Any]]]]]] = {
        split: {} for split in ["train", "dev", "test"]
    }
    for split in dials.keys():
        test_split = split == "test"
        c = 0
        for dial_id, dial in tqdm(
            dials[split].items(), desc="Patching", unit="dialogues"
        ):
            special = False
            if test_split:
                # special = dial["turns"][0] in  ["I'm looking for a moderately priced restaurant serving cuban food.", "I will be visiting Cambridge and I need a list of the main attractions in the south.", "I'd like to find a hotel with free wifi and free parking, please.", "Hi, I am looking for an attraction called Queen's College. Can you give me some information about it please?"]
                special = dial_id in [104, 261, 1115, 2520]
            if test_split and ("source" not in dial or dial["source"] != "multiwoz"):
                # only have DS for multiwoz
                patched[split][dial_id] = deepcopy(dial)
                continue
            matched = False
            for mw_split in ["train", "dev", "test"]:
                for mw_dial_id, mw_dial in mwoz_dials[mw_split].items():
                    is_match, edits = matches_with_mwoz(
                        dial["turns"],
                        mw_dial["turns"],
                        test_split=test_split,
                        special=special,
                    )
                    if is_match:
                        matched = True
                        c += 1
                        patched[split][dial_id] = patch_dial(
                            dial,
                            mw_split,
                            mw_dial,
                            mw_dial_id,
                            edits,
                            test_split=test_split,
                            special=special,
                        )
                        break
                if matched:
                    break
        print(split, c)
    return patched


def prepare_kb(kb: Dict[str, dict]) -> Dict[str, Dict[tuple, dict]]:
    prepared: Dict[str, Dict[tuple, dict]] = {split: {} for split in kb}
    for split, kb_split in kb.items():
        for domain, entities in kb_split.items():
            for ent_id, entity in entities.items():
                # skip entities not from multiwoz
                if "city" in entity and entity["city"] != "Cambridge":
                    continue
                ent_name = entity["name"]
                if ent_name is None:
                    assert domain in ["taxi", "train"]
                    ent_name = domain.upper()
                for doc_id, doc in entity["docs"].items():
                    prepared[split][(domain, ent_id, doc_id)] = {
                        "ent_name": ent_name,
                        "doc": doc,
                    }
    return prepared


def get_list_and_ids(
    prep_kb: Dict[str, Dict[tuple, dict]]
) -> Tuple[Dict[str, List[str]], Dict[str, List[Dict[str, str]]]]:
    kb_list: Dict[str, List[str]] = {}
    kb_ids: Dict[str, List[Dict[str, str]]] = {}

    for split, kb_split in prep_kb.items():
        kb_tuple = [
            # ({"id": "_".join([domain, ent_id, doc_id])}, f"{doc['ent_name']}, {doc['doc']['title']} {doc['doc']['body']}")
            (
                {"domain": domain, "entity_id": ent_id, "doc_id": doc_id},
                f"{doc['ent_name']}, {doc['doc']['title']} {doc['doc']['body']}",
            )
            for (domain, ent_id, doc_id), doc in kb_split.items()
        ]
        kb_ids[split], kb_list[split] = list(zip(*kb_tuple))
        kb_ids[split] = list(kb_ids[split])
        kb_list[split] = list(kb_list[split])

    return kb_list, kb_ids


def add_top_k_retrieved_docs(
    dials: Dict[str, Dict[int, Dict[str, Union[Any, List[Dict[str, Any]]]]]],
    vector_stores: Dict[str, FAISS],
    k: int = 5,
) -> Dict[str, Dict[int, Dict[str, Union[Any, List[Dict[str, Any]]]]]]:
    for split in dials:
        tot = 0
        n_found = 0
        for dial_id, dial in tqdm(
            dials[split].items(), desc=f"Retrieving top-{k}", unit="dialogue"
        ):
            hist = []
            for turn in dial["turns"]:
                if turn["speaker"] == "U":
                    hist.append(f"User: {turn['text']}")
                    continue
                if "knowledge" not in turn or turn["knowledge"] == None:
                    continue

                tot += 1

                top_k = query_vector_store(" ".join(hist), vector_stores["train"], k)
                # top_k = query_vector_store(" ".join(hist[-min(len(hist), 4):]), vector_stores["train"], k)
                turn["top_k"] = [res[0].metadata for res in top_k]

                # append system turn after
                hist.append(f"System: {turn['text']}")

                found = False
                for res in turn["top_k"]:
                    if (
                        res["domain"] == turn["knowledge"][0]["domain"]
                        and res["entity_id"] == str(turn["knowledge"][0]["entity_id"])
                        and str(res["doc_id"] == turn["knowledge"][0]["doc_id"])
                    ):
                        found = True
                if found:
                    n_found += 1
        print("-" * 10 + f" {split} " + "-" * 10)
        print(f"tot: {tot}")
        print(f"found: {n_found}")
        print(f"%: {round(n_found/tot*100, 3)}")
    return dials


def add_retrieval(
    dials: Dict[str, Dict[int, Dict[str, Union[Any, List[Dict[str, Any]]]]]]
) -> Dict[str, Dict[int, Dict[str, Union[Any, List[Dict[str, Any]]]]]]:
    # load kb
    kb: Dict[str, dict] = {}
    for split, unstr_kb_path in zip(
        ["train", "test"],
        ["./original_data/DSTC9/data/", "./original_data/DSTC9/data/"],
    ):
        with open(os.path.join(unstr_kb_path, "knowledge.json")) as f:
            kb_split = json.load(f)
        kb[split] = kb_split
    # prepare kb
    prep_kb: Dict[str, Dict[tuple, dict]] = prepare_kb(kb)
    kb_list, kb_ids = get_list_and_ids(prep_kb)

    vector_stores: Dict[str, FAISS] = {}
    for split in ["train", "test"]:
        vector_stores[split] = get_vector_store(kb_list[split], metadatas=kb_ids[split])

    dials = add_top_k_retrieved_docs(dials, vector_stores)
    return dials


def main():
    dials = load_dstc9()
    mwoz_dials = load_mwoz21()
    dials = add_ds_to_dstc9(dials, mwoz_dials)

    dials = add_retrieval(dials)

    save_path = "./data/DSTC9"
    os.makedirs(save_path, exist_ok=True)
    for split in dials:
        split_file = f"{split}.json"
        if split == "dev":
            split_file = "valid.json"
        with open(os.path.join(save_path, split_file), "w") as f:
            json.dump(dials[split], f, indent=4)


if __name__ == "__main__":
    main()
