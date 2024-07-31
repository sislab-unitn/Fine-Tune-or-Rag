import os
import json
from typing import Any, Counter, Dict, List, Tuple, Union
from nltk.tokenize import word_tokenize
from dstc9_prepare import print_turns

def get_slots_counts(dials: Dict[int, Dict[str, Union[Any, List[Dict[str, Any]]]]]) -> Tuple[Counter, Counter, Counter]:
    domains = Counter()
    slots = Counter()
    values = Counter()
    for dial_id, dial in dials.items():
        for turn in dial["turns"]:
            domains.update([slot.split("-")[0] for slot in turn["dialogue_state"].keys()])
            slots.update(turn["dialogue_state"].keys())
            values.update(turn["dialogue_state"].values())
    return domains, slots, values

def get_splits_counts(dials: Dict[int, Dict[str, Union[Any, List[Dict[str, Any]]]]]) -> Counter:
    splits_c = Counter()
    for dial_id, dial in dials.items():
        splits_c.update([dial["mwoz21_split"]])
    return splits_c

def print_stats(dials):
    train_vocab = Counter()
    train_domains = Counter()
    train_slots = Counter()
    train_values = Counter()
    for split in dials.keys():
        print("-"*20 + f" {split} " + "-"*20)
        n_dials = len(dials[split])
        n_turns = sum([len(dial["turns"]) for dial in dials[split].values()])
        print(f"# dials: {n_dials}")
        print(f"# turns: {n_turns}")
        tokens = []
        for dial in dials[split].values():
            for turn in dial["turns"]:
                tokens.extend(word_tokenize(turn["text"]))
        print(f"# tokens: {len(tokens)}")
        vocab = Counter(tokens)
        print(f"# vocab: {len(vocab)}")
        if split == "train":
            train_vocab = vocab
        else:
            print(f"# oov: {len(set(vocab.keys()).difference(set(train_vocab.keys())))}")

        domains, slots, values = get_slots_counts(dials[split])
        print(f"# domains: {len(domains)}")
        print(f"# slots: {len(slots)}")
        print(f"values vocab size: {len(values)}")
        print(f"# values: {sum(values.values())}")
        if split == "train":
            train_domains = domains
            train_slots = slots
            train_values = values
        else:
            missing_doms = set(train_domains.keys()).difference(set(domains.keys()))
            unk_doms = set(domains.keys()).difference(set(train_domains.keys()))
            missing_slots = set(train_slots.keys()).difference(set(slots.keys()))
            print(f"# unk domains: {len(unk_doms)}")
            print(f"missing domains: {missing_doms}")
            print(f"# missing slots: {len(missing_slots)}")
            print(f"missing slots: {missing_slots}")
            print(f"# unk slots: {len(set(slots.keys()).difference(set(train_slots.keys())))}")
            print(f"# unk values: {len(set(values.keys()).difference(set(train_values.keys())))}")
        print(f"avg # slots per dial: {sum(slots.values())/n_dials}")
        print(f"avg # domains per dial: {sum(domains.values())/n_dials}")

        split_counts = get_splits_counts(dials[split])
        split_ratios = {s: round(counts/n_dials, 3) for s, counts in split_counts.items()}
        print(f"Ratios of dials from each original mwoz2.1 splits: {split_ratios}")


def remove_non_mwoz(test_dials: Dict[int, Dict[str, Union[Any, List[Dict[str, Any]]]]]):
    to_remove = set()
    for dial_id, dial in test_dials.items():
        if "source" not in dial or dial["source"] != "multiwoz":
            to_remove.add(dial_id)
    for dial_id in to_remove:
        del test_dials[dial_id]

def main():
    ds_path = "./data/DSTC9/"
    dials = {}
    for split in ["train", "valid", "test"]:
        with open(os.path.join(ds_path, f"{split}.json"), "r") as f:
            if split == "valid":
                split = "dev"
            dials[split] = json.load(f)

    remove_non_mwoz(dials["test"])
    print_stats(dials)

if __name__ == "__main__":
    main()
