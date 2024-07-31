from collections import Counter
import os
import json


def main():
    """
    This function prints statistics about the DSTC9 dataset knowledge base.
    """
    for split, folder in zip(["train", "test"], ["data", "data_eval"]):
        print("-" * 10 + f" {split} " + "-" * 10)
        with open(
            os.path.join("./original_data/DSTC9/", folder, "knowledge.json")
        ) as f:
            kb = json.load(f)
        with open(f"./data/DSTC9/{split}.json") as f:
            train_set = json.load(f)
        doc_counter = Counter()
        ent_counter = Counter()
        n_docs = 0
        n_entities = 0
        cities = Counter()
        for domain, entities in kb.items():
            # ent_counter.update([domain] * len(entities))
            # n_entities += len(entities)
            for ent_id, entity in entities.items():
                if "city" in entity:
                    cities.update([entity["city"]])
                    if entity["city"] != "Cambridge":
                        continue
                # count only Cambridge entities
                ent_counter.update([domain])
                n_entities += 1
                doc_counter.update([domain] * len(entity["docs"]))
                n_docs += len(entity["docs"])
                # for doc_id, doc in entity["docs"].items():
        print(f"# docs in each domain: {doc_counter}")
        print(f"# entities in each domain: {ent_counter}")
        print(f"avg # docs per entity: {round(n_docs/n_entities, 3)}")
        print(f"cities: {cities}")

        n_known_name = 0
        n_unk_name = 0
        for dial_id, dial in train_set.items():
            if split == "test" and (
                "source" not in dial or dial["source"] != "multiwoz"
            ):
                continue
            for turn in dial["turns"]:
                if turn["speaker"] == "U":
                    continue
                if "knowledge" in turn and turn["knowledge"] is not None:
                    name_for = set()
                    for slot, value in turn["dialogue_state"].items():
                        domain, slot = slot.split("-")
                        if slot == "name":
                            name_for.add(domain)
                    dom = turn["knowledge"][0]["domain"]
                    ent_id = str(turn["knowledge"][0]["entity_id"])
                    ent_name = kb[dom][ent_id]["name"]
                    found = False
                    for domain in name_for:
                        try:
                            if (
                                ent_name is not None
                                and turn["dialogue_state"][f"{domain}-name"]
                                .lower()
                                .strip()
                                == ent_name.lower().strip()
                            ) or domain in ["train", "taxi"]:
                                found = True
                                break
                        except:
                            print(f"{domain}-name")
                            print(turn["dialogue_state"])
                            import pdb

                            pdb.set_trace()
                    if found:
                        n_known_name += 1
                    else:
                        n_unk_name += 1
        print(
            f"# knowledge turns where name is known: {n_known_name} ({n_known_name/(n_known_name+n_unk_name)})"
        )
        print(
            f"# knowledge turns where name is unknown: {n_unk_name} ({n_unk_name/(n_known_name+n_unk_name)})"
        )

    train_docs = set()
    test_docs = set()
    for split in ["train", "valid", "test"]:
        with open(f"./data/DSTC9/{split}.json") as f:
            data = json.load(f)
        for dial_id, dial in data.items():
            if split == "test" and (
                "source" not in dial or dial["source"] != "multiwoz"
            ):
                continue
            for turn in dial["turns"]:
                if turn["speaker"] == "U":
                    continue
                if "knowledge" not in turn or turn["knowledge"] is None:
                    continue
                docs = set()
                for entry in turn["knowledge"]:
                    docs.add((entry["domain"], entry["entity_id"], entry["doc_id"]))
                if split == "test":
                    test_docs.update(docs)
                else:
                    train_docs.update(docs)
    print(f"# train docs: {len(train_docs)}")
    print(f"# test docs: {len(test_docs)}")
    print(f"# train docs in test: {len(test_docs.intersection(train_docs))}")
    print(
        f"% train docs in test: {round(len(test_docs.intersection(train_docs))/len(test_docs)*100,3)}"
    )


if __name__ == "__main__":
    main()
