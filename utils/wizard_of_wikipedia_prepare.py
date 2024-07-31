import json
import os
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
from langchain_core.documents import Document

from retriever import get_vector_store, query_vector_store


def get_gold_passage(turn: dict) -> Tuple[Optional[str], Optional[str]]:
    if len(turn["checked_sentence"]) == 0:
        return None, None
    key, sentence = next(iter(turn["checked_sentence"].items()))
    if key == "no_passages_used":
        return None, None
    if len(turn["checked_passage"]) == 0:
        topic = " ".join(key.split("_")[1:-1])
    else:
        topic = next(iter(turn["checked_passage"].values()))

    return topic, sentence


def get_retrieved_passages(turn: dict) -> Dict[str, List[str]]:
    retrieved_passages = {}
    for passages_per_topic in turn["retrieved_passages"]:
        for topic, passages in passages_per_topic.items():
            retrieved_passages[topic] = passages

    return retrieved_passages


def get_self_and_partner_passages(
    passages: List[Dict[str, List[str]]]
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:

    if len(passages) == 0:
        self_passages, partner_passages = {}, {}
    elif len(passages) == 1:
        self_passages, partner_passages = {}, passages[-1]
    else:
        self_passages, partner_passages = passages[-2], passages[-1]

    return self_passages, partner_passages


def check_if_gold_matches(
    sentence: str,
    chosen_topic: str,
    chosen_topic_passage: List[str],
    self_passages: Dict[str, List[str]],
    partner_passages: Dict[str, List[str]],
) -> Optional[str]:

    if sentence is None:
        return None

    if sentence in chosen_topic_passage:
        return chosen_topic

    for topic, passage in self_passages.items():
        if sentence in passage:
            return topic

    for topic, passage in partner_passages.items():
        if sentence in passage:
            return topic

    raise ValueError("Gold sentence not in retrieved passages")


def retrieve_sentences_based_on_dialogue_history(
    dialogue_history: List[str],
    chosen_topic: str,
    chosen_topic_passage: List[str],
    self_passages: Dict[str, List[str]],
    partner_passages: Dict[str, List[str]],
    top_k: int,
) -> List[Tuple[Document, float]]:
    # prepend the chosen topic to the topic passages
    sentence_set = set([(chosen_topic, s) for s in chosen_topic_passage])

    # prepend the chosen topic to the self and partner passages
    sentence_set.update(
        [(topic, s) for topic, sentences in self_passages.items() for s in sentences]
    )
    sentence_set.update(
        [(topic, s) for topic, sentences in partner_passages.items() for s in sentences]
    )

    vector_store = get_vector_store(
        [f"{t}, {s}" for t, s in sentence_set],
        silent=True,
        metadatas=[{"topic": t, "sentence": s} for t, s in sentence_set],
    )
    dh_roles = []
    topic, dialogue_history = dialogue_history[0], dialogue_history[1:]
    if len(dialogue_history) % 2 == 0:
        dh_roles = ["System", "User"]
    else:
        dh_roles = ["User", "System"]

    # create the query using the dialogue history
    query = [f"Topic: {topic}"]
    for i, turn in enumerate(dialogue_history):
        query.append(f"{dh_roles[i % 2]}: {turn}")
    query = "\n".join(query)

    return query_vector_store(query, vector_store, top_k)


def prepare_wizard_of_wikipedia(data_dir, output_dir, top_k=5):

    # create the output folder
    os.makedirs(output_dir, exist_ok=True)

    for split, filename in zip(
        ["train", "valid", "test"],
        ["train.json", "valid_topic_split.json", "test_topic_split.json"],
    ):
        with open(os.path.join(data_dir, filename), "r") as f:
            data = json.load(f)

        dataset = {}
        discarded_turns = {}
        for dialogue_id, dialogue in enumerate(
            tqdm(data, desc=f"Processing {split} dialogues")
        ):
            chosen_topic = dialogue["chosen_topic"]
            chosen_topic_passage = dialogue["chosen_topic_passage"]
            dialoue_history = [
                chosen_topic  # add the chosen topic as first turn like in the WoW paper
            ]
            passages = []
            for turn_id, turn in enumerate(dialogue["dialog"]):
                text = turn["text"]
                if turn["speaker"].endswith("Wizard"):
                    try:
                        gold_topic, sentence = get_gold_passage(turn)

                        self_passages, partner_passages = get_self_and_partner_passages(
                            passages
                        )

                        # gold_topic = check_if_gold_matches(
                        #     sentence,  # type: ignore
                        #     chosen_topic,
                        #     chosen_topic_passage,
                        #     self_passages,
                        #     partner_passages,
                        # )

                        retrieved_sentences = (
                            retrieve_sentences_based_on_dialogue_history(
                                dialoue_history,
                                chosen_topic,
                                chosen_topic_passage,
                                self_passages,
                                partner_passages,
                                top_k,
                            )
                        )

                        dataset[f"dial_{dialogue_id}_turn_{turn_id}"] = {
                            "dialogue_id": dialogue_id,
                            "turn_id": turn_id,
                            "speaker": "wizard",
                            "dialogue_history": dialoue_history.copy(),
                            "text": text,
                            "gold_sentence": {
                                "sentence": sentence,
                                "topic": gold_topic,
                            },
                            "chosen_topic": chosen_topic,
                            # "chosen_topic_passage": chosen_topic_passage,
                            # "partner_passages": partner_passages,
                            # "self_passages": self_passages,
                            "retrieved_sentences": [
                                {
                                    "sentence": doc.metadata["sentence"],
                                    "topic": doc.metadata["topic"],
                                    "score": score.item(),  # type: ignore
                                }
                                for doc, score in retrieved_sentences
                            ],
                        }

                    except ValueError as e:
                        discarded_turns[f"dial_{dialogue_id}_turn_{turn_id}"] = str(e)

                # append the passages for the current turn and use them for the next one
                passages.append(get_retrieved_passages(turn))
                # append the text to the dialogue history
                dialoue_history.append(text)

        with open(os.path.join(output_dir, f"{split}.json"), "w") as f:
            json.dump(dataset, f, indent=4)

        with open(os.path.join(output_dir, f"{split}_discarded_turns.json"), "w") as f:
            json.dump(discarded_turns, f, indent=4)

        total_turns = len(dataset)
        discarded_turns_count = len(discarded_turns)
        percentage_discarded_turns = (
            discarded_turns_count / (total_turns + discarded_turns_count)
        ) * 100
        print(f"Percentage of discarded turns: {percentage_discarded_turns}%")


if __name__ == "__main__":
    prepare_wizard_of_wikipedia(
        "original_data/WizardOfWikipedia", "data/WizardOfWikipedia", top_k=5
    )
