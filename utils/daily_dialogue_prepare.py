import json
import os

from tqdm import tqdm


def preprocess_dialogues(file_name):
    dialogues = []
    with open(file_name) as f:
        for line in f:
            turns = line.split("__eou__")
            dialogues.append(
                {
                    i: {
                        "text": turn.strip(),
                        "speaker": "user" if i % 2 == 0 else "system",
                    }
                    for i, turn in enumerate(turns[:-1])
                }
            )
    return dialogues


def prepare_dataset(dialogues, file_name):
    dataset = {}
    for dial_idx, dial in enumerate(tqdm(dialogues, desc=f"Preparing {file_name}")):
        dial_history = []
        for turn_idx, turn in dial.items():
            if turn["speaker"] == "system":
                dataset[f"dial_{dial_idx}_turn_{turn_idx}"] = {
                    "dialogue_id": dial_idx,
                    "turn_id": turn_idx,
                    "speaker": turn["speaker"],
                    "text": turn["text"],
                    "dial_history": dial_history.copy(),
                }
            dial_history.append(turn["text"])

    for sample in dataset.values():
        assert sample["speaker"] == "system", "Evaluating user responses"

    with open(file_name, "w") as f:
        json.dump(dataset, f, indent=2)


if __name__ == "__main__":
    train_dialogues = preprocess_dialogues(
        "original_data/ijcnlp_dailydialog/train/dialogues_train.txt"
    )
    valid_dialogues = preprocess_dialogues(
        "original_data/ijcnlp_dailydialog/validation/dialogues_validation.txt"
    )
    test_dialogues = preprocess_dialogues(
        "original_data/ijcnlp_dailydialog/test/dialogues_test.txt"
    )

    os.makedirs("data/DailyDialog", exist_ok=True)
    prepare_dataset(train_dialogues, "data/DailyDialog/train.json")
    prepare_dataset(valid_dialogues, "data/DailyDialog/valid.json")
    prepare_dataset(test_dialogues, "data/DailyDialog/test.json")
