import json
import math
import random
import re
from typing import Any, Dict, List, Set, Tuple
from pathlib import Path

import numpy as np
import plotly.express as px
import torch
from nltk.tokenize import sent_tokenize
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm


def collect_task_replies(task_folder: str) -> dict:
    task_replies = {}
    task = Path(task_folder)
    for model in task.iterdir():
        if model.is_dir():
            for elem in model.iterdir():
                if elem.is_file():
                    if elem.name.startswith("generation_results_"):
                        with open(elem, "r") as f:
                            replies = json.load(f)["results"]
                        for sample_id, reply in replies.items():
                            if sample_id not in task_replies:
                                task_replies[sample_id] = {
                                    "gt": reply["target"],
                                }
                            task_replies[sample_id][
                                f'{model.name}_prompt_{elem.name.split("generation_results_")[-1].split(".json")[0]}'
                            ] = reply["output"]
                elif elem.is_dir():
                    for file in elem.iterdir():
                        if file.is_file():
                            if file.name.startswith("generation_results_"):
                                if "_".join(elem.name.split("_")[:-1]) in file.name:
                                    with open(file, "r") as f:
                                        replies = json.load(f)["results"]
                                    for sample_id, reply in replies.items():
                                        if sample_id not in task_replies:
                                            task_replies[sample_id] = {
                                                "gt": reply["target"],
                                            }
                                        task_replies[sample_id][
                                            f'{model.name}_ft_{file.name.split("generation_results_")[-1].split(".json")[0]}'
                                        ] = reply["output"]

    return task_replies


def fix_replies_if_necessary(
    replies: Dict[str, str],
    max_tokens: int,
    max_sentences: int,
    max_characters: int = 20,
    max_candidate_length: int = 100,  # n tokens measured on the generated answers to avoid discarding too many sentences
):
    for model, reply in replies.items():
        if model != "gt":
            # remove lists, i.e. 1. ... 2. ... 3. ...
            reply = re.sub(r"[0-9]+\.", "", reply)
            # remove lists, i.e. 1) ... 2) ... 3) ...
            reply = re.sub(r"[0-9]+\)", "", reply)
            # remove utf-8 characters
            reply = re.sub(r"[^\x00-\x7F]+", "", reply)

            if len(reply.split()) > max_tokens:
                # if the reply is longer than max_tokens, take at most max_sentences that (cumulatively) are under max_tokens
                sents = sent_tokenize(reply)[:max_sentences]
                # add at least the first sentence
                reply = sents[0] + " "
                for sent in sents[1:]:
                    if (len(reply.split()) + len(sent.split())) <= max_tokens:
                        reply += sent + " "
                    else:
                        break

            # remove newlines
            reply = re.sub(r"\n", " ", reply)

            # remove spaces at the beginning and end of the reply
            reply = reply.strip()

            # remove multiple spaces
            reply = re.sub(r"\s+", " ", reply)

            if len(reply.split()) > max_candidate_length:
                raise ValueError(
                    f"Reply is still longer than {max_candidate_length} words: {reply}"
                )

            if len(reply) == 0 or len(reply.split()) == 0:
                raise ValueError(f"Reply is empty: {reply}")

            if any([len(token) > max_characters for token in reply.split()]):
                raise ValueError(
                    f"Reply contains tokens with more than {max_characters} characters: {reply}"
                )

        replies[model] = reply

    return replies


def discard_samples_with_highest_bleu(
    replies: Dict[str, str],
    bleu_threshold: float,
):
    bleu_scores = {}
    for model, reply in replies.items():
        references = [
            reference.split()
            for reference_type, reference in replies.items()
            if reference_type != model
        ]
        assert len(references) == len(replies) - 1, "Invalid references."
        bleu = sentence_bleu(
            references, reply.split(), weights=(0.25, 0.25, 0.25, 0.25)
        )
        bleu_scores[model] = bleu

        assert bleu <= bleu_threshold, f"BLEU score is higher than {bleu_threshold}."  # type: ignore

    return bleu_scores


def get_candidates_per_batch(
    candidates: List[str], n_candidates_per_batch: int
) -> List[List[str]]:

    models = [model for model in candidates if model != "gt"]
    random.shuffle(models)

    candidates_per_batch = []
    model_set = set()

    for i in range(0, len(models), n_candidates_per_batch):
        batch = []
        for model in models[i : i + n_candidates_per_batch]:
            assert model not in model_set, f"Model {model} is already in the batch"
            batch.append(model)
            model_set.add(model)
        candidates_per_batch.append(batch)

    return candidates_per_batch


def compute_cross_bleu(
    samples: Dict[str, Dict[str, Any]],
    candidates: List[str],
    diag_to_zero: bool = False,
    operation: str = "mean",
):

    config_num = len(candidates)
    cross_bleu2_tok_matrix = np.zeros((config_num, config_num))
    cross_bleu4_tok_matrix = np.zeros((config_num, config_num))

    for sample in tqdm(samples.values(), desc="Computing cross BLEU"):
        for i in range(len(candidates)):
            for j in range(len(candidates)):
                # Token level bleu
                bleu2_tok_temp = sentence_bleu(
                    [sample[candidates[i]].split(" ")],
                    sample[candidates[j]].split(" "),
                    weights=(0.5, 0.5, 0, 0),
                )
                bleu4_tok_temp = sentence_bleu(
                    [sample[candidates[i]].split(" ")],
                    sample[candidates[j]].split(" "),
                    weights=(0.25, 0.25, 0.25, 0.25),
                )
                # Update bleu matrices
                if operation == "mean":
                    cross_bleu2_tok_matrix[i, j] += bleu2_tok_temp
                    cross_bleu4_tok_matrix[i, j] += bleu4_tok_temp
                elif operation == "max":
                    cross_bleu2_tok_matrix[i, j] = max(
                        cross_bleu2_tok_matrix[i, j], bleu2_tok_temp
                    )
                    cross_bleu4_tok_matrix[i, j] = max(
                        cross_bleu4_tok_matrix[i, j], bleu4_tok_temp
                    )

    if operation == "mean":
        cross_bleu2_tok_matrix /= len(samples)
        cross_bleu4_tok_matrix /= len(samples)

    if diag_to_zero:
        for i in range(cross_bleu2_tok_matrix.shape[0]):
            cross_bleu2_tok_matrix[i, i] = 0
            cross_bleu4_tok_matrix[i, i] = 0

    return (
        cross_bleu2_tok_matrix,
        cross_bleu4_tok_matrix,
    )


def modified_trunc(values, decs=3):
    return np.trunc(values * 10**decs) / (10**decs)


def create_2D_Heatmaps(
    bleu_dict: List[Dict[str, Any]],
    candidates: List[str],
    configs: Dict[str, Any] = {},
):

    font_size = configs.get("font_size", 15)
    x_tickangle = configs.get("x_tickangle", -45)
    height = configs.get("height", 600)
    width = configs.get("width", 600)
    text_auto = configs.get("text_auto", True)

    for data in bleu_dict:
        fig = px.imshow(
            modified_trunc(data["results"]),
            x=candidates,
            y=candidates[::-1],
            title=data["title"],
            text_auto=text_auto,
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(tickangle=x_tickangle)
        fig.update_layout(
            font_size=font_size,
            height=height,
            width=width,
        ).show()


def plot_cross_bleu(
    samples: Dict[str, Dict[str, Any]],
    candidates: List[str],
    replacements: Dict[str, str] = {},
    configs: Dict[str, Any] = {},
    results_to_plot: List[str] = ["b2_tok", "b4_tok"],
    diag_to_zero: bool = True,
    operation: str = "mean",
):

    new_samples: Dict[str, Dict[str, Any]] = {}
    for idx, sample in samples.items():
        new_samples[idx] = {c: sample[c] for c in candidates}

    b2_tok, b4_tok = compute_cross_bleu(
        new_samples, candidates, diag_to_zero=diag_to_zero, operation=operation
    )

    replaced_candidates: List[str] = []
    for c in candidates:
        for old, new in replacements.items():
            c = c.replace(old, new)
        replaced_candidates.append(c)

    candidates = replaced_candidates

    # reverse the order for plotting
    b2_tok, b4_tok = (
        b2_tok[::-1, :],
        b4_tok[::-1, :],
    )

    bleu_dict = []
    for result in results_to_plot:
        if result == "b2_tok":
            bleu_dict.append(
                {
                    "results": b2_tok,
                    "title": f"Bleu-2 token level ({operation})",
                }
            )
        elif result == "b4_tok":
            bleu_dict.append(
                {
                    "results": b4_tok,
                    "title": f"Bleu-4 token level ({operation})",
                }
            )

    create_2D_Heatmaps(bleu_dict, candidates, configs)


def get_samples_above_threshold(
    samples: Dict[str, Dict[str, Any]],
    candidates: List[str],
    threshold: float,
) -> Set[str]:
    samples_above_threshold = set()
    for sample_id, sample in tqdm(samples.items()):
        for i in candidates:
            for j in candidates:
                if i != j:
                    score = sentence_bleu(
                        [sample[i].split(" ")],
                        sample[j].split(" "),
                        weights=(0.25, 0.25, 0.25, 0.25),
                    )

                    if score >= threshold:  # type: ignore
                        samples_above_threshold.add(sample_id)
                        break

    return samples_above_threshold


def compute_average_attribution(
    int_grad: Dict[str, Dict[str, Any]],
    top_k_percentage: float,
    tokens_to_remove: List[int],
    tokens_to_find: List[List[int]],
    prompt_to_find: List[List[int]] = [],
) -> Tuple[np.ndarray, np.ndarray]:
    top_k_scores = []
    scores = []
    for sample_id, sample in int_grad.items():

        idx_to_ignore = []
        attribution = []
        for i in range(len(sample["attribution"].source)):
            # save which tokens to ignore
            if sample["attribution"].source[i].id in tokens_to_remove:
                idx_to_ignore.append(i)
            else:
                # add the remaning ones to the new attribution
                attribution.append(sample["attribution"].source[i])

        # create a mask for the indexes to be ignored
        mask = torch.ones(len(sample["attribution"].target_attributions))
        mask[idx_to_ignore] = 0
        mask = mask.bool()

        sample["attribution"].source = attribution
        sample["attribution"].target_attributions = sample[
            "attribution"
        ].target_attributions[mask]

        found_tokens = []
        # find the tokens in the attribution corresponding to the prompt
        for tokens in prompt_to_find:
            i = 0
            found = []
            for i in range(len(sample["attribution"].source)):
                tok = sample["attribution"].source[i]
                if tok.id == tokens[0]:
                    for j in range(1, len(tokens)):
                        if sample["attribution"].source[i + j].id == tokens[j]:
                            if j == len(tokens) - 1:
                                # append the next token which is not included in the prompt
                                found.append(j + 1)
                                i += j
                        else:
                            break

                if len(found) == 0:
                    i += 1

            if len(found) < 1:
                raise AssertionError(f"Token {tokens} not found in {sample_id}")
            elif len(found) > 1:
                raise AssertionError(f"Multiple tokens found for {tokens}")

            found_tokens.append(found[0])

        # find the tokens in the attribution corresponding to the segments
        for tokens in tokens_to_find:
            i = 0
            found = []
            for i in range(len(sample["attribution"].source)):
                tok = sample["attribution"].source[i]
                if tok.id == tokens[0]:
                    for j in range(1, len(tokens)):
                        if sample["attribution"].source[i + j].id == tokens[j]:
                            if j == len(tokens) - 1:
                                found.append(i)
                                i += j
                        else:
                            break

                if len(found) == 0:
                    i += 1

            if len(found) < 1:
                raise AssertionError(f"Token {tokens} not found in {sample_id}")
            elif len(found) > 1:
                raise AssertionError(f"Multiple tokens found for {tokens}")

            found_tokens.append(found[0])

        # append the last token for the remaining segment
        found_tokens.append(len(sample["attribution"].source))

        # get the average attribution across each generation step
        avg_attribution = (
            sample["attribution"]
            .target_attributions[: len(sample["attribution"].source)]
            .abs()
            .mean(1)
        )

        # keep track of the average attribution
        start = 0
        segment_scores = []
        for i, end in enumerate(found_tokens):
            segment_scores.append(avg_attribution[start:end].mean().item())
            start = end
        segment_scores = np.array(segment_scores)
        segment_scores = segment_scores / segment_scores.sum()
        scores.append(segment_scores)

        # extract the top k tokens with highest attribution
        n_input_elements = len(sample["attribution"].source)
        n_top_sample = math.ceil(top_k_percentage * n_input_elements)
        sorted_elem, indexes = torch.sort(avg_attribution, descending=True)
        sorted_elem = sorted_elem[:n_top_sample]
        indexes = indexes[:n_top_sample]

        start = 0
        segment_scores = []
        for i, end in enumerate(found_tokens):
            elems = []
            for elem, idx in zip(sorted_elem, indexes):
                if idx >= start and idx < end:
                    elems.append(elem)
            segment_scores.append(len(np.array(elems)) / (end - start))
            start = end
        segment_scores = np.array(segment_scores)
        segment_scores = segment_scores / segment_scores.sum()
        top_k_scores.append(segment_scores)

    scores = np.array(scores)
    top_k_scores = np.array(top_k_scores)

    return scores.mean(0), top_k_scores.mean(0)
