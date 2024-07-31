import csv
import json
import os

from tqdm import tqdm

from retriever import get_vector_store, query_vector_store

TOTAL_SUMMARIES = 1572
TOTAL_QUESTIONS = 46765


def prepare_narrative_qa(data_dir, output_dir, top_k: int):

    titles = {}
    with open(os.path.join(data_dir, "documents.csv"), "r") as f:
        documents = csv.DictReader(f)
        for doc in tqdm(documents, total=TOTAL_SUMMARIES, desc="Reading documents"):
            document_id = doc["document_id"]
            title = doc["wiki_title"]
            titles[document_id] = title

    kb = {}
    texts_to_index = {}
    document_ids = {}
    with open(os.path.join(data_dir, "summaries.csv"), "r") as f:
        summaries = csv.DictReader(f)
        for summary in tqdm(summaries, total=TOTAL_SUMMARIES, desc="Reading summaries"):
            document_id = summary["document_id"]
            split = summary["set"]
            if split not in kb:
                kb[split] = {}
                texts_to_index[split] = []
                document_ids[split] = []
            # store the summary
            kb[split][document_id] = {
                "summary": summary["summary"],
                "title": titles[document_id],
            }
            # store the document id
            document_ids[split].append({"document_id": document_id})
            # concatenate title and summary
            texts_to_index[split].append(f"{titles[document_id]}, {summary['summary']}")

    # create the vector stores
    vectors = {}
    print("Creating vector stores...")
    for split, texts in texts_to_index.items():
        vectors[split] = get_vector_store(texts, metadatas=document_ids[split])

    questions = {}
    with open(os.path.join(data_dir, "qaps.csv"), "r") as f:
        qaps = csv.DictReader(f)
        for qap in tqdm(qaps, total=TOTAL_QUESTIONS, desc="Reading questions"):
            document_id = qap["document_id"]
            question = qap["question"]
            split = qap["set"]
            answer1 = qap["answer1"]
            answer2 = qap["answer2"]

            if split not in questions:
                questions[split] = {
                    "count": 0,
                    "questions": [],
                }

            questions[split]["questions"].append(
                {
                    "question": question,
                    "answers": [answer1, answer2],
                    "question_id": questions[split]["count"],
                    "document_id": document_id,
                }
            )
            questions[split]["count"] += 1

    datasets = {}
    print("Querying vector stores...")
    for split, q in questions.items():
        for qap in tqdm(q["questions"], total=q["count"], desc=f"Querying {split}"):
            question = qap["question"]
            results = query_vector_store(question, vectors[split], top_k)

            qap["retrieved_summaries"] = []
            for doc, score in results:
                qap["retrieved_summaries"].append(
                    {
                        "summary_id": doc.metadata["document_id"],
                        "score": score.item(),  # type: ignore
                    }
                )

        datasets[split] = q

    os.makedirs(output_dir, exist_ok=True)

    for split, q in datasets.items():
        with open(
            os.path.join(output_dir, f"{split}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(q, f, indent=4)

        with open(
            os.path.join(output_dir, f"{split}_kb.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(kb[split], f, indent=4)


if __name__ == "__main__":
    prepare_narrative_qa(
        data_dir="original_data/NarrativeQA",
        output_dir="data/NarrativeQA",
        top_k=5,
    )
