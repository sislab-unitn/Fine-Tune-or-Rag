from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings


def get_vector_store(
    knowledge_base: List[str],
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    metadatas: Optional[List[dict]] = None,
    silent: bool = False,
) -> FAISS:
    # https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.huggingface.HuggingFaceEmbeddings.html
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
    )
    # https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html
    vector = FAISS.from_texts(knowledge_base, embeddings, metadatas=metadatas)
    if not silent:
        print(f"Vector store created with {vector.index.ntotal} indexes.")
    return vector


def query_vector_store(
    query: str, vector: FAISS, k: int
) -> List[Tuple[Document, float]]:
    return vector.similarity_search_with_score(query, k=k)
