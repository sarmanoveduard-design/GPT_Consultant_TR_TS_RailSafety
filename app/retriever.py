# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import requests
from dotenv import load_dotenv
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from sklearn.neighbors import NearestNeighbors
import numpy as np

from .utils import embed_texts, embed_query

DOC_LOCAL_PATH = "data/TR_TS_RailSafety.txt"


def fetch_doc_if_needed() -> str:
    """Скачивает документ как txt из Google Docs, если локального файла нет."""
    if os.path.exists(DOC_LOCAL_PATH):
        with open(DOC_LOCAL_PATH, "r", encoding="utf-8") as f:
            return f.read()

    load_dotenv()
    url = os.getenv("DOC_URL")
    if not url:
        raise RuntimeError("Не задан DOC_URL в .env")

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    text = resp.text
    os.makedirs("data", exist_ok=True)
    with open(DOC_LOCAL_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    return text


@dataclass
class VectorIndex:
    nn: NearestNeighbors         # обученный поисковик
    mat: np.ndarray              # матрица эмбеддингов (уже L2-нормированы)
    chunks: List[str]            # исходные текстовые куски


def build_chunks(text: str, mode: str = "rc") -> List[str]:
    """
    Разбивает текст двумя режимами:
    - mode='rc'  : RecursiveCharacterTextSplitter
    - mode='tok' : TokenTextSplitter
    """
    if mode == "rc":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, chunk_overlap=150, separators=["\n\n", "\n", " ", ""]
        )
    elif mode == "tok":
        splitter = TokenTextSplitter(chunk_size=450, chunk_overlap=50)
    else:
        raise ValueError("mode должен быть 'rc' или 'tok'")

    return [c for c in splitter.split_text(text) if c.strip()]


def build_faiss_index(chunks: List[str]) -> VectorIndex:
    """
    Название оставлено прежним, чтобы не менять остальной код.
    Строим индекс на sklearn.NearestNeighbors (косинусная метрика).
    """
    mat = embed_texts(chunks)  # shape (N, D), уже L2-нормировано в utils.embed_texts
    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(mat)
    return VectorIndex(nn=nn, mat=mat, chunks=chunks)


def search_topk(vindex: VectorIndex, query: str, k: int = 5) -> List[Tuple[str, float]]:
    """
    Возвращает top-k кусков с близостью (similarity = 1 - cosine_distance).
    """
    q = embed_query(query)  # shape (1, D), L2-нормирован
    distances, indices = vindex.nn.kneighbors(q, n_neighbors=min(k, len(vindex.chunks)))
    res: List[Tuple[str, float]] = []
    for j, dist in zip(indices[0], distances[0]):
        sim = 1.0 - float(dist)  # cosine_similarity
        res.append((vindex.chunks[int(j)], sim))
    # на всякий случай отсортируем по убыванию схожести
    res.sort(key=lambda x: x[1], reverse=True)
    return res
