# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


def get_openai_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Не найден OPENAI_API_KEY в .env")
    return OpenAI(api_key=api_key)


def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    client = get_openai_client()
    resp = client.embeddings.create(model=model, input=texts)
    vecs = [d.embedding for d in resp.data]
    arr = np.array(vecs, dtype=np.float32)
    # нормируем для косинусной близости через inner product
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms


def embed_query(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    return embed_texts([text], model=model)[0:1, :]


def format_context(chunks_with_scores: List[Tuple[str, float]]) -> str:
    lines = []
    for i, (chunk, score) in enumerate(chunks_with_scores, start=1):
        lines.append(f"[ctx#{i} score={score:.3f}]\n{chunk}\n")
    return "\n".join(lines)
