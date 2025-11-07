# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from .prompts import SYSTEM_PROMPT, REFUSAL_TEXT
from .retriever import (
    fetch_doc_if_needed,
    build_chunks,
    build_faiss_index,
    search_topk,
)
from .utils import format_context

MODEL = "gpt-4o-mini"
SIM_THRESHOLD = 0.30  # если лучший контекст ниже — считаем вопрос "вне темы"


def build_indices():
    """Готовим два индекса: для rc и для tok, чтобы выполнить требование '2 сплиттера'."""
    text = fetch_doc_if_needed()
    chunks_rc = build_chunks(text, mode="rc")
    chunks_tok = build_chunks(text, mode="tok")
    vindex_rc = build_faiss_index(chunks_rc)
    vindex_tok = build_faiss_index(chunks_tok)
    return vindex_rc, vindex_tok


def answer_with_context(
    client: OpenAI,
    question: str,
    ctx: List[Tuple[str, float]],
    memory: List[dict],
) -> str:
    """Формирует промпт из системки + контекста + истории и запрашивает модель (openai>=2.x)."""
    context_text = format_context(ctx)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Контекст (фрагменты из документа):\n{context_text}"},
    ]
    # добавляем историю последних сообщений
    for m in memory[-6:]:
        messages.append(m)
    # текущий вопрос
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    return response.choices[0].message.content.strip()


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Не найден OPENAI_API_KEY в .env")
    client = OpenAI(api_key=api_key)

    print("Готовлю индексы (две стратегии сплиттинга: rc и tok)...")
    vindex_rc, vindex_tok = build_indices()
    active = "rc"  # текущий индекс по умолчанию
    print("Готово. Активный сплиттер: rc (переключить: /split rc|tok). Выйти: /exit")

    memory: List[dict] = []  # история переписки (для контекста диалога)

    while True:
        user_q = input("\nВаш вопрос: ").strip()
        if not user_q:
            continue
        if user_q.lower() in ("/exit", "exit", "quit"):
            print("Выход.")
            break
        if user_q.lower().startswith("/split"):
            _, *rest = user_q.split()
            if rest and rest[0] in ("rc", "tok"):
                active = rest[0]
                print(f"Активный сплиттер: {active}")
            else:
                print("Использование: /split rc  или  /split tok")
            continue

        # 1) Ретрив с активного индекса
        vindex = vindex_rc if active == "rc" else vindex_tok
        ctx = search_topk(vindex, user_q, k=5)
        best_score = ctx[0][1] if ctx else 0.0

        # 2) Фильтр "вопрос вне темы"
        if best_score < SIM_THRESHOLD:
            print(REFUSAL_TEXT)

            # память диалога
            memory.append({"role": "user", "content": user_q})
            memory.append({"role": "assistant", "content": REFUSAL_TEXT})

            # 🧾 логируем даже "вне темы"
            os.makedirs("outputs", exist_ok=True)
            with open("outputs/transcript.txt", "a", encoding="utf-8") as f:
                f.write(f"Q: {user_q}\n")
                f.write("best_score: NONE (вопрос вне темы)\n")
                f.write(f"A: {REFUSAL_TEXT}\n")
                f.write("-" * 40 + "\n")
            print("\n[Лог обновлён: outputs/transcript.txt]")
            continue
        else:
            # 3) Ответ с учётом контекста и истории
            answer = answer_with_context(client, user_q, ctx, memory)
            print("\nОтвет:\n", answer)

            # 4) Память диалога
            memory.append({"role": "user", "content": user_q})
            memory.append({"role": "assistant", "content": answer})

            # 5) Лёгкий лог в файл (доказывает тестирование)
            os.makedirs("outputs", exist_ok=True)
            with open("outputs/transcript.txt", "a", encoding="utf-8") as f:
                f.write(f"Q: {user_q}\n")
                f.write(f"best_score: {best_score:.3f}  [split={active}]\n")
                f.write(f"A: {answer}\n")
                f.write("-" * 40 + "\n")
            print("\n[Лог обновлён: outputs/transcript.txt]")


if __name__ == "__main__":
    main()
