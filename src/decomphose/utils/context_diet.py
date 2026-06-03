from __future__ import annotations

import re
from typing import Any

from decomphose.types import ContextDocument, MicroTask

CONTEXT_MARKER = re.compile(r"\[context:([^\]]+)\]", re.IGNORECASE)


def extract_context_documents(messages: list[dict[str, Any]]) -> list[ContextDocument]:
    docs: list[ContextDocument] = []
    seen: set[str] = set()

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, str):
            continue

        for match in CONTEXT_MARKER.finditer(content):
            key = match.group(1).strip()
            if key in seen:
                continue
            seen.add(key)
            start = match.start()
            next_match = CONTEXT_MARKER.search(content, match.end())
            end = next_match.start() if next_match else len(content)
            docs.append(ContextDocument(key=key, content=content[start:end].strip()))

    if not docs:
        combined = "\n\n".join(
            m["content"]
            for m in messages
            if isinstance(m.get("content"), str)
        )
        if combined:
            docs.append(ContextDocument(key="full-thread", content=combined))

    return docs


def slice_context_for_task(documents: list[ContextDocument], task: MicroTask) -> str:
    if not task.relevant_context_keys:
        return "\n\n---\n\n".join(d.content for d in documents)

    selected = [d for d in documents if d.key in task.relevant_context_keys]
    if not selected:
        return documents[0].content if documents else ""

    return "\n\n---\n\n".join(f"[{d.key}]\n{d.content}" for d in selected)


def estimate_token_budget(text: str) -> int:
    return (len(text) + 3) // 4
