"""
Thin model layer. All model calls go through here so the rest of the code never
touches a provider SDK directly.

LLM_MODE=fake gives deterministic, offline behaviour for unit tests and CI:
hashed bag-of-words embeddings and rule-based chat responses.
"""
import hashlib
import json
import math
import os
import re
from functools import lru_cache

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from . import config

_retry = retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3), reraise=True)


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env locally or as a Space secret.")
    return OpenAI(api_key=key, timeout=60)


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ---------------------------------------------------------------- embeddings

def embed(texts: list[str]) -> list[list[float]]:
    if config.LLM_MODE == "fake":
        return [_fake_embed(t) for t in texts]
    vectors = []
    for i in range(0, len(texts), 100):
        vectors.extend(_embed_batch(texts[i:i + 100]))
    return vectors


@_retry
def _embed_batch(batch: list[str]) -> list[list[float]]:
    resp = _client().embeddings.create(model=config.EMBED_MODEL, input=batch)
    return [d.embedding for d in resp.data]


def _fake_embed(text: str, dim: int = 256) -> list[float]:
    vec = [0.0] * dim
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0 if (h >> 8) % 2 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


# ---------------------------------------------------------------------- chat

def chat(messages: list[dict], model: str | None = None, json_mode: bool = False) -> str:
    if config.LLM_MODE == "fake":
        return _fake_chat(messages)
    return _chat_openai(messages, model or config.CHAT_MODEL, json_mode)


@_retry
def _chat_openai(messages, model, json_mode):
    kwargs = {"model": model, "messages": messages, "temperature": 0}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _client().chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def chat_json(messages: list[dict], schema: type[BaseModel], model: str | None = None) -> BaseModel:
    """Call the model in JSON mode and validate against a pydantic schema. Raises on invalid output."""
    raw = chat(messages, model=model, json_mode=True)
    return schema.model_validate_json(_strip_fences(raw))


def _fake_chat(messages: list[dict]) -> str:
    system = messages[0]["content"]
    user = messages[-1]["content"]
    task = re.search(r"TASK: (\w+)", system)
    task = task.group(1) if task else ""

    if task == "SCOPE":
        q = re.search(r"Current question:\s*(.+)", user, re.S)
        return json.dumps({"category": "chart_question",
                           "standalone_question": (q.group(1).strip() if q else user),
                           "reason": "fake mode"})
    if task == "ANSWER":
        m = re.search(r"\[S1\][^\n]*\n(.+?)(?:\n\[S2\]|\n\nQuestion:)", user, re.S)
        body = m.group(1).strip() if m else ""
        first = re.split(r"(?<=[.!?])\s+", body)[0] if body else ""
        return f"{first} [S1]" if first else "NOT_DOCUMENTED: nothing relevant in the sources."
    if task == "VERIFY":
        ans = re.search(r"Answer:\s*(.+)", user, re.S)
        claim = ans.group(1).strip() if ans else ""
        return json.dumps({"claims": [{"claim": claim, "supported": True, "sources": ["S1"]}]})
    if task == "JUDGE":
        return json.dumps({"correct": True, "explanation": "fake mode"})
    return ""
