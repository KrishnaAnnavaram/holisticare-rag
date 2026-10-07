"""The assistant: triage, retrieval, generation, checks and the final answer.

Order of work for one question:

1. Safety triage. A ``stop`` category returns its fixed text. No retrieval, no model call.
2. Retrieval query from the question and the earlier user messages.
3. Hybrid retrieval and the relevance gate. No evidence: a fixed "no source" answer.
4. Generation: a chat model with message-list history, or the extractive answer.
5. Checks: citation check (fallback to the extractive answer), dose filter, cautions, disclaimer.
"""
from __future__ import annotations

from .answer import Answer, check_citations, citations_for, extractive_answer
from .config import Settings
from .embed import Embedder, make_embedder
from .index import KnowledgeIndex, ensure_index
from .llm import ChatModel, OllamaChat, OpenAICompatChat
from .prompts import build_messages
from .retrieve import retrieve
from .rewrite import retrieval_query
from .safety import CAUTION_TEXT, DISCLAIMER, remove_doses, triage
from .text import sentences

NO_SOURCE_TEXT = (
    "The indexed documents do not cover this question, so I cannot answer it from a source. "
    "Ask a health professional, or add a reliable document about this topic to the knowledge base."
)
DOSE_NOTE = "Dose or schedule details were removed. Ask a doctor or a pharmacist."


def make_chat_model(s: Settings) -> ChatModel | None:
    if s.llm == "ollama":
        return OllamaChat(s.llm_model, s.llm_base_url, s.temperature, s.max_tokens, s.timeout_s)
    if s.llm == "openai":
        return OpenAICompatChat(s.llm_model, s.llm_base_url, s.temperature, s.max_tokens, s.timeout_s)
    return None


class Assistant:
    def __init__(self, index: KnowledgeIndex, embedder: Embedder, settings: Settings, chat: ChatModel | None = None):
        self.index, self.embedder, self.settings, self.chat = index, embedder, settings, chat

    @classmethod
    def from_settings(cls, s: Settings, chat: ChatModel | None = None) -> tuple["Assistant", str]:
        emb = make_embedder(s.embedder, s.embed_model, s.llm_base_url, s.timeout_s)
        idx, status = ensure_index(s, emb)
        return cls(idx, emb, s, chat if chat is not None else make_chat_model(s)), status

    def ask(self, question: str, history: list[dict] | None = None) -> Answer:
        history = history or []
        question = question.strip()
        tri = triage(question)
        if tri.stop:
            return Answer(text=tri.message, category=tri.category, disclaimer=DISCLAIMER)
        rq = retrieval_query(question, history)
        ret = retrieve(self.index, self.embedder, rq, self.settings.top_k, self.settings.min_coverage)
        cautions = [CAUTION_TEXT[c] for c in tri.cautions]
        if not ret.has_evidence:
            return Answer(NO_SOURCE_TEXT, "no_source", cautions=cautions, retrieval_query=rq,
                          retrieved=[p.chunk.source for p in ret.passages],
                          notes=[f"best passage coverage {ret.best_coverage:.2f} < {self.settings.min_coverage}"],
                          disclaimer=DISCLAIMER)
        notes: list[str] = []
        model = "extractive"
        text = ""
        if self.chat is not None:
            try:
                reply = self.chat.complete(build_messages(question, ret.passages, history, self.settings.history_turns))
                reasons = check_citations(reply, len(ret.passages))
                if reasons:
                    notes.append("model answer refused: " + "; ".join(reasons))
                else:
                    text, model = reply.strip(), self.chat.name
            except Exception as exc:  # noqa: BLE001 - any provider error falls back to the extractive answer
                notes.append(f"model error: {type(exc).__name__}")
        if not text:
            text = extractive_answer(question, ret.passages)
        kept, removed = remove_doses(sentences(text))
        if removed:
            text = " ".join(kept)
            notes.append(DOSE_NOTE)
        return Answer(text=text, category="ok", citations=citations_for(text, ret.passages), cautions=cautions,
                      notes=notes, retrieval_query=rq, retrieved=[p.chunk.source for p in ret.passages],
                      model=model, disclaimer=DISCLAIMER)
