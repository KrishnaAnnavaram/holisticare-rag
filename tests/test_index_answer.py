"""Problems 4-6, 8 and 9: index lifecycle, generation settings, configuration and claim-level citations."""
from dataclasses import replace

import numpy as np
import pytest

from holisticare_rag.answer import check_citations, extractive_answer
from holisticare_rag.config import Settings, read_dotenv
from holisticare_rag.embed import HashingEmbedder
from holisticare_rag.index import BM25, ensure_index, load_index
from holisticare_rag.ingest import Source, ingest, split_text
from holisticare_rag.llm import OllamaChat, OpenAICompatChat, ScriptedChat
from holisticare_rag.retrieve import retrieve
from holisticare_rag.service import Assistant


def test_index_is_reused_and_rebuilt_when_documents_change(settings, kb):
    emb = HashingEmbedder()
    _, s1 = ensure_index(settings, emb)
    _, s2 = ensure_index(settings, emb)
    assert (s1, s2) == ("built", "reused")
    (kb / "documents" / "conventional" / "new_topic.md").write_text("# Zelto itch\n\nZelto itch is fictional.\n", encoding="utf-8")
    idx, s3 = ensure_index(settings, emb)
    assert s3 == "rebuilt" and any(c.source == "conventional/new_topic.md" for c in idx.chunks)
    _, s4 = ensure_index(replace(settings, chunk_chars=500), emb)
    assert s4 == "rebuilt"


def test_index_files_need_no_pickle(settings):
    ensure_index(settings, HashingEmbedder())
    idx = load_index(settings.index_dir)
    assert idx.vectors.dtype == np.float32 and len(idx.chunks) == idx.vectors.shape[0]


def test_registry_labels_and_unknown_documents(kb):
    docs = kb / "documents"
    (docs / "extra.txt").write_text("Plain text about Velmora rash care.", encoding="utf-8")
    chunks, rep = ingest(docs)
    labels = {c.source: c.evidence for c in chunks}
    assert labels["conventional/velmora_rash.md"] == "evidence_based"
    assert labels["traditional/kelvar_root.md"] == "traditional"
    assert labels["extra.txt"] == "unknown" and any("extra.txt" in w for w in rep.warnings)
    with pytest.raises(ValueError):
        Source(file="x", title="x", evidence="miracle")


def test_split_text_respects_the_limit():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    parts = split_text(text, 300, 50)
    assert all(len(p) <= 300 for p in parts) and len(parts) > 5
    assert split_text("short", 300, 50) == ["short"]


def test_bm25_prefers_the_matching_document():
    bm = BM25([["velmora", "rash"], ["tarsil", "cough"], ["cough", "cough", "night"]])
    s = bm.scores(["tarsil", "cough"])
    assert s.argmax() == 1 and s[0] == 0


def test_retrieval_gate(settings):
    idx, _ = ensure_index(settings, HashingEmbedder())
    good = retrieve(idx, HashingEmbedder(), "symptoms of Velmora rash", 4, 0.5)
    assert good.has_evidence and good.passages[0].chunk.source == "conventional/velmora_rash.md"
    assert [p.number for p in good.passages] == list(range(1, len(good.passages) + 1))
    bad = retrieve(idx, HashingEmbedder(), "capital city of Australia", 4, 0.5)
    assert not bad.has_evidence


def test_citation_check():
    assert check_citations("Velmora rash itches at night [1]. It peels [2].", 2) == []
    assert "do not exist" in check_citations("Velmora rash itches at night [7].", 2)[0]
    assert "sentences have a citation" in check_citations("Velmora rash itches at night. It peels too much here.", 2)[0]
    assert check_citations("", 2) == ["empty answer"]


def test_uncited_model_answer_falls_back_to_extractive(settings):
    chat = ScriptedChat(["Velmora rash is caused by stress and you should take antibiotics for it."])
    a = Assistant.from_settings(settings, chat=chat)[0].ask("What are the symptoms of Velmora rash?")
    assert a.model == "extractive" and any("model answer refused" in n for n in a.notes)
    assert a.citations and all(c.quote for c in a.citations)


def test_model_errors_fall_back(settings):
    class Broken:
        name = "broken"

        def complete(self, messages):
            raise ConnectionError("offline")

    a = Assistant.from_settings(settings, chat=Broken())[0].ask("What are the symptoms of Velmora rash?")
    assert a.model == "extractive" and any("ConnectionError" in n for n in a.notes)


def test_cited_model_answer_is_kept_with_quotes(settings):
    chat = ScriptedChat(["The patches of Velmora rash itch most at night [1]."])
    a = Assistant.from_settings(settings, chat=chat)[0].ask("What are the symptoms of Velmora rash?")
    assert a.model == "scripted" and a.citations[0].number == 1
    assert "itch" in a.citations[0].quote.lower()


def test_extractive_answer_cites_every_sentence(settings):
    idx, _ = ensure_index(settings, HashingEmbedder())
    ret = retrieve(idx, HashingEmbedder(), "How is Tarsil cough treated?", 4, 0.5)
    text = extractive_answer("How is Tarsil cough treated?", ret.passages)
    assert check_citations(text, len(ret.passages)) == []
    assert "humidifier" in text.lower()


def test_generation_settings_are_sent():
    seen = {}

    def post(url, body, headers):
        seen.update(url=url, body=body, headers=headers)
        return {"message": {"content": "ok [1]."}}

    chat = OllamaChat("llama3.1", temperature=0.1, max_tokens=300, post=post)
    assert chat.complete([{"role": "user", "content": "hi"}]) == "ok [1]."
    assert seen["url"].endswith("/api/chat") and seen["body"]["options"] == {"temperature": 0.1, "num_predict": 300}
    assert seen["body"]["stream"] is False


def test_openai_compatible_adapter_reads_the_key(monkeypatch):
    monkeypatch.delenv("HOLISTICARE_LLM_API_KEY", raising=False)
    with pytest.raises(ValueError):
        OpenAICompatChat("m")
    seen = {}

    def post(url, body, headers):
        seen.update(url=url, body=body, headers=headers)
        return {"choices": [{"message": {"content": "fine [1]."}}]}

    chat = OpenAICompatChat("m", base_url="http://localhost:8000/v1", api_key="k", post=post)
    assert chat.complete([]) == "fine [1]." and seen["headers"]["Authorization"] == "Bearer k"
    assert seen["body"]["temperature"] == 0.1 and seen["url"] == "http://localhost:8000/v1/chat/completions"


def test_settings_from_env_and_limits(tmp_path):
    s = Settings.from_env({"HOLISTICARE_DOCS_DIR": "kb", "HOLISTICARE_TOP_K": "6", "HOLISTICARE_LLM": "ollama"})
    assert str(s.docs_dir) == "kb" and s.top_k == 6 and s.llm == "ollama"
    for bad in ({"HOLISTICARE_TEMPERATURE": "0.7"}, {"HOLISTICARE_LLM": "gpt"}, {"HOLISTICARE_CHUNK_CHARS": "100"},
                {"HOLISTICARE_MIN_COVERAGE": "0"}):
        with pytest.raises(ValueError):
            Settings.from_env(bad)
    (tmp_path / ".env").write_text("HOLISTICARE_TOP_K=3\nHOLISTICARE_LLM_API_KEY=\n", encoding="utf-8")
    assert read_dotenv(tmp_path / ".env") == {"HOLISTICARE_TOP_K": "3"}


def test_missing_documents_folder(tmp_path):
    with pytest.raises(FileNotFoundError, match="holisticare synth"):
        Assistant.from_settings(replace(Settings(), docs_dir=tmp_path / "none", index_dir=tmp_path / "i"))


def test_streamlit_app_caches_the_assistant():
    pytest.importorskip("streamlit")
    from holisticare_rag import app

    assert hasattr(app._assistant, "clear")  # st.cache_resource wrapper
