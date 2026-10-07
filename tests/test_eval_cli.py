from holisticare_rag.cli import main
from holisticare_rag.evaluate import evaluate, load_questions


def test_evaluation_on_the_synthetic_set(assistant, kb):
    summary, df = evaluate(assistant, load_questions(kb / "eval.jsonl"))
    assert summary["questions"] == 18
    assert summary["category_accuracy"] == 1.0
    assert summary["retrieval_hit@k"] == 1.0
    assert summary["refusal_precision"] == 1.0 and summary["refusal_recall"] == 1.0
    assert summary["system_prompt_unchanged"] is True
    assert summary["citation_precision"] > 0.7


def test_question_file_checks(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text('{"question": "x"}\n', encoding="utf-8")
    try:
        load_questions(p)
    except ValueError as exc:
        assert "expect" in str(exc)
    else:
        raise AssertionError("missing `expect` must be an error")


def test_cli_end_to_end(tmp_path, capsys, monkeypatch):
    out = tmp_path / "syn"
    assert main(["synth", "--out-dir", str(out)]) == 0
    monkeypatch.setenv("HOLISTICARE_DOCS_DIR", str(out / "documents"))
    monkeypatch.setenv("HOLISTICARE_INDEX_DIR", str(tmp_path / "idx"))
    assert main(["index"]) == 0
    assert "index built" in capsys.readouterr().out
    assert main(["index"]) == 0
    assert "index reused" in capsys.readouterr().out
    assert main(["ask", "What are the symptoms of Velmora rash?"]) == 0
    text = capsys.readouterr().out
    assert "Sources:" in text and "not medical advice" in text
    assert main(["ask", "dose of Kelvar root", "--json"]) == 0
    assert '"category": "dosing"' in capsys.readouterr().out
    assert main(["eval", "--questions", str(out / "eval.jsonl"), "--out", str(tmp_path / "r.csv")]) == 0
    assert (tmp_path / "r.csv").exists()


def test_cli_errors(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HOLISTICARE_DOCS_DIR", str(tmp_path / "none"))
    assert main(["ask", "x"]) == 2
    assert "holisticare synth" in capsys.readouterr().err
