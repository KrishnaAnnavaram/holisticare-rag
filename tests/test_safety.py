"""Problem 1 (no guardrails) and problem 7 (answers with no relevant source)."""
import pytest

from holisticare_rag.llm import ScriptedChat
from holisticare_rag.safety import DISCLAIMER, remove_doses, triage
from holisticare_rag.service import Assistant


@pytest.mark.parametrize("q, cat", [
    ("I have crushing chest pain", "emergency"),
    ("my mother cannot breathe", "emergency"),
    ("his face is drooping and he has slurred speech", "emergency"),
    ("I think my child swallowed poison", "emergency"),
    ("I want to kill myself", "self_harm"),
    ("thinking about self-harm again", "self_harm"),
    ("What dose of turmeric is best?", "dosing"),
    ("How many tablets can I take per day?", "dosing"),
    ("is 500 mg too much", "dosing"),
    ("Should I stop taking my insulin?", "stop_medication"),
    ("Can herbs work instead of chemotherapy?", "stop_medication"),
    ("What is Velmora rash?", "ok"),
])
def test_triage_categories(q, cat):
    assert triage(q).category == cat


def test_cautions_for_pregnancy_and_children():
    t = triage("I am pregnant and my toddler has a cough")
    assert t.category == "ok" and set(t.cautions) == {"pregnancy", "child"}


def test_stop_category_makes_no_retrieval_and_no_model_call(settings):
    chat = ScriptedChat(["should not be used [1]."])
    a = Assistant.from_settings(settings, chat=chat)[0].ask("I have chest pain and sweat a lot")
    assert a.category == "emergency" and "emergency" in a.text.lower()
    assert chat.calls == [] and a.citations == [] and a.disclaimer == DISCLAIMER


def test_off_topic_question_gets_no_source_and_no_model_call(settings):
    chat = ScriptedChat(["Brakes need new pads [1]."])
    a = Assistant.from_settings(settings, chat=chat)[0].ask("How do I repair the brakes on my bicycle?")
    assert a.category == "no_source" and chat.calls == [] and a.citations == []


def test_dose_sentences_are_removed():
    kept, removed = remove_doses(["Take 2 grams twice a day.", "It can interact with blood thinners.",
                                  "Use 5 ml once a day."])
    assert kept == ["It can interact with blood thinners."] and removed == 2


def test_dose_in_a_model_answer_is_removed(settings):
    chat = ScriptedChat(["Kelvar root can interact with blood thinners [1]. "
                         "Traditional texts describe 2 grams of root powder twice a day [1]."])
    a = Assistant.from_settings(settings, chat=chat)[0].ask("What cautions apply to Kelvar root powder?")
    assert a.category == "ok" and a.model == "scripted"
    assert "2 grams" not in a.text and "blood thinners" in a.text
    assert any("Dose or schedule details were removed" in n for n in a.notes)


def test_traditional_sources_are_labelled(assistant):
    a = assistant.ask("Can Mirasa leaf paste help Velmora rash?")
    assert "From traditional-medicine sources" in a.text
    assert "not established by clinical evidence" in a.text
    assert {c.evidence for c in a.citations} <= {"evidence_based", "traditional"}
    assert any(c.evidence == "traditional" for c in a.citations)


def test_every_answer_has_the_disclaimer(assistant):
    for q in ("What are the symptoms of Velmora rash?", "What is the capital of France?", "dose of Kelvar?"):
        assert assistant.ask(q).disclaimer == DISCLAIMER
