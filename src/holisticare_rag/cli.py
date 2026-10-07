"""Command line: ``holisticare <command>``."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

from . import __version__, synthetic
from .config import Settings
from .embed import make_embedder
from .evaluate import evaluate, load_questions
from .index import ensure_index
from .service import Assistant


def _print_answer(a, as_json: bool) -> None:
    if as_json:
        print(json.dumps(asdict(a), indent=2))
        return
    print(a.text)
    for c in a.cautions:
        print(f"\nCAUTION: {c}")
    if a.citations:
        print("\nSources:")
        for c in a.citations:
            where = c.source + (f", page {c.page}" if c.page else "") + (f", {c.section}" if c.section else "")
            print(f"  [{c.number}] {c.title} ({where}; {c.evidence})\n      \"{c.quote}\"")
    for n in a.notes:
        print(f"\nnote: {n}")
    print(f"\n{a.disclaimer}")


def cmd_synth(args, s):
    out = synthetic.write(args.out_dir)
    print(f"wrote the synthetic knowledge base to {out / 'documents'} and the question set to {out / 'eval.jsonl'}")


def cmd_index(args, s):
    emb = make_embedder(s.embedder, s.embed_model, s.llm_base_url, s.timeout_s)
    idx, status = ensure_index(s, emb, force=args.force)
    m = idx.manifest
    print(f"index {status}: {m['documents']} documents, {m['chunks']} chunks, embedder {m['embedder']} -> {s.index_dir}")


def cmd_ask(args, s):
    assistant, status = Assistant.from_settings(s)
    if status != "reused":
        print(f"(index {status})", file=sys.stderr)
    _print_answer(assistant.ask(args.question), args.json)


def cmd_chat(args, s):  # pragma: no cover - interactive
    assistant, _ = Assistant.from_settings(s)
    history: list[dict] = []
    print("Type a question. An empty line ends the chat.")
    while True:
        try:
            q = input("\nyou> ").strip()
        except EOFError:
            break
        if not q:
            break
        a = assistant.ask(q, history)
        _print_answer(a, False)
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": a.text}]


def cmd_eval(args, s):
    assistant, _ = Assistant.from_settings(s)
    summary, df = evaluate(assistant, load_questions(args.questions))
    print(df[["id", "expect", "got", "category_ok", "hit", "cited", "cited_expected", "supported",
              "cited_sentences"]].to_string(index=False))
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)


def cmd_ui(args, s):  # pragma: no cover - starts a server
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(Path(__file__).with_name("app.py"))])


def cmd_demo(args, s):
    out = Path(args.out_dir)
    synthetic.write(out)
    s = replace(s, docs_dir=out / "documents", index_dir=out / "index", llm="extractive")
    cmd_index(argparse.Namespace(force=True), s)
    for q in ("How is Tarsil cough treated?", "How many mg of Kelvar root should I take?",
              "How do I repair the brakes on my bicycle?"):
        print(f"\n== {q}")
        cmd_ask(argparse.Namespace(question=q, json=False), s)
    print("\n== evaluation on the synthetic question set")
    cmd_eval(argparse.Namespace(questions=out / "eval.jsonl", out=out / "eval_results.csv"), s)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="holisticare", description="Cited health information from your own documents.")
    p.add_argument("--version", action="version", version=f"holisticare-rag {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("synth", help="write the synthetic knowledge base and question set")
    s.add_argument("--out-dir", default="data/synthetic")
    s.set_defaults(func=cmd_synth)
    s = sub.add_parser("index", help="build the index, or reuse it if the documents did not change")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_index)
    s = sub.add_parser("ask", help="answer one question")
    s.add_argument("question")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_ask)
    s = sub.add_parser("chat", help="interactive chat with history")
    s.set_defaults(func=cmd_chat)
    s = sub.add_parser("eval", help="run the question set")
    s.add_argument("--questions", default="data/synthetic/eval.jsonl")
    s.add_argument("--out")
    s.set_defaults(func=cmd_eval)
    s = sub.add_parser("ui", help="start the Streamlit app (extra ui)")
    s.set_defaults(func=cmd_ui)
    s = sub.add_parser("demo", help="offline demo on the synthetic knowledge base")
    s.add_argument("--out-dir", default="data/synthetic")
    s.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rc = args.func(args, Settings.from_env())
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return rc or 0


if __name__ == "__main__":
    raise SystemExit(main())
