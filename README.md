<div align="center">

# holisticare-rag — Local Health Information Assistant With Cited Sources

**holisticare-rag is a local question-answer assistant for health information in your own documents. It takes a question through these steps to give a short answer with checked citations:**

`safety triage` → `history-aware query` → `hybrid retrieval + relevance gate` → `generate (temperature ≤ 0.3)` → `check citations and doses`.

![Safety](https://img.shields.io/badge/Safety-triage_before_retrieval-1F3864?style=for-the-badge)
![Retrieval](https://img.shields.io/badge/Retrieval-BM25_%2B_vectors-2E5FD9?style=for-the-badge)
![Citations](https://img.shields.io/badge/Citations-checked_per_sentence-6E86E8?style=for-the-badge)
![Tests](https://img.shields.io/badge/Tests-44_passing-3DA35B?style=for-the-badge)
![Offline demo](https://img.shields.io/badge/Offline_demo-Yes-F5C542?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-A0399B?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-hashing_vectors-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-optional-000000?style=flat-square&logo=ollama&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-optional-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Docs](https://img.shields.io/badge/Docs-ASD--STE100-5D6D7E?style=flat-square)

**[Summary](#1-summary)** ·
**[Workflow](#4-the-end-to-end-workflow)** ·
**[Run it](#10-how-to-run-holisticare-rag)** ·
**[Configuration](#104-environment-variables)** ·
**[Known problems](#13-known-problems)** ·
**[Glossary](#15-glossary)**

</div>

> [!NOTE]
> This README uses ASD-STE100 Simplified Technical English. The writing rules and the project
> vocabulary are in [`docs/ste-style-guide.md`](docs/ste-style-guide.md). Each term in the
> [Glossary](#15-glossary) has only one meaning.

> [!WARNING]
> Do not use holisticare-rag as a medical decision tool. It gives general information from the documents that you
> index, and it can be wrong. A qualified health professional must review every decision about a diagnosis, a
> treatment or a medicine.

---

holisticare-rag answers health questions from a local library of documents that you choose and label. The main idea is safety before generation. Fixed rules catch emergencies, self-harm, dosing and stop-medication questions before any retrieval. A relevance gate refuses questions that the documents do not cover. Each answer sentence must cite a passage, and evidence-based and traditional-medicine sources stay apart.

This README is the **one location that explains all of holisticare-rag**. It gives these topics:

- the general design
- each component and its procedure, step by step
- the decision rules
- the data map
- the runbook
- the validation results and the known problems

| If you are… | Read |
|---|---|
| A manager or reviewer | [1](#1-summary), [3](#3-design-rules), [4](#4-the-end-to-end-workflow), [12](#12-validation-results), [13](#13-known-problems) |
| A developer who joins the project | All sections, in sequence. Keep [10](#10-how-to-run-holisticare-rag) and [13](#13-known-problems) open while you work |
| An operator who runs holisticare-rag | [10](#10-how-to-run-holisticare-rag), [8](#8-the-safety-rules), then [13](#13-known-problems) |

---

## Table of contents

1. 🧭 [Summary](#1-summary)
2. 🏗️ [How holisticare-rag is built](#2-how-holisticare-rag-is-built)
   - 2.1 [Components](#21-components)
   - 2.2 [System context](#22-system-context)
   - 2.3 [Repository layout](#23-repository-layout)
3. 🛡️ [Design rules](#3-design-rules)
4. 🔄 [The end-to-end workflow](#4-the-end-to-end-workflow)
   - 4.1 [Full flow](#41-full-flow)
   - 4.2 [The life cycle of one question](#42-the-life-cycle-of-one-question)
   - 4.3 [Who does which step](#43-who-does-which-step)
5. 🔵 [The knowledge base and the index](#5-the-knowledge-base-and-the-index)
6. 🟢 [Retrieval and the relevance gate](#6-retrieval-and-the-relevance-gate)
7. 🟣 [Generation and the answer checks](#7-generation-and-the-answer-checks)
8. ⚖️ [The safety rules](#8-the-safety-rules)
9. 🗂️ [Data and file map](#9-data-and-file-map)
10. ▶️ [How to run holisticare-rag](#10-how-to-run-holisticare-rag)
    - 10.1 [Prerequisites](#101-prerequisites) · 10.2 [Installation](#102-installation) · 10.3 [Run holisticare-rag](#103-run-holisticare-rag) · 10.4 [Environment variables](#104-environment-variables)
11. 🧩 [How to extend holisticare-rag](#11-how-to-extend-holisticare-rag)
12. ✅ [Validation results](#12-validation-results)
13. ⚠️ [Known problems](#13-known-problems)
14. 📌 [Key points](#14-key-points)
15. 📖 [Glossary](#15-glossary)
16. 📄 [License](#16-license)

---

## 1. Summary

**The problem.** People ask health questions, and some of the questions are dangerous to answer from a chat model. The difficult questions are:

- Which questions must never get a generated answer (emergencies, self-harm, doses, stopping a treatment)?
- What does the assistant do when the documents do not cover a question?
- How does a follow-up such as "what about for children?" find the earlier topic?
- How can a user check which passage supports each sentence?
- How does the user see the difference between clinical evidence and traditional use?

holisticare-rag gives each of these questions its own component. All of them run locally, and the default setup needs no model server.

| Item | Value |
|---|---|
| Input | A question, the chat history, and a folder of `.md`, `.txt` or `.pdf` documents with a `sources.json` registry |
| Output | An answer with a citation per sentence, a quote per citation, evidence labels, cautions, notes and a disclaimer |
| Components | **8**: ingestion, index, retrieval, query rewrite, safety triage, prompts, chat models, answer checks |
| Providers | Optional: Ollama or an OpenAI-compatible server for generation, sentence-transformers or Ollama for vectors |
| Offline mode | Everything. Hashing vectors, BM25 and an extractive answer need no model and no network |
| Safety | Triage before retrieval, a relevance gate, a citation check, a dose filter, a disclaimer on every answer |
| Tests | **44** unit tests pass in CI (`pytest`), 1 skips without the `ui` extra (Streamlit). With the extra: 45 pass |

```mermaid
flowchart LR
    Q["question + history"] --> T["safety triage"] --> R["hybrid retrieval + gate"] --> G["generate or extract"] --> C["citation and dose checks"] --> A["answer + sources + disclaimer"]
```

---

## 2. How holisticare-rag is built

### 2.1 Components

| Component | Module | Purpose |
|---|---|---|
| Settings | `src/holisticare_rag/config.py` | Environment variables and `.env`, with limits (temperature ≤ 0.3) |
| Text helpers | `src/holisticare_rag/text.py` | Tokens, content tokens, query synonyms, sentences |
| Ingestion | `src/holisticare_rag/ingest.py` | Source registry, document loading, section-aware chunks |
| Embedders | `src/holisticare_rag/embed.py` | Hashing (default), sentence-transformers (extra `st`), Ollama |
| Index | `src/holisticare_rag/index.py` | BM25, vectors, JSON files, content fingerprint |
| Retrieval | `src/holisticare_rag/retrieve.py` | BM25 + cosine, reciprocal-rank fusion, relevance gate |
| Query rewrite | `src/holisticare_rag/rewrite.py` | History-aware retrieval query for follow-ups |
| Safety | `src/holisticare_rag/safety.py` | Triage rules, fixed replies, cautions, dose filter, disclaimer |
| Prompts | `src/holisticare_rag/prompts.py` | Constant system message, history as messages, numbered passages |
| Chat models | `src/holisticare_rag/llm.py` | `OllamaChat`, `OpenAICompatChat`, `ScriptedChat` (tests) |
| Answers | `src/holisticare_rag/answer.py` | Extractive answer, citation check, supporting quotes |
| Assistant | `src/holisticare_rag/service.py` | The full procedure for one question |
| Evaluation | `src/holisticare_rag/evaluate.py` | Question-set metrics |
| Synthetic data | `src/holisticare_rag/synthetic.py` | Fictional knowledge base and question set |
| Streamlit app | `src/holisticare_rag/app.py` | Chat page with cached resources (extra `ui`) |
| CLI | `src/holisticare_rag/cli.py` | The `holisticare` command |

The component map shows which module calls which module. An arrow points from the caller to the module that it uses.

```mermaid
flowchart TB
    subgraph ENTRY["Entry points"]
        CLI["cli.py<br/>holisticare command"]
        APP["app.py<br/>Streamlit chat, extra ui"]
        EVA["evaluate.py<br/>evaluate, load_questions"]
    end
    SVC["service.py<br/>Assistant.ask"]
    subgraph KB["Knowledge base"]
        IDX["index.py<br/>ensure_index, BM25"]
        ING["ingest.py<br/>ingest, docs_fingerprint"]
        EMB["embed.py<br/>make_embedder"]
    end
    subgraph QA["Question steps"]
        SAF["safety.py<br/>triage, remove_doses"]
        RW["rewrite.py<br/>retrieval_query"]
        RET["retrieve.py<br/>retrieve"]
        PR["prompts.py<br/>build_messages"]
        LLM["llm.py<br/>OllamaChat, OpenAICompatChat"]
        ANS["answer.py<br/>extractive_answer, check_citations"]
    end
    CFG["config.py<br/>Settings"]
    SYN["synthetic.py<br/>write, eval_set"]
    TXT["text.py<br/>content_tokens, sentences"]

    CLI --> CFG
    CLI --> SYN
    CLI --> EVA
    CLI --> SVC
    APP --> SVC
    EVA --> SVC
    SVC --> IDX
    SVC --> EMB
    IDX --> ING
    SVC --> SAF
    SVC --> RW
    SVC --> RET
    SVC --> PR
    SVC --> LLM
    SVC --> ANS
    RET --> TXT
    ANS --> TXT
```

### 2.2 System context

```mermaid
flowchart TB
    U["user"] --> CLI["holisticare CLI"]
    U --> UI["Streamlit chat (optional)"]
    DOCS["document folder + sources.json (local)"] --> CLI
    CLI --> IDX["index folder (JSON + NumPy)"]
    UI --> IDX
    CLI --> LLM["Ollama or OpenAI-compatible server (optional)"]
    UI --> LLM
```

### 2.3 Repository layout

```
holisticare-rag/
├── .github/workflows/ci.yml   # pytest on Python 3.11
├── data/README.md             # how to add documents, registry fields, open sources (no data files)
├── docs/ste-style-guide.md    # writing rules and project vocabulary
├── src/holisticare_rag/       # the package (one module per component, see 2.1)
├── tests/                     # 45 offline tests on the fictional knowledge base (1 needs Streamlit)
├── .env.example               # variable names only
├── pyproject.toml             # dependencies, extras, the holisticare command
└── LICENSE                    # MIT
```

---

## 3. Design rules

### 3.1 Safety before retrieval
`safety.triage` runs first on every question. An emergency, self-harm, dosing or stop-medication question gets a fixed reply. No retrieval and no model call happen for these questions.

### 3.2 No answer without a source
The relevance gate needs a passage that holds at least half of the content tokens of the question (`HOLISTICARE_MIN_COVERAGE`). If no passage passes, the answer says that the documents do not cover the question. The model is not called.

### 3.3 User text never becomes an instruction
The system message is a constant. The history goes to the model as separate user and assistant messages. No template engine reads user text, so braces and "ignore the instructions" stay plain data.

```mermaid
flowchart LR
    SYS["SYSTEM<br/>constant in prompts.py"] --> M1["message 1<br/>role system"]
    H[/"Chat history"/] --> KEEP["Keep the last<br/>HISTORY_TURNS turns,<br/>user and assistant only"]
    KEEP --> M2["messages 2 to n<br/>role user or assistant"]
    P[/"Numbered passages"/] --> BLK["passage_block"]
    Q[/"Question"/] --> LAST["last message, role user<br/>Passages + Question"]
    BLK --> LAST
    M1 --> LIST[/"Message list<br/>to the chat model"/]
    M2 --> LIST
    LAST --> LIST
```

### 3.4 One citation for each sentence
A model answer must cite a passage in at least 80 % of its sentences, and each citation must point to a passage that exists. Otherwise the assistant uses the extractive answer. Each citation shows the passage sentence that supports it best.

### 3.5 Evidence labels stay visible
Each document has an evidence label in `sources.json`: `evidence_based`, `traditional` or `unknown`. The extractive answer puts the labels in separate parts. A traditional part always ends with a statement that clinical evidence does not establish the use.

### 3.6 Low temperature and a token cap
The settings refuse a temperature above 0.3. Ollama gets `temperature` and `num_predict`. An OpenAI-compatible server gets `temperature` and `max_tokens`.

### 3.7 A versioned index with no pickle
The index is JSON Lines, JSON and a NumPy array loaded with `allow_pickle=False`. A SHA-256 fingerprint of the documents, the registry, the chunk settings and the embedder decides if the index is reused or built again.

---

## 4. The end-to-end workflow

### 4.1 Full flow

```mermaid
flowchart TD
    DOCS[/"Document folder<br/>+ sources.json"/] --> ENS["ensure_index<br/>reuse, build or rebuild"]
    ENS --> IDX[("Index folder<br/>chunks.jsonl, vectors.npy,<br/>manifest.json")]
    Q[/"question + history"/] --> T{"safety triage"}
    T -->|"emergency, self_harm, dosing, stop_medication"| FIX[/"fixed reply + disclaimer"/]
    T -->|"ok (+ cautions)"| RW["retrieval query (adds earlier topic for follow-ups)"]
    RW --> BM["BM25 ranking"]
    RW --> VS["vector ranking"]
    IDX --> BM
    IDX --> VS
    BM --> F["reciprocal-rank fusion"]
    VS --> F
    F --> G{"relevance gate"}
    G -->|"no passage passes"| NS[/"no-source reply"/]
    G -->|"passages"| M{"chat model set?"}
    M -->|"yes"| LLM["model with constant system message and history messages"]
    M -->|"no"| EX["extractive answer"]
    LLM --> CC{"citation check"}
    LLM -->|"model error"| EX
    CC -->|"fails"| EX
    CC -->|"passes"| DF["dose filter"]
    EX --> DF
    DF --> OUT[/"answer, quotes, labels, cautions, disclaimer"/]
    OUT --> HUMAN{{"HUMAN<br/>a health professional<br/>reviews each decision"}}
    FIX --> HUMAN

    classDef human fill:#fff3cd,stroke:#b8901f,color:#3d2f00,font-weight:bold
    class HUMAN human
```

### 4.2 The life cycle of one question

The `category` field of the `Answer` gives the end state of each question.

```mermaid
stateDiagram-v2
    state "Received" as Received
    state "Retrieval query" as Query
    state "Passages found" as Retrieved
    state "Model reply" as Reply
    state "Draft answer" as Draft
    [*] --> Received: ask(question, history)
    Received --> self_harm: triage stop
    Received --> emergency: triage stop
    Received --> stop_medication: triage stop
    Received --> dosing: triage stop
    Received --> Query: triage ok, cautions kept
    Query --> Retrieved: retrieve
    Retrieved --> no_source: best coverage below MIN_COVERAGE
    Retrieved --> Reply: chat model set
    Retrieved --> Draft: no chat model, extractive_answer
    Reply --> Draft: check_citations passes
    Reply --> Draft: check fails or model error, extractive_answer
    Draft --> ok: remove_doses, citations_for, disclaimer
    self_harm --> [*]
    emergency --> [*]
    stop_medication --> [*]
    dosing --> [*]
    no_source --> [*]
    ok --> [*]
```

1. Check the question with the triage rules.
2. If the category stops the question, return the fixed reply and the disclaimer.
3. If the question is a follow-up, add the content tokens of the earlier user questions to the retrieval query.
4. Rank the chunks with BM25 and with cosine similarity, then fuse the two lists.
5. Keep the top `HOLISTICARE_TOP_K` passages that share at least one token with the query.
6. If no passage reaches the minimum coverage, return the no-source reply.
7. Generate the answer with the chat model, or extract it from the passages.
8. Check the citations. If the check fails, use the extractive answer.
9. Remove each sentence that states a dose or a schedule.
10. Add the quotes, the cautions, the notes and the disclaimer.

### 4.3 Who does which step

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant CLI as holisticare CLI
    participant AS as Assistant
    participant IX as Index folder
    participant SF as safety.py
    participant RT as retrieve.py
    participant LM as Chat model, Ollama or OpenAI-compatible
    participant AN as answer.py

    U->>CLI: holisticare ask "question"
    CLI->>CLI: Settings.from_env
    CLI->>AS: Assistant.from_settings
    AS->>IX: ensure_index, compare fingerprint
    IX-->>AS: index and status reused, built or rebuilt
    CLI->>AS: ask(question)
    AS->>SF: triage(question)
    SF-->>AS: category, cautions
    AS->>AS: retrieval_query(question, history)
    AS->>RT: retrieve(index, embedder, query, top_k, min_coverage)
    RT-->>AS: passages, has_evidence, best_coverage
    AS->>LM: complete(build_messages), only if a chat model is set
    LM-->>AS: reply text
    AS->>AN: check_citations(reply)
    AN-->>AS: reasons, empty if the reply passes
    AS->>AN: extractive_answer, if no valid reply
    AS->>SF: remove_doses(sentences)
    AS->>AN: citations_for(text, passages)
    AS-->>CLI: Answer
    CLI-->>U: answer, cautions, sources with quotes, notes, disclaimer
```

---

## 5. The knowledge base and the index

**Purpose.** Turn a folder of labelled documents into searchable chunks, and build the index again only when something changes.

```mermaid
flowchart LR
    DIR[/"Document folder"/] --> LIST["list_documents<br/>.md, .txt, .pdf"]
    REG[/"sources.json"/] --> LOADR["load_registry"]
    LIST --> SRC{"Entry in<br/>the registry?"}
    LOADR --> SRC
    SRC -- "no" --> UNK["Label unknown,<br/>warning in IngestReport"]
    SRC -- "yes" --> PG["_pages<br/>one page at a time for a PDF"]
    UNK --> PG
    PG --> SEC["_sections<br/>split at Markdown headings"]
    SEC --> SPL["split_text<br/>CHUNK_CHARS, CHUNK_OVERLAP"]
    SPL --> CH["Chunk: source, title, section,<br/>page, evidence, tradition"]
    CH --> EMB["embedder.embed"]
    EMB --> SAVE["save_index"]
    SAVE --> OUT[("chunks.jsonl, vectors.npy,<br/>manifest.json")]
```

| Input | Output |
|---|---|
| Document folder, `sources.json`, chunk settings, embedder | `chunks.jsonl`, `vectors.npy`, `manifest.json` in the index folder |

**Procedure**

1. List the `.md`, `.txt` and `.pdf` files in the document folder and its sub-folders.
2. Find the registry entry of each file. A file with no entry gets the label `unknown`. The ingestion report records a warning, but the `index` command does not print it.
3. Read the text (one page at a time for a PDF).
4. Split the text into sections at Markdown headings.
5. Split each section at sentence and paragraph borders into chunks of at most `HOLISTICARE_CHUNK_CHARS` characters, with an overlap.
6. Give each chunk its source, title, section, page, evidence label and tradition.
7. Calculate the vectors of all chunks with the embedder.
8. Save the chunks, the vectors and the manifest with the fingerprint.

**Rules**

- `ensure_index` reuses the saved index only if the fingerprint and the format version match. The status is `reused`, `built` or `rebuilt`.

```mermaid
flowchart TD
    IN[/"Settings and embedder"/] --> FP["docs_fingerprint: SHA-256 of paths, contents,<br/>sources.json, chunk settings, embedder name"]
    FP --> M{"manifest.json exists<br/>and no --force?"}
    M -- "no" --> NEW{"manifest.json exists?"}
    M -- "yes" --> SAME{"Same fingerprint and<br/>FORMAT_VERSION?"}
    SAME -- "yes" --> LOAD["load_index<br/>allow_pickle False"]
    LOAD --> R[/"status reused"/]
    SAME -- "no" --> NEW
    NEW -- "yes" --> RB["build_index, save_index"]
    NEW -- "no" --> B["build_index, save_index"]
    RB --> S1[/"status rebuilt"/]
    B --> S2[/"status built"/]
```
- A PDF needs the extra `pdf` (`pypdf`). Scanned PDFs without a text layer give no text.

---

## 6. Retrieval and the relevance gate

**Purpose.** Find the passages that answer the question, and refuse when there are none.

```mermaid
flowchart TD
    Q[/"Retrieval query"/] --> QT["query_tokens<br/>content tokens + QUERY_SYNONYMS"]
    QT --> E{"No query token<br/>or empty index?"}
    E -- "yes" --> NONE[/"No passages, has_evidence false"/]
    E -- "no" --> BM["BM25 scores<br/>k1 1.5, b 0.75"]
    E -- "no" --> COS["Cosine: chunk vectors<br/>x embed(query)"]
    BM --> TOP["Top 30 of each list"]
    COS --> TOP
    TOP --> RRF["Reciprocal-rank fusion<br/>RRF_K 60"]
    RRF --> DED["Remove duplicates,<br/>keep the top TOP_K"]
    DED --> CV["Coverage of each passage:<br/>best of full query and<br/>each question sentence"]
    CV --> Z["Remove passages<br/>with coverage 0, number them"]
    Z --> G{"Best coverage at least<br/>MIN_COVERAGE?"}
    G -- "yes" --> EV[/"Passages, has_evidence true"/]
    G -- "no" --> NS[/"Passages, has_evidence false<br/>no-source reply"/]
```

| Input | Output |
|---|---|
| Retrieval query, index, embedder | Numbered passages with BM25 score, cosine, coverage and fused score, and the gate result |

**Procedure**

1. Make the content tokens of the query and add the query synonyms (for example `treated` adds `care`).
2. Rank the chunks by BM25 (k1 = 1.5, b = 0.75) and by cosine, and keep the top 30 of each list.
3. Fuse the two lists with reciprocal-rank fusion (k = 60).
4. Remove duplicate chunks. Keep the top `HOLISTICARE_TOP_K` passages.
5. Calculate the coverage of each passage: the share of the query content tokens that it contains. Use the full query and each question sentence with two or more content tokens, and keep the highest value.
6. Remove passages with a coverage of 0.
7. Pass the gate if the best coverage is at least `HOLISTICARE_MIN_COVERAGE` (default 0.5).

**Rules**

- A follow-up has a leading phrase such as "what about" or "and", or three or fewer content tokens with a pronoun. Its retrieval query adds up to 8 tokens from earlier user questions. A question with two or fewer content tokens also gets these tokens.
- The model answers the original question. Only the retrieval query changes.

```mermaid
flowchart LR
    Q[/"Question + history"/] --> H{"History<br/>is empty?"}
    H -- "yes" --> SAME[/"Retrieval query =<br/>the question"/]
    H -- "no" --> F{"is_follow_up, or<br/>2 or fewer content tokens?"}
    F -- "no" --> SAME
    F -- "yes" --> WALK["Read earlier user messages,<br/>newest first"]
    WALK --> ADD["Add their new content tokens,<br/>up to 8"]
    ADD --> OUT[/"Retrieval query =<br/>question + added tokens"/]
```

---

## 7. Generation and the answer checks

**Purpose.** Write a short answer from the passages only, and check it before the user sees it.

```mermaid
flowchart TD
    IN[/"Question, passages, history"/] --> CM{"HOLISTICARE_LLM"}
    CM -- "extractive" --> EX["extractive_answer"]
    CM -- "ollama or openai" --> BM["build_messages"]
    BM --> CALL["chat.complete<br/>temperature, token cap"]
    CALL -- "error" --> NE["note: model error"]
    NE --> EX
    CALL -- "reply" --> CC{"check_citations<br/>gives reasons?"}
    CC -- "yes" --> NR["note: model answer refused"]
    NR --> EX
    CC -- "no" --> TXT["Model text,<br/>model name kept"]
    EX --> DOSE["remove_doses<br/>on each sentence"]
    TXT --> DOSE
    DOSE --> DN{"Sentence removed?"}
    DN -- "yes" --> NOTE["Add DOSE_NOTE"]
    DN -- "no" --> CIT["citations_for<br/>best_sentence quote per passage"]
    NOTE --> CIT
    CIT --> OUT[/"Answer, category ok"/]
```

| Input | Output |
|---|---|
| Question, passages, history | `Answer`: text, category, citations with quotes, cautions, notes, retrieved sources, model name, disclaimer |

**Procedure**

1. Build the messages: the constant system message, the last `HOLISTICARE_HISTORY_TURNS` turns, then the numbered passages and the question.
2. Send the messages to the chat model, if one is set.
3. Check the reply with `check_citations`.
4. If there is no chat model, or the reply fails, or the model raises an error, make the extractive answer.
5. Remove dose sentences and add a note if a sentence was removed.
6. For each cited passage, select the passage sentence with the largest token overlap with the citing sentences.

| Extractive answer rule | Value |
|---|---|
| Sentences per passage | 1 (the sentence with the largest overlap with the question) |
| Maximum sentences | 4 |
| Parts | evidence-based, unlabelled, traditional (with the evidence statement) |
| Citation form | `sentence [n].` |

```mermaid
flowchart LR
    P[/"Passages in rank order"/] --> R["Next passage: rank its sentences<br/>by overlap with the question"]
    R --> PICK{"One of the top 2 has overlap<br/>above 0 and is not picked?"}
    PICK -- "yes" --> ADD["Keep that sentence<br/>with citation [n]"]
    PICK -- "no" --> MORE{"More passages?"}
    ADD --> MAX{"4 sentences?"}
    MAX -- "no" --> MORE
    MORE -- "yes" --> R
    MAX -- "yes" --> GRP["Group by evidence label"]
    MORE -- "no" --> GRP
    GRP --> EB["From evidence-based sources"]
    GRP --> UN["From sources without<br/>an evidence label"]
    GRP --> TR["From traditional-medicine sources<br/>+ TRADITIONAL_NOTE"]
    EB --> OUT[/"Extractive answer"/]
    UN --> OUT
    TR --> OUT
```

| Citation check | Result |
|---|---|
| No sentence with three or more content tokens | Refused: empty answer |
| A citation number outside 1..number of passages | Refused |
| Fewer than 80 % of the sentences have a citation | Refused |

```mermaid
flowchart TD
    IN[/"Model reply, number of passages"/] --> S["Sentences with 3 or more<br/>content tokens"]
    S --> E{"No sentence?"}
    E -- "yes" --> R1[/"Refused: empty answer"/]
    E -- "no" --> N{"Citation number outside<br/>1 to number of passages?"}
    N -- "yes" --> R2["Reason: passages<br/>that do not exist"]
    N -- "no" --> C{"Cited share<br/>below 0.8?"}
    R2 --> C
    C -- "yes" --> R3["Reason: too few<br/>cited sentences"]
    C -- "no" --> D{"Any reason?"}
    R3 --> D
    D -- "yes" --> REF[/"Refused, use the extractive answer"/]
    D -- "no" --> PASS[/"Passes"/]
```

---

## 8. The safety rules

`safety.triage` checks the lower-case question against the pattern lists in a fixed order. The first match stops the question.

```mermaid
flowchart TD
    Q[/"Question, lower case"/] --> SH{"_SELF_HARM<br/>pattern?"}
    SH -- "yes" --> R1[/"self_harm<br/>SELF_HARM_TEXT"/]
    SH -- "no" --> EM{"_EMERGENCY<br/>pattern?"}
    EM -- "yes" --> R2[/"emergency<br/>EMERGENCY_TEXT"/]
    EM -- "no" --> SM{"_STOP_MED<br/>pattern?"}
    SM -- "yes" --> R3[/"stop_medication<br/>STOP_MEDICATION_TEXT"/]
    SM -- "no" --> DO{"_DOSING<br/>pattern?"}
    DO -- "yes" --> R4[/"dosing<br/>DOSING_TEXT"/]
    DO -- "no" --> CA["Check _PREGNANCY<br/>and _CHILD patterns"]
    CA --> OK[/"ok, with 0, 1 or 2 cautions<br/>retrieval continues"/]
    R1 --> STOP["stop: no retrieval,<br/>no model call, + DISCLAIMER"]
    R2 --> STOP
    R3 --> STOP
    R4 --> STOP
```

| Category | Example trigger | Result | Retrieval or model call? |
|---|---|---|---|
| `self_harm` | "want to die", "kill myself", "self-harm" | Fixed crisis reply with crisis line examples | No |
| `emergency` | "chest pain", "cannot breathe", "slurred speech", "overdose", "seizure" | Fixed reply: call the local emergency number | No |
| `stop_medication` | "stop taking my insulin", "instead of chemotherapy" | Fixed reply: do not stop or replace a treatment without the doctor | No |
| `dosing` | "how many mg", "dose", "dosage", "500 mg" | Fixed reply: ask a doctor or a pharmacist | No |
| `ok` + `pregnancy` caution | "pregnant", "breastfeeding" | Answer plus a fixed caution | Yes |
| `ok` + `child` caution | "child", "baby", "toddler", "my son" | Answer plus a fixed caution | Yes |
| `no_source` | No passage reaches the minimum coverage | Fixed no-source reply | Retrieval only |

**Rules**

- The rules run in this order: self-harm, emergency, stop-medication, dosing, cautions.
- The dose filter removes each answer sentence with an amount (`mg`, `ml`, `g`, `tablets`, `drops`, `teaspoons`, …) or a schedule ("twice a day").
- Every answer, also a fixed reply, has the disclaimer.
- The triage is a keyword rule set. It is not a clinical triage tool.

---

## 9. Data and file map

| Path | Committed? | Contents |
|---|---|---|
| `data/README.md` | Yes | How to add documents, registry fields, open sources, question-set format |
| `data/documents/` | No (git ignores it) | Your documents and `sources.json` (default of `HOLISTICARE_DOCS_DIR`) |
| `data/synthetic/documents/` | No (git ignores it) | Fictional knowledge base from `holisticare synth` |
| `data/synthetic/eval.jsonl` | No (git ignores it) | Synthetic question set (18 questions) |
| `data/synthetic/eval_results.csv` | No (git ignores it) | Per-question results of `holisticare demo` |
| `.index/chunks.jsonl` | No (git ignores it) | Chunks with source, section, page, evidence label |
| `.index/vectors.npy` | No (git ignores it) | Chunk vectors (float32) |
| `.index/manifest.json` | No (git ignores it) | Format version, fingerprint, embedder, counts, chunk settings |
| `.env` | No (git ignores it) | Local settings and `HOLISTICARE_LLM_API_KEY` |

---

## 10. How to run holisticare-rag

### 10.1 Prerequisites

| Need | For |
|---|---|
| Python 3.11+ | All components |
| numpy, pandas, scikit-learn | Core (installed with the package) |
| Ollama with a model (for example `ollama pull llama3.1`) | `HOLISTICARE_LLM=ollama` |
| An OpenAI-compatible server and `HOLISTICARE_LLM_API_KEY` | `HOLISTICARE_LLM=openai` |
| `pypdf` (extra `pdf`) | PDF documents |
| `sentence-transformers` (extra `st`) | `HOLISTICARE_EMBEDDER=sentence-transformers` |
| `streamlit` (extra `ui`) | The chat page |

### 10.2 Installation

```bash
git clone https://github.com/KrishnaAnnavaram/holisticare-rag.git
cd holisticare-rag
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"         # add ,pdf ,st ,ui as necessary
```

### 10.3 Run holisticare-rag

```bash
# Offline demo: fictional knowledge base, three questions, the question set (a few seconds)
holisticare demo

# Fictional knowledge base, then each step
holisticare synth --out-dir data/synthetic
export HOLISTICARE_DOCS_DIR=data/synthetic/documents     # Windows PowerShell: $env:HOLISTICARE_DOCS_DIR="data/synthetic/documents"
holisticare index
holisticare ask "How is Tarsil cough treated?"
holisticare ask "How many mg of Kelvar root should I take?" --json
holisticare eval --questions data/synthetic/eval.jsonl --out reports/eval.csv
holisticare chat

# Your documents in data/documents with sources.json, generated with a local Ollama model
HOLISTICARE_LLM=ollama HOLISTICARE_LLM_MODEL=llama3.1 holisticare ask "What is asthma?"
holisticare ui
pytest -q
```

| Command | Result |
|---|---|
| `holisticare synth` | Writes the fictional knowledge base and the question set |
| `holisticare index [--force]` | Builds, rebuilds or reuses the index and prints the status |
| `holisticare ask "<question>"` | Prints the answer, the cautions, the sources with quotes, the notes and the disclaimer |
| `holisticare chat` | Interactive chat with history (an empty line ends it) |
| `holisticare eval` | Prints per-question results and the summary metrics |
| `holisticare ui` | Starts the Streamlit chat page |
| `holisticare demo` | Runs `synth`, `index`, three questions and `eval` |

`holisticare demo` runs these steps in this order, with the extractive answer and a new index:

```mermaid
flowchart LR
    S["synthetic.write<br/>documents + eval.jsonl"] --> I["cmd_index, force<br/>out-dir/index"]
    I --> A1["ask: Tarsil cough<br/>answer"]
    A1 --> A2["ask: mg of Kelvar root<br/>dosing reply"]
    A2 --> A3["ask: bicycle brakes<br/>no-source reply"]
    A3 --> E["cmd_eval<br/>18 questions"]
    E --> OUT[/"Summary metrics +<br/>eval_results.csv"/]
```

### 10.4 Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `HOLISTICARE_DOCS_DIR` | index | Document folder, default `data/documents` |
| `HOLISTICARE_INDEX_DIR` | index | Index folder, default `.index` |
| `HOLISTICARE_CHUNK_CHARS` | index | Maximum chunk length in characters, default 900 (minimum 200) |
| `HOLISTICARE_CHUNK_OVERLAP` | index | Overlap between chunks, default 120 |
| `HOLISTICARE_EMBEDDER` | index, retrieval | `hashing` (default), `sentence-transformers` or `ollama` |
| `HOLISTICARE_EMBED_MODEL` | index, retrieval | Model name for the two model embedders, default `sentence-transformers/all-MiniLM-L6-v2` |
| `HOLISTICARE_TOP_K` | retrieval | Passages per question, default 4 |
| `HOLISTICARE_MIN_COVERAGE` | retrieval | Relevance gate, default 0.5 |
| `HOLISTICARE_HISTORY_TURNS` | prompts | History turns sent to the model, default 6 |
| `HOLISTICARE_LLM` | generation | `extractive` (default), `ollama` or `openai` |
| `HOLISTICARE_LLM_MODEL` | generation | Model name, default `llama3.1` |
| `HOLISTICARE_LLM_BASE_URL` | generation, Ollama vectors | Server URL, default `http://localhost:11434` (Ollama) or `https://api.openai.com/v1` |
| `HOLISTICARE_LLM_API_KEY` | generation | Key for `openai` (credential) |
| `HOLISTICARE_TEMPERATURE` | generation | Temperature, default 0.1, maximum 0.3 |
| `HOLISTICARE_MAX_TOKENS` | generation | Token cap (`num_predict` for Ollama), default 512 |
| `HOLISTICARE_TIMEOUT_S` | generation, vectors | HTTP timeout in seconds, default 60 |

The settings come from the environment and from a local `.env` file. An environment variable wins over the `.env` file. Credentials are only in a local `.env` file. Git ignores this file. Do not print or commit credentials.

---

## 11. How to extend holisticare-rag

| You want to… | Do this | Code change? |
|---|---|---|
| Add documents | Put the files in the document folder, add entries to `sources.json`, run `holisticare index` | No |
| Use a local model | Install Ollama, pull a model, set `HOLISTICARE_LLM=ollama` | No |
| Use better vectors | Install the extra `st` and set `HOLISTICARE_EMBEDDER=sentence-transformers` | No |
| Add a triage rule | Add a pattern to the lists in `safety.py` and a test in `tests/test_safety.py` | Small |
| Add a query synonym | Add it to `QUERY_SYNONYMS` in `text.py` | Small |
| Add a reranker | Sort the passages in `retrieve.py` with a cross-encoder before the cut to `top_k` | Yes |
| Add a model-based triage | Add a classifier after the rules in `safety.triage`. Keep the rules first | Yes |
| Add a clinician-reviewed question set | Write a JSON Lines file (see `data/README.md`) and run `holisticare eval` | No |

---

## 12. Validation results

| Validation | Result | Command |
|---|---|---|
| Unit tests (CI installs only `.[dev]`) | **44 passed, 1 skipped** (the Streamlit test) | `pytest -q` |
| Unit tests with the `ui` extra (Streamlit) | **45 passed** | `pytest -q` |
| Question set on the fictional knowledge base | See the table below | `holisticare demo` |

The demo uses the fictional knowledge base (7 documents, 28 chunks, hashing vectors) and the extractive answer. The question set has 18 questions: 11 answerable, 2 off-topic and 5 triage cases. The answerable questions include 2 follow-ups, 2 with injection text or braces and 1 with a pregnancy caution. **These numbers come from synthetic, fictional text.** They show that the pipeline and the rules work. They do not show the quality on real medical documents.

`evaluate.evaluate` asks each question of the set and calculates the metrics from the answers:

```mermaid
flowchart LR
    QS[/"eval.jsonl: question, expect,<br/>sources, history"/] --> ASK["Assistant.ask<br/>for each item"]
    ASK --> ROW["Row: category_ok, hit,<br/>cited, cited_expected, supported"]
    ROW --> DF["pandas table"]
    DF --> SUM["Summary: category_accuracy,<br/>retrieval_hit@k, citation_precision,<br/>supported_sentences, refusal precision and recall"]
    DF --> CSV[/"eval_results.csv"/]
    SUM --> SP["system_prompt_unchanged<br/>SYSTEM compared again"]
```

| Metric (synthetic question set) | Value |
|---|---|
| Category accuracy (18 questions) | 1.000 |
| Retrieval hit@4 (11 answerable questions) | 1.000 |
| Citation precision | 0.833 |
| Supported sentences | 1.000 (by construction for extractive answers, meaningful for model answers) |
| Refusal precision / recall (no-source path) | 1.000 / 1.000 |
| System message unchanged after the injection questions | Yes |

The citation precision is below 1, because some answers also cite a related document that the question set does not list. For example, a question about Kelvar root and Tarsil cough also cites the Tarsil cough page.

---

## 13. Known problems

Read these problems before you show holisticare-rag to anyone.

| # | Area | Problem | Impact and action |
|---|---|---|---|
| 1 | Responsible use | holisticare-rag is not a medical device and not a decision tool. | A qualified health professional must review every decision. Show the disclaimer to every user. |
| 2 | Evaluation | The question set is synthetic. No clinician reviewed real answers. | Before any wider use, a clinician must review a question set for your documents. |
| 3 | Triage | The triage is a keyword rule set. It misses unusual wording and other languages, and it can stop a harmless question. | Keep the rules, and add a reviewed classifier. Test new rules with red-team questions. |
| 4 | Sources | Answers are only as good as the indexed documents. Traditional-medicine texts can describe uses with no clinical support, and documents can show bias. | Label every document in `sources.json`. Prefer reviewed, current, openly licensed sources. |
| 5 | Retrieval | The default hashing vectors match words, not meaning. Synonyms outside `QUERY_SYNONYMS` can fail the relevance gate. | Use `sentence-transformers` vectors for real documents and check the gate value. |
| 6 | Model answers | The citation check tests the form of the citations, not the truth of each claim. | Read the quotes. Keep the temperature low. |
| 7 | Dose filter | The dose filter is a pattern rule. It can miss an amount in unusual words. | Do not index documents with dosing tables if you can avoid it. |
| 8 | PDF | Scanned PDFs without a text layer give no text. There is no OCR. | Use text PDFs or convert them to text first. |
| 9 | Privacy | The default setup is local. With `HOLISTICARE_LLM=openai`, questions go to that server. | Use a local server for personal health questions. |

---

## 14. Key points

1. **Safety comes first.** Emergency, self-harm, dosing and stop-medication questions get fixed replies with no generation.
2. **No source means no answer.** The relevance gate refuses questions that the documents do not cover.
3. **User text is data.** The system message is constant, and the history goes in as messages.
4. **Each sentence has a citation.** Each citation has a supporting quote from the passage.
5. **Evidence labels stay visible.** Traditional use is never shown as clinical evidence.
6. **The index is safe to load.** It has no pickle, and a fingerprint keeps it current.
7. **A professional reviews decisions.** Every answer has the disclaimer.

---

## 15. Glossary

| Term | Meaning |
|---|---|
| **Caution** | A fixed warning added to an answer for pregnancy or children |
| **Chunk** | A part of one section of one document, with its metadata |
| **Citation** | A passage number in square brackets at the end of a sentence |
| **Content token** | A lower-case word that is not a stopword |
| **Coverage** | The share of the query content tokens that a passage contains |
| **Disclaimer** | The fixed text that every answer has |
| **Evidence label** | `evidence_based`, `traditional` or `unknown`, from `sources.json` |
| **Extractive answer** | An answer made of passage sentences, with citations, and no model |
| **Fingerprint** | The SHA-256 value of the documents, the registry, the chunk settings and the embedder |
| **Fixed reply** | The text that a stopping triage category returns |
| **Follow-up** | A question that needs the earlier topic to be clear |
| **Passage** | A chunk that the assistant gives to the model or to the extractive answer, with a number |
| **Quote** | The passage sentence that best supports the citing sentences |
| **Registry** | The file `sources.json` with one entry per document |
| **Relevance gate** | The rule that refuses a question when no passage reaches the minimum coverage |
| **Triage** | The keyword rules that run before retrieval |

---

## 16. License

[MIT](LICENSE) © 2026 Krishna Annavaram
