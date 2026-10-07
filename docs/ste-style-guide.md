# ASD-STE100 Simplified Technical English: the standard for this repository

Use these rules for every README and for `docs/ste-style-guide.md` in each repository. Copy this file
into the repository as `docs/ste-style-guide.md` and add a **project vocabulary** section (Section 3)
with the technical names and technical verbs of that project.

## 1. Rules for the text

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, "test" is a noun or a verb, "check" is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: "prepare", "do", "find", "get", "make".
4. Do not use an "-ing" form as a noun or an adjective ("the running job", "after indexing").
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and "can" for a possibility.
8. Keep the articles "a", "an" and "the" in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: "If the index is stale, build it again."
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase ("The cost model") or an imperative ("Run the demo").
   Do not start a heading with an "-ing" form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or "check that" |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

These terms have one meaning in the holisticare-rag documentation. Code names are in backticks.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **document** | One `.md`, `.txt` or `.pdf` file in the document folder. | file (for a source), book, paper (in prose) |
| **document folder** | The folder that the index reads (`HOLISTICARE_DOCS_DIR`). | corpus folder, library |
| **registry** | The file `sources.json` with one entry per document. | catalogue, metadata file |
| **evidence label** | `evidence_based`, `traditional` or `unknown`, from the registry. | evidence level, tag, category |
| **tradition** | The free-text field of a registry entry, for example `ayurveda`. | school, system |
| **chunk** | A part of one section of one document, with its metadata. | segment, piece, node |
| **index** | The files `chunks.jsonl`, `vectors.npy` and `manifest.json` in the index folder. | vector store, database, FAISS index |
| **fingerprint** | The SHA-256 value of the documents, the registry, the chunk settings and the embedder. | hash (alone), checksum |
| **embedder** | The component that changes text into a vector. | encoder, embedding model (in prose) |
| **content token** | A lower-case word that is not a stopword. | keyword, term |
| **retrieval query** | The text that retrieval uses: the question, plus earlier topic tokens for a follow-up. | rewritten question, search string |
| **follow-up** | A question that needs the earlier topic to be clear. | continuation, context question |
| **passage** | A chunk that the assistant gives to the model or to the extractive answer, with a number. | context, snippet, document (for a chunk) |
| **coverage** | The share of the query content tokens that a passage contains. | relevance, match score |
| **relevance gate** | The rule that refuses a question when no passage reaches the minimum coverage. | threshold check, filter |
| **triage** | The keyword rules that run on each question before retrieval. | screening, classifier |
| **category** | The result of the triage or of the gate: `ok`, `no_source`, `emergency`, `self_harm`, `dosing`, `stop_medication`. | intent, label, class |
| **fixed reply** | The text that a stopping category returns. | canned answer, template |
| **caution** | A fixed warning added to an answer for pregnancy or children. | alert, note (for a caution) |
| **chat model** | The optional model that writes the answer (Ollama or OpenAI-compatible). | LLM (in prose), bot, AI |
| **extractive answer** | An answer made of passage sentences, with citations, and no model. | fallback answer, summary |
| **citation** | A passage number in square brackets at the end of a sentence. | reference, footnote, source tag |
| **quote** | The passage sentence that best supports the citing sentences. | evidence text, excerpt |
| **dose filter** | The rule that removes answer sentences with an amount or a schedule. | dose guard, sanitiser |
| **disclaimer** | The fixed text that every answer has. | warning (for the disclaimer), notice |
| **note** | A line that tells the user what a check did, for example a refused model answer. | log, message |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **index** | Read the documents, make chunks and vectors, and save them with the fingerprint. |
| **reuse** | Load the saved index because the fingerprint did not change. |
| **retrieve** | Get the passages for a retrieval query. |
| **fuse** | Make one ranked list from the BM25 list and the vector list. |
| **triage** | Give a question its category with the keyword rules. |
| **refuse** | Give a fixed reply or the no-source reply, or reject a model answer. |
| **generate** | Get an answer from the chat model. |
| **extract** | Make the extractive answer from the passages. |
| **cite** | Put a passage number at the end of a sentence. |
| **evaluate** | Run the question set and calculate the metrics. |
