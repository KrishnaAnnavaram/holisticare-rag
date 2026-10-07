# Data

Git does not track the files in this folder, only this README. No document, index or health text is in the
repository.

## Synthetic knowledge base (default, no download)

```bash
holisticare synth --out-dir data/synthetic
```

The generator (`src/holisticare_rag/synthetic.py`) writes `data/synthetic/documents/` with `sources.json`, and
`data/synthetic/eval.jsonl`. All conditions and plants in it (for example "Velmora rash" and "Kelvar root") are
**fictional**. The text is not medical information. The tests and `holisticare demo` use it.

## Your own documents

Put `.md`, `.txt` or `.pdf` files (PDF needs the extra `pdf`) under `data/documents/` (or set
`HOLISTICARE_DOCS_DIR`). Sub-folders are allowed. Add one entry per file to `data/documents/sources.json`:

```json
[
  {"file": "general/asthma_medlineplus.md", "title": "Asthma", "license": "Public domain (U.S. NLM)",
   "url": "https://medlineplus.gov/asthma.html", "evidence": "evidence_based", "tradition": "conventional"},
  {"file": "ayurveda/review_2020.pdf", "title": "Review of an Ayurvedic therapy", "license": "CC BY 4.0",
   "url": "https://www.ncbi.nlm.nih.gov/pmc/", "evidence": "traditional", "tradition": "ayurveda"}
]
```

| Field | Required | Values |
|---|---|---|
| `file` | Yes | Path relative to the document folder, with `/` |
| `title` | Yes | Title shown in the citations |
| `license` | No | The license or terms of the document |
| `url` | No | Where the document comes from |
| `evidence` | No | `evidence_based`, `traditional` or `unknown` (default) |
| `tradition` | No | Free text, for example `conventional`, `ayurveda`, `homeopathy`, `herbal` |

Openly licensed starting points:

| Source | URL | Terms |
|---|---|---|
| MedlinePlus health topics | https://medlineplus.gov/ | Public domain (U.S. National Library of Medicine), except marked third-party content |
| NCBI Bookshelf open-access titles | https://www.ncbi.nlm.nih.gov/books/ | License per title (many are CC BY or CC BY-NC) |
| PubMed Central open-access subset | https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/ | License per article |
| WHO publications | https://www.who.int/publications | CC BY-NC-SA 3.0 IGO for most titles |

Do not add copyrighted books unless you have the right to use them. Keep the URL and the license of each file in
`sources.json`.

## Question set

`holisticare eval --questions <file>` reads JSON Lines. Each line has `question`, `expect` (`ok`, `no_source`,
`emergency`, `self_harm`, `dosing` or `stop_medication`), `sources` (files that must support an `ok` answer) and an
optional `history` (a list of `{"role", "content"}` messages). A clinician must review a question set for real
documents before you trust its results.
