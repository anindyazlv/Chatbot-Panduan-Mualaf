# Mualaf Guide Chatbot

A retrieval-augmented generation (RAG) chatbot that answers beginner questions about the basics of Islam, grounded only in the book *The Revealed Path: A Guide Book for New Muslims* (Second Edition, 2014) derived from https://beginnings.org.uk/downloads/English-E-Book-1-Guide-Book-for-New-Muslims.pdf. Built with LlamaIndex, Hugging Face models, and Streamlit, and designed to run on a CPU.

> **Disclaimer:** This chatbot only summarizes the guidebook. It is not a fatwa and does not replace an Islamic teacher or mentor.

## Features

- Answers questions using only retrieved passages from the guidebook; if nothing relevant is found, it returns a fallback message instead of guessing.
- Shows the source passages for every answer, together with the original PDF page rendered as an image (so Arabic text stays intact).
- Versioned indexes: every run of `build_index.py` creates a new `indexes/vN` folder with a `manifest.json` (embedding model, chunk settings, cleaning version, page count, PDF hash).
- Adjustable top-k and similarity cutoff from the Streamlit sidebar.
- Retrieval-only mode (language model switched off) to inspect what the index returns.
- Retrieval evaluation script with Hit@k, MRR, and a similarity-cutoff sweep.

## How it works

```
PDF -> PyMuPDF text extraction -> cleaning (preprocess.py)
    -> chunking -> embeddings (multilingual-e5-base) -> vector index (indexes/vN)

Question -> retrieve top-k chunks -> drop chunks below similarity cutoff
         -> Qwen2.5-0.5B-Instruct answers from the context -> answer + sources
```

| Component | Choice |
|---|---|
| Framework | LlamaIndex |
| Embedding model | `intfloat/multilingual-e5-base` |
| Language model | `Qwen/Qwen2.5-0.5B-Instruct` |
| PDF reader | PyMuPDF |
| UI | Streamlit |

## Project structure

```
.
├── data/                 # source PDF
├── indexes/              # versioned vector indexes (v1, v2, ...)
├── build_index.py        # builds a new versioned index from the PDF
├── config.py             # models, chunking, retrieval, and path settings
├── eval_retrieval.py     # retrieval evaluation (Hit@k, MRR, cutoff sweep)
├── main_app.py           # Streamlit chat app
├── preprocess.py         # PDF text cleaning
├── qa_set.json           # evaluation questions with expected keywords
└── requirements.txt
```

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/anindyazlv/Mualaf-Guide-Chatbot.git
cd Mualaf-Guide-Chatbot

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

The first run downloads the embedding and language models from Hugging Face, so an internet connection is needed once. The app uses a GPU automatically if CUDA is available and falls back to CPU otherwise. For a CPU-only install, you can install PyTorch from the CPU wheel index first (`pip install torch --index-url https://download.pytorch.org/whl/cpu`).

## Usage

### 1. Build the index

Place the PDF in `data/` with the file name set in `config.py` (`PDF_NAME`), then run:

```bash
python build_index.py
```

This creates the next version automatically (`indexes/v1`, then `v2`, and so on).

### 2. Run the app

```bash
streamlit run main_app.py
```

In the sidebar you can pick the index version, turn the language model on or off, and change top-k and the similarity cutoff.

### 3. Evaluate retrieval

```bash
python eval_retrieval.py v1
```

Replace `v1` with the index version to evaluate.

## Configuration

Settings live in `config.py`:

| Setting | Meaning |
|---|---|
| `LLM_MODEL` | Hugging Face model used to compose answers |
| `EMBEDDING_MODEL` | Hugging Face model used for embeddings |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | Chunking parameters used when indexing |
| `TOP_K` | Number of chunks retrieved per question |
| `SIM_CUTOFF` | Minimum similarity score for a chunk to be used (currently `0.82`, chosen from the cutoff sweep) |
| `INDEX_ROOT`, `DATA_DIR` | Index and data folders |
| `PDF_NAME`, `EDITION` | Source PDF file name and edition label |
| `CLEANING_VERSION` | Label recorded in the manifest (currently `c3`); bump it when `preprocess.py` changes |
| `FALLBACK` | Message returned when no relevant passage is found |

If you change the PDF, chunking, embedding model, or `preprocess.py`, rebuild the index. Changing only `SIM_CUTOFF`, `TOP_K`, or the language model does not require a rebuild.

## Evaluation

`qa_set.json` is a list of questions. In-scope questions have `keywords` that must appear in a retrieved chunk; out-of-scope questions (topics the book does not cover) have `"keywords": null`.

```json
[
  {"question": "What percentage of wealth is paid as zakat?", "keywords": ["2.5 percent"]},
  {"question": "How many rakats are in the Maghrib prayer?", "keywords": null}
]
```

The script reports:

- **Hit@k**: share of in-scope questions whose keyword appears in the top k chunks.
- **MRR**: mean reciprocal rank of the first matching chunk.
- **Average top score** for in-scope vs. out-of-scope questions.
- **Cutoff sweep**: for several `SIM_CUTOFF` values, the share of in-scope questions kept and out-of-scope questions rejected.

Latest results on 27 in-scope and 8 out-of-scope questions (top-k = 4, `multilingual-e5-base`, cleaning version `c4`):

| Metric | Value |
|---|---|
| Hit@1 | 77.78% |
| Hit@3 | 92.59% |
| Hit@4 | 96.30% |
| MRR | 0.849 |

Similarity scores for the two groups overlap (in-scope minimum 0.811, out-of-scope maximum 0.828), so no cutoff separates them perfectly. Re-run the sweep and choose `SIM_CUTOFF` again whenever the index or embedding model changes.

## Known limitations

- The language model is small (0.5B parameters) and does not always follow the "answer only from the context" instruction, so check answers against the cited source.
- The source PDF uses custom transliteration fonts, so raw extracted text contains garbled characters. `preprocess.py` fixes the cases found so far; new ones may need to be added.
- The guidebook is conceptual and does not cover detailed rulings (for example, number of rakats, tayammum, or Islamic insurance). Such questions should return the fallback message.
- The evaluation set is small, so the percentages are rough.
