# RAG System for Non-Invasive DNA Sampling Protocol Analysis

A Retrieval-Augmented Generation (RAG) pipeline designed to automatically extract, classify, and evaluate DNA sampling protocols from scientific articles, with a focus on detecting invasiveness according to Taberlet et al. (1999).

Developed during an L3 internship.

---

## Overview

This system processes scientific papers in TEI-XML format (produced by GROBID), extracts relevant sections, indexes them using semantic embeddings, and queries a local LLM to:

- Identify DNA sampling protocols described in each article
- Evaluate whether each protocol is invasive, minimally invasive, or non-invasive
- Detect cases where authors classify an invasive protocol as non-invasive

---

## Project Structure

```
RAG/
├── config.json                  # GROBID server configuration
├── requirements.txt             # Python dependencies
│
├── Chunking_intro.py            # Extracts title, date, authors, abstract from TEI-XML
├── Chunking_protocole.py        # Extracts relevant sections (methods, etc.) from TEI-XML
├── Division.py                  # Splits merged chunk files into per-article files
├── Prépa_data.py                # Converts chunks to JSON for indexing
├── Excel.py                     # Exports article metadata to Excel
│
├── UniversityLLMAdapter.py      # Haystack component wrapping a local OpenAI-compatible LLM API
├── components.py                # Custom Haystack components (HyDE, dynamic threshold)
├── pipelines.py                 # Haystack pipeline definitions (HyDE, indexing)
├── prompts.py                   # All LLM prompts (protocol detection, invasivity analysis, etc.)
├── rag_system.py                # Core RAG classes (RAGSystem, RefineRAGSystem, RAGNonInvasiveDetection)
├── mainRag.py                   # Entry point: parallel processing of all JSON files
└── LLM_AAJ.py                   # LLM-based matching of results against ground truth
```

---

## Pipeline Overview

### 1. Data Preparation

**Input:** TEI-XML files produced by [GROBID](https://github.com/kermitt2/grobid)

```
results/*.xml
    -> Chunking_protocole.py  -> Chunks/chunking.txt
    -> Division.py            -> ChunksDivise/<article>.txt
    -> Prépa_data.py          -> output/<article>.json
```

Two chunking modes are available:
- **Standard:** Extracts methods-related sections, excludes introduction, results, discussion, etc.
- **Invasive detection:** Includes title, abstract, and additional sections to support the secondary analysis task.

### 2. RAG Analysis

**Input:** `output/*.json`

The main pipeline (`mainRag.py`) processes each article through `RefineRAGSystem`:

1. **HyDE retrieval** - generates hypothetical documents to improve semantic search
2. **Protocol detection** - identifies sampling protocol descriptions in retrieved chunks
3. **Extract fusion** - merges redundant extracts describing the same protocol
4. **Invasivity analysis** - classifies each protocol using detailed criteria and the "seven sins" framework
5. **Final fusion** - consolidates all analyses into a structured JSON output

For protocols classified as invasive, a second system (`RAGNonInvasiveDetection`) checks whether the authors themselves present the protocol as non-invasive.

**Output:** `Resultats/Protocoles.csv`

### 3. Evaluation

`LLM_AAJ.py` compares extracted sample types against a ground truth Excel file using fuzzy title matching and LLM-based verdict generation.

---

## Installation

```bash
pip install -r requirements.txt
```

The system uses [allenai-specter](https://huggingface.co/sentence-transformers/allenai-specter) as the embedding model and requires access to a local OpenAI-compatible LLM API.

---

## Usage

```bash
python mainRag.py \
  --api_key YOUR_API_KEY \
  --api_url http://your-llm-server/api/chat/completions \
  --input_dir output \
  --output_dir Resultats \
  --top_k 8 \
  --max_workers 4
```

All arguments can also be set via environment variables (`LLM_API_KEY`, `LLM_API_URL`).

---

## Key Design Choices

- **HyDE (Hypothetical Document Embeddings):** Instead of embedding the raw query, a hypothetical answer is generated and embedded, improving retrieval relevance on scientific text.
- **Dynamic similarity threshold:** Retrieved documents are filtered based on the score distribution rather than a fixed cutoff.
- **Refine pattern:** Analysis is split into multiple focused LLM calls (detect, fuse, evaluate, consolidate) rather than a single large prompt.

---

## Dependencies

- [Haystack](https://github.com/deepset-ai/haystack)
- [sentence-transformers](https://www.sbert.net/)
- [GROBID](https://github.com/kermitt2/grobid) (for TEI-XML generation, run separately)
- [ecologits](https://github.com/genai-impact/ecologits)
- [rapidfuzz](https://github.com/maxbachmann/RapidFuzz)
- pandas, numpy, lxml, nltk, openpyxl
