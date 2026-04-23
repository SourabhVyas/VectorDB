# VectorDB Embedding Evaluation Pipeline

This repository builds and evaluates text embeddings for two related tasks:

- Classification embeddings for fake-news style label prediction.
- Retrieval embeddings for FAISS-based semantic search (RAG-style lookup).

The main workflow is implemented in a notebook and reusable source modules.

## What This Project Does

1. Preprocesses raw records from a unified JSON dataset.
2. Extracts classifier and retrieval embeddings from multiple model families.
3. Fine-tunes base classifier models (optional).
4. Re-extracts embeddings from fine-tuned checkpoints.
5. Evaluates:
   - Classification metrics (Accuracy, F1, ROC-AUC, PR-AUC)
   - Retrieval metrics (MRR, Recall@k)
6. Writes a summary CSV under artifacts.
7. Builds a FAISS index and runs example similarity queries.

## Repository Layout

- `src/preprocess.py`: Cleans/normalizes records, extracts labels, and writes preprocessed JSON.
- `src/embeddings.py`: Embedding extraction for classifier and retrieval tasks, plus FAISS index builder.
- `src/ft_classify.py`: Hugging Face Trainer-based fine-tuning for sequence classification.
- `src/eval_utils.py`: Classification and retrieval evaluation helpers, plus deferral simulation.
- `notebooks/embeddings.ipynb`: End-to-end experiment notebook.
- `artifacts/embedding_eval_summary.csv`: Evaluation summary output.

## Data Expectations

The notebook expects these paths (relative to repository root):

- Input: `dataset/unified_facts.json`
- Preprocessed output: `dataset/preprocessed.json`
- Cache/output directory: `cache/`
- Metrics artifact: `artifacts/embedding_eval_summary.csv`

Each input record should include at least:

- `statement`
- `justification` (or `justifications`)
- Label metadata inside `meta` (or `metadata`) using one of the supported label keys.

Label mapping used in preprocessing:

- `False` -> 0
- `True` -> 1
- `Not sure` -> 2

## Environment Setup

Use Python 3.10+ (3.11 recommended).

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install numpy torch tqdm transformers sentence-transformers faiss-cpu scikit-learn datasets nltk
```

Optional (only if needed by your workflow):

```bash
pip install spacy requests beautifulsoup4 readability-lxml
python -m spacy download en_core_web_sm
```

## Quick Start (Notebook)

1. Open `notebooks/embeddings.ipynb`.
2. Ensure `dataset/unified_facts.json` exists.
3. Run all cells in order.

The notebook will:

- Preprocess records.
- Compute embeddings for classifier and retrieval candidates.
- Fine-tune `bert-base-uncased` and `roberta-base`.
- Recompute embeddings from fine-tuned checkpoints.
- Evaluate all candidates.
- Save CSV summary to `artifacts/embedding_eval_summary.csv`.
- Build/search a FAISS index with sample queries.

## Current Artifact Snapshot

From `artifacts/embedding_eval_summary.csv` currently in this repo:

- Best classifier row: fine-tuned `roberta-base` checkpoint with accuracy ~0.823 and ROC-AUC ~0.877.
- Retrieval candidates show near-perfect self-retrieval metrics (MRR ~= 0.999, Recall@1 ~= 0.998, Recall@5 = 1.0).

## Programmatic Usage

You can also import modules directly:

```python
from src.preprocess import preprocess
from src.embeddings import get_classifier_embeddings, get_rag_embeddings
from src.eval_utils import eval_embedding_clf, rag_self_retrieval_eval

records = preprocess("dataset/unified_facts.json", "dataset/preprocessed.json", max_j_sents=5)
clf_embs, clf_ids = get_classifier_embeddings("bert-base-uncased", "dataset/preprocessed.json", out_dir="cache")
rag_embs, rag_ids = get_rag_embeddings("sentence-transformers/all-mpnet-base-v2", "dataset/preprocessed.json", out_dir="cache")

clf_stats, _ = eval_embedding_clf(clf_embs, clf_ids, "dataset/preprocessed.json")
rag_stats = rag_self_retrieval_eval(rag_embs, rag_ids, topk=5)

print(clf_stats)
print(rag_stats)
```

## Notes

- Embeddings and IDs are cached under `cache/embeddings/...`.
- If a model path exists locally, loaders prefer local checkpoint files.
- Retrieval embeddings are L2-normalized before FAISS inner-product indexing.
