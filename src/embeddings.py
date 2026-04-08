import os
from pathlib import Path
import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, DPRQuestionEncoder, AutoTokenizer as HFTokenizer
from sentence_transformers import SentenceTransformer
import faiss

def safe_name(model_id: str) -> str:
    try:
        p = Path(model_id)
        if p.parts and (p.exists() or str(model_id).find(os.sep) != -1 or str(model_id).find("/") != -1):
            return p.name.replace(":", "_")
    except Exception:
        pass
    return str(model_id).replace("/", "_").replace(":", "_")

def is_dpr_model(model_name: str) -> bool:
    mn = model_name.lower()
    return mn.startswith("facebook/dpr-") or "dpr-" in mn

def get_classifier_embeddings(model_id, preproc_json, out_dir, bs=64, max_len=256, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    emb_dir = out_dir / "embeddings" / "clf"
    emb_dir.mkdir(parents=True, exist_ok=True)

    name = safe_name(model_id)
    emb_path = emb_dir / f"clf_{name}.npy"
    ids_path = emb_dir / f"clf_{name}_ids.txt"

    if emb_path.exists() and ids_path.exists():
        return np.load(str(emb_path)), [l.strip() for l in open(ids_path, "r", encoding="utf8")]

    # Load tokenizer and model from local path if it exists, otherwise from HF hub
    p = Path(model_id)
    try:
        if p.exists():
            model_loc = str(p.resolve())
            print(f"Loading local checkpoint from {model_loc}")
            tok = AutoTokenizer.from_pretrained(model_loc, use_fast=True, local_files_only=True)
            model = AutoModel.from_pretrained(model_loc, local_files_only=True).to(device).eval()
        else:
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            model = AutoModel.from_pretrained(model_id).to(device).eval()
    except Exception as e:
        try:
            alt = str(p.resolve())
            tok = AutoTokenizer.from_pretrained(alt, use_fast=True, local_files_only=True)
            model = AutoModel.from_pretrained(alt, local_files_only=True).to(device).eval()
        except Exception:
            raise RuntimeError(f"Failed to load model/tokenizer for '{model_id}': {e}")

    records = json.load(open(preproc_json, "r", encoding="utf8"))
    texts = [r["statement"] for r in records]
    ids = [r["id"] for r in records]

    embs = []
    for i in tqdm(range(0, len(texts), bs), desc=f"clf-encode-{safe_name(model_id)}"):
        batch = texts[i:i+bs]
        enc = tok(batch, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, return_dict=True)
            if getattr(out, "pooler_output", None) is not None:
                pooled = out.pooler_output
            else:
                last = out.last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1)
                pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        embs.append(pooled.cpu().numpy())

    embs = np.vstack(embs).astype("float32")
    np.save(emb_path, embs)
    open(ids_path, "w", encoding="utf8").write("\n".join(ids))
    return embs, ids

def encode_with_transformers_dpr(model_name, texts, bs=32, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = HFTokenizer.from_pretrained(model_name, use_fast=True)
    model = DPRQuestionEncoder.from_pretrained(model_name).to(device).eval()
    embs = []
    for i in tqdm(range(0, len(texts), bs), desc=f"dpr-encode-{safe_name(model_name)}"):
        batch = texts[i:i+bs]
        enc = tokenizer(batch, truncation=True, padding=True, return_tensors="pt", max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, return_dict=True)
            vec = out.pooler_output
            embs.append(vec.cpu().numpy())
    return np.vstack(embs).astype("float32")

def encode_with_sentence_transformers(model_name, texts, bs=64, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name, device=device)
    embs = model.encode(texts, batch_size=bs, show_progress_bar=True, convert_to_numpy=True)
    return embs.astype("float32")

def get_rag_embeddings(model_name, preproc_json, out_dir, bs=64):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = out_dir / "embeddings" / "rag"
    emb_dir.mkdir(parents=True, exist_ok=True)

    name = safe_name(model_name)
    emb_path = emb_dir / f"rag_{name}.npy"
    ids_path = emb_dir / f"rag_{name}_ids.txt"

    if emb_path.exists() and ids_path.exists():
        return np.load(str(emb_path)), [l.strip() for l in open(ids_path, "r", encoding="utf8")]

    records = json.load(open(preproc_json, "r", encoding="utf8"))
    texts = [r["doc"] for r in records]
    ids = [r["id"] for r in records]

    try:
        if is_dpr_model(model_name):
            embs = encode_with_transformers_dpr(model_name, texts, bs=min(bs, 32))
        else:
            embs = encode_with_sentence_transformers(model_name, texts, bs=bs)
    except Exception as e:
        print(f"Primary encode failed for {model_name}: {e}. Falling back to all-mpnet-base-v2.")
        embs = encode_with_sentence_transformers("sentence-transformers/all-mpnet-base-v2", texts, bs=max(8, bs // 2))

    faiss.normalize_L2(embs)
    np.save(emb_path, embs)
    open(ids_path, "w", encoding="utf8").write("\n".join(ids))
    return embs, ids

def build_faiss_index(embs):
    d = embs.shape[1]
    idx = faiss.IndexFlatIP(d)
    idx.add(embs)
    return idx