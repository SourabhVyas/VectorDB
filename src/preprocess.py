import json, re
from pathlib import Path
import nltk
nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Any, Tuple

URL_RE = re.compile(r'https?://\S+|\bwww\.\S+')
SMART_QUOTES = {'\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'"}
MAX_J_SENTS_DEFAULT = 10

def normalize_text(s: str) -> str:
    if not s:
        return ""
    for k, v in SMART_QUOTES.items():
        s = s.replace(k, v)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def clean_just_list(just_list, max_j_sents=MAX_J_SENTS_DEFAULT):
    non_url = []
    for t in just_list:
        if not t:
            continue
        t = URL_RE.sub('', t)
        t = normalize_text(t)
        if t:
            non_url.append(t)
    if not non_url:
        return ""
    txt = " ".join(non_url)
    sents = sent_tokenize(txt)
    return " ".join(sents[:max_j_sents])

def _extract_label_from_meta(meta: Dict[str, Any]) -> Tuple[Any, Any]:
    if not meta or not isinstance(meta, dict):
        return None, None
    # check common keys
    for k in ("label", "Label", "verdict", "truth", "truth_label", "meta_label"):
        if k in meta and meta[k] is not None:
            raw = str(meta[k]).strip()
            if raw == "True":
                return "True", 1
            if raw == "False":
                return "False", 0
            if raw == "Not sure":
                return "Not sure", 2
            return raw, None
    return None, None

def preprocess(in_path, out_path, max_j_sents=MAX_J_SENTS_DEFAULT) -> List[Dict]:
    in_path, out_path = Path(in_path), Path(out_path)
    if out_path.exists():
        return json.load(open(out_path, "r", encoding="utf8"))
    data = json.load(open(in_path, "r", encoding="utf8"))
    out = []
    for i, rec in enumerate(data):
        just_list = rec.get("justification") or rec.get("justifications") or []
        if not just_list:
            continue
        just_text = clean_just_list(just_list, max_j_sents=max_j_sents)
        if not just_text:
            continue
        stmt = normalize_text(rec.get("statement", "") or "")
        doc = stmt + " ||| " + just_text if just_text else stmt

        # robust id
        rid = rec.get("id") or rec.get("uuid") or f"rec_{i}"

        # robust meta: prefer 'meta' then 'metadata', ensure dict
        meta = rec.get("meta", None)
        if meta is None:
            meta = rec.get("metadata", {}) or {}
        if not isinstance(meta, dict):
            meta = {}

        label_str, numeric_label = _extract_label_from_meta(meta)

        out.append({
            "id": rid,
            "statement": stmt,
            "justification": just_text,
            "doc": doc,
            "meta": meta,
            "raw_label": meta.get("label") if isinstance(meta, dict) else None,
            "label": label_str,
            "numeric_label": numeric_label
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w", encoding="utf8"), indent=2)
    return out