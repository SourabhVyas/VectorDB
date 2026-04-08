import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import faiss
from typing import Tuple, List, Dict, Any

def id_to_numeric_map(preproc_json: str) -> Dict[str, Any]:
    recs = json.load(open(preproc_json, "r", encoding="utf8"))
    return {r.get("id"): r.get("numeric_label") for r in recs}

def eval_embedding_clf(emb_np: np.ndarray, ids: List[str], preproc_json: str, test_size=0.2, random_state=42, normalize=False) -> Tuple[Dict, LogisticRegression]:
    """
    Train logistic regression only on binary-labelled examples (numeric_label 0/1).
    emb_np rows align with ids list.
    """
    id_map = id_to_numeric_map(preproc_json)

    X_all, y_all = [], []
    for i, rid in enumerate(ids):
        lbl = id_map.get(rid, None)
        if lbl in (0, 1):
            X_all.append(emb_np[i])
            y_all.append(int(lbl))

    if len(y_all) == 0:
        raise ValueError("No binary-labelled records found for training/eval.")

    X = np.vstack(X_all)
    y = np.array(y_all)
    if normalize:
        faiss.normalize_L2(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    clf = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    p = clf.predict_proba(X_test)[:, 1]
    out = {
        "acc": float(accuracy_score(y_test, p >= 0.5)),
        "f1": float(f1_score(y_test, p >= 0.5)),
        "roc_auc": float(roc_auc_score(y_test, p)),
        "pr_auc": float(average_precision_score(y_test, p)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test))
    }
    return out, clf

def rag_self_retrieval_eval(embs: np.ndarray, ids: List[str], topk=50) -> Dict:
    embs = embs.astype("float32").copy()
    faiss.normalize_L2(embs)
    d = embs.shape[1]
    idx = faiss.IndexFlatIP(d)
    idx.add(embs)
    max_k = min(topk, embs.shape[0])
    D, I = idx.search(embs, max_k)
    N = embs.shape[0]
    rr_total = 0.0
    recall_counts = {1: 0, 5: 0, 10: 0}
    ks = [k for k in (1, 5, 10) if k <= max_k]
    for i in range(N):
        retrieved_idxs = I[i]
        retrieved_ids = [ids[j] for j in retrieved_idxs]
        if ids[i] in retrieved_ids:
            pos = retrieved_ids.index(ids[i]) + 1
            rr_total += 1.0 / pos
        for k in ks:
            if ids[i] in retrieved_ids[:k]:
                recall_counts[k] += 1
    out = {"MRR": rr_total / N}
    for k in ks:
        out[f"Recall@{k}"] = recall_counts[k] / N
    return out

def simulate_deferral(probs: np.ndarray, true_labels: np.ndarray, t_low: float, t_high: float):
    assert 0.0 <= t_low < t_high <= 1.0
    n = len(probs)
    decisions = []
    deferred_idx = []
    correct_non_deferred = 0
    non_deferred_count = 0
    for i, p in enumerate(probs):
        if p <= t_low:
            pred = 0
            non_deferred_count += 1
            if pred == true_labels[i]:
                correct_non_deferred += 1
        elif p >= t_high:
            pred = 1
            non_deferred_count += 1
            if pred == true_labels[i]:
                correct_non_deferred += 1
        else:
            deferred_idx.append(i)
            pred = None
        decisions.append(pred)
    deferral_rate = len(deferred_idx) / n
    acc_non_deferred = (correct_non_deferred / non_deferred_count) if non_deferred_count > 0 else None
    return {
        "deferral_rate": deferral_rate,
        "n_deferred": len(deferred_idx),
        "acc_non_deferred": acc_non_deferred,
        "indices_deferred": deferred_idx,
        "decisions": decisions
    }