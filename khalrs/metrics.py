import math

def ranking_metrics(ranked_items, relevant_items, ks=(5, 10, 20)):
    rel = set(int(x) for x in relevant_items)
    if not rel:
        return {}
    out = {}
    first_rank = next((i for i, x in enumerate(ranked_items, 1) if int(x) in rel), None)
    out["MRR"] = 0.0 if first_rank is None else 1.0 / first_rank
    for k in ks:
        top = [int(x) for x in ranked_items[:k]]
        hits = [1 if x in rel else 0 for x in top]
        n = sum(hits)
        out[f"HR@{k}"] = 1.0 if n else 0.0
        out[f"Precision@{k}"] = n / float(k)
        out[f"Recall@{k}"] = n / float(len(rel))
        dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
        ideal = min(len(rel), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
        out[f"NDCG@{k}"] = dcg / idcg if idcg else 0.0
    return out

def mean_metrics(rows):
    if not rows:
        return {}
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
