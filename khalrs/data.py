from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
import pandas as pd
import torch
from torch.utils.data import Dataset

def _parse_ids(value):
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [int(x) for x in str(value).split(";") if str(x).strip()]

@dataclass
class UserContext:
    semester_id: int
    courses: list
    domains: list

class DataBundle:
    def __init__(self, root, max_sessions=8, max_items=16, max_domains=8, max_courses=8, kg_neighbors=16):
        root = Path(root)
        self.root = root
        self.max_sessions = max_sessions
        self.max_items = max_items
        self.max_domains = max_domains
        self.max_courses = max_courses
        self.kg_neighbors = kg_neighbors

        self.interactions = pd.read_csv(root / "interactions.csv")
        need = {"user_id", "item_id", "timestamp", "session_id"}
        if need - set(self.interactions.columns):
            raise ValueError(f"interactions.csv missing: {sorted(need - set(self.interactions.columns))}")
        self.interactions = self.interactions.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        self.kg = pd.read_csv(root / "kg.tsv", sep="\t")
        self.context_df = pd.read_csv(root / "user_context.csv")
        self.items_df = pd.read_csv(root / "items.csv")

        self.user_context = {}
        for r in self.context_df.itertuples(index=False):
            self.user_context[int(r.user_id)] = UserContext(
                int(r.semester_id),
                _parse_ids(getattr(r, "courses", ""))[:max_courses],
                _parse_ids(getattr(r, "domains", ""))[:max_domains],
            )
        self.item_ids = sorted(int(x) for x in self.items_df.item_id.unique())
        max_context = 0
        for c in self.user_context.values():
            if c.courses or c.domains:
                max_context = max(max_context, max(c.courses + c.domains))
        max_entity = max(int(self.kg["head"].max()), int(self.kg["tail"].max()), max(self.item_ids), max_context)
        self.num_entities = max_entity + 1
        self.num_relations = int(self.kg["relation"].max()) + 1
        self.num_users = int(self.interactions["user_id"].max()) + 1
        self.num_semesters = max(c.semester_id for c in self.user_context.values()) + 1

        et = root / "entity_types.csv"
        self.entity_types = {}
        if et.exists():
            x = pd.read_csv(et)
            self.entity_types = {int(r.entity_id): str(r.entity_type) for r in x.itertuples(index=False)}

        self.triples = list(self.kg[["head", "relation", "tail"]].itertuples(index=False, name=None))
        self.neighbor_ids, self.neighbor_mask = self._build_neighbor_table()
        self.user_interacted = defaultdict(set)
        for r in self.interactions.itertuples(index=False):
            self.user_interacted[int(r.user_id)].add(int(r.item_id))
        self.splits = self._chronological_split()

    def _chronological_split(self):
        out = {"train": [], "val": [], "test": []}
        for uid, g in self.interactions.groupby("user_id", sort=False):
            rows = list(g.itertuples(index=False))
            if len(rows) < 5:
                continue
            n = len(rows)
            a = max(1, int(n * 0.70))
            b = min(max(a + 1, int(n * 0.80)), n - 1)
            out["train"].extend(rows[:a])
            out["val"].extend(rows[a:b])
            out["test"].extend(rows[b:])
        return out

    def _build_neighbor_table(self):
        adj = defaultdict(list)
        for h, r, t in self.triples:
            adj[int(h)].append(int(t))
            adj[int(t)].append(int(h))
        width = self.kg_neighbors + 1
        ids = torch.zeros(self.num_entities, width, dtype=torch.long)
        mask = torch.zeros(self.num_entities, width, dtype=torch.bool)
        for e in range(1, self.num_entities):
            vals = [e] + adj.get(e, [])[:self.kg_neighbors]
            ids[e, :len(vals)] = torch.tensor(vals)
            mask[e, :len(vals)] = True
        return ids, mask

    def get_context(self, uid):
        return self.user_context.get(uid, UserContext(0, [], []))

    def _history_sessions(self, rows):
        sessions, cur, sid = [], [], None
        for r in rows:
            rsid = int(r.session_id)
            if sid is None or rsid == sid:
                cur.append(int(r.item_id))
                sid = rsid
            else:
                sessions.append(cur[-self.max_items:])
                cur = [int(r.item_id)]
                sid = rsid
        if cur:
            sessions.append(cur[-self.max_items:])
        return sessions[-self.max_sessions:]

    def make_train_samples(self):
        by = defaultdict(list)
        for r in self.splits["train"]:
            by[int(r.user_id)].append(r)
        samples = []
        for uid, rows in by.items():
            for i in range(1, len(rows)):
                samples.append({
                    "user_id": uid,
                    "positive": int(rows[i].item_id),
                    "history": self._history_sessions(rows[:i]),
                    "context": self.get_context(uid),
                })
        return samples

    def make_eval_users(self, split="val"):
        buckets = {k: defaultdict(list) for k in ("train", "val", "test")}
        for name in buckets:
            for r in self.splits[name]:
                buckets[name][int(r.user_id)].append(r)
        target = buckets[split]
        out = []
        for uid, targets in target.items():
            history = list(buckets["train"][uid])
            if split == "test":
                history += list(buckets["val"][uid])
            if not history or not targets:
                continue
            out.append({
                "user_id": uid,
                "history": self._history_sessions(history),
                "context": self.get_context(uid),
                "relevant": sorted(set(int(r.item_id) for r in targets)),
                "seen": sorted(set(int(r.item_id) for r in history)),
            })
        return out

class RecommendationDataset(Dataset):
    def __init__(self, bundle, negative_ratio=4, seed=42):
        self.bundle = bundle
        self.samples = bundle.make_train_samples()
        self.negative_ratio = negative_ratio
        self.rng = random.Random(seed)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        neg = []
        while len(neg) < self.negative_ratio:
            x = self.rng.choice(self.bundle.item_ids)
            if x not in self.bundle.user_interacted[s["user_id"]]:
                neg.append(x)
        return {**s, "negatives": neg}

def _tensorize(samples, max_sessions, max_items, max_domains, max_courses):
    b = len(samples)
    history = torch.zeros(b, max_sessions, max_items, dtype=torch.long)
    item_mask = torch.zeros(b, max_sessions, max_items, dtype=torch.bool)
    session_mask = torch.zeros(b, max_sessions, dtype=torch.bool)
    domains = torch.zeros(b, max_domains, dtype=torch.long)
    domain_mask = torch.zeros(b, max_domains, dtype=torch.bool)
    courses = torch.zeros(b, max_courses, dtype=torch.long)
    course_mask = torch.zeros(b, max_courses, dtype=torch.bool)
    for i, s in enumerate(samples):
        for si, seq in enumerate(s["history"][-max_sessions:]):
            seq = seq[-max_items:]
            if seq:
                history[i, si, :len(seq)] = torch.tensor(seq)
                item_mask[i, si, :len(seq)] = True
                session_mask[i, si] = True
        ds = s["context"].domains[:max_domains]
        cs = s["context"].courses[:max_courses]
        if ds:
            domains[i, :len(ds)] = torch.tensor(ds)
            domain_mask[i, :len(ds)] = True
        if cs:
            courses[i, :len(cs)] = torch.tensor(cs)
            course_mask[i, :len(cs)] = True
    return {
        "history_items": history,
        "item_mask": item_mask,
        "session_mask": session_mask,
        "domains": domains,
        "domain_mask": domain_mask,
        "courses": courses,
        "course_mask": course_mask,
        "semester_id": torch.tensor([s["context"].semester_id for s in samples], dtype=torch.long),
        "user_id": torch.tensor([s["user_id"] for s in samples], dtype=torch.long),
    }

def collate_recommendation(samples, max_sessions=8, max_items=16, max_domains=8, max_courses=8):
    out = _tensorize(samples, max_sessions, max_items, max_domains, max_courses)
    out["positive"] = torch.tensor([s["positive"] for s in samples], dtype=torch.long)
    out["negatives"] = torch.tensor([s["negatives"] for s in samples], dtype=torch.long)
    return out

def collate_eval(samples, max_sessions=8, max_items=16, max_domains=8, max_courses=8):
    out = _tensorize(samples, max_sessions, max_items, max_domains, max_courses)
    out["relevant"] = [s["relevant"] for s in samples]
    out["seen"] = [s["seen"] for s in samples]
    return out
