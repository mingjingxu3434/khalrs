from collections import defaultdict
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class RelationProjectionKG(nn.Module):
    def __init__(self, num_entities, num_relations, d, d_r, padding_idx=0):
        super().__init__()
        self.entity = nn.Embedding(num_entities, d, padding_idx=padding_idx)
        self.relation = nn.Embedding(num_relations, d_r)
        self.projection = nn.Parameter(torch.empty(num_relations, d_r, d))
        nn.init.xavier_uniform_(self.entity.weight)
        with torch.no_grad():
            self.entity.weight[padding_idx].zero_()
        nn.init.xavier_uniform_(self.relation.weight)
        nn.init.xavier_uniform_(self.projection)

    def score(self, head, rel, tail):
        eh = self.entity(head)
        et = self.entity(tail)
        r = self.relation(rel)
        wr = self.projection[rel]
        ph = torch.bmm(wr, eh.unsqueeze(-1)).squeeze(-1)
        pt = torch.bmm(wr, et.unsqueeze(-1)).squeeze(-1)
        return ((ph + r - pt) ** 2).sum(dim=-1)

    def margin_loss(self, pos, neg, margin=1.0):
        ph, pr, pt = pos
        nh, nr, nt = neg
        return F.relu(margin + self.score(ph, pr, pt) - self.score(nh, nr, nt)).mean()

class KGTripleDataset(Dataset):
    def __init__(self, triples, num_entities, entity_types=None, seed=42):
        self.triples = [tuple(map(int, x)) for x in triples]
        self.num_entities = num_entities
        self.rng = random.Random(seed)
        self.entity_types = entity_types or {}
        self.type_to_entities = defaultdict(list)
        for eid in range(1, num_entities):
            self.type_to_entities[self.entity_types.get(eid, None)].append(eid)

    def __len__(self):
        return len(self.triples)

    def _same_type(self, eid):
        typ = self.entity_types.get(eid, None)
        pool = self.type_to_entities.get(typ) or list(range(1, self.num_entities))
        return self.rng.choice(pool)

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        if self.rng.random() < 0.5:
            nh, nt = self._same_type(h), t
        else:
            nh, nt = h, self._same_type(t)
        return (
            torch.tensor([h, r, t], dtype=torch.long),
            torch.tensor([nh, r, nt], dtype=torch.long),
        )
