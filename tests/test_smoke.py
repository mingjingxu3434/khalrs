from functools import partial
import subprocess, sys
import torch
from torch.utils.data import DataLoader
from khalrs.config import load_config
from khalrs.data import DataBundle, RecommendationDataset, collate_recommendation
from khalrs.model import KHALRS

def test_forward(tmp_path):
    data=tmp_path/"toy"
    subprocess.check_call([sys.executable,"scripts/make_toy_data.py","--out",str(data),"--users","8","--items","20"])
    cfg=load_config("configs/paper.yaml")
    m=cfg["model"]
    m["max_sessions"]=4; m["max_items_per_session"]=8; m["kg_neighbors"]=4
    b=DataBundle(data,m["max_sessions"],m["max_items_per_session"],m["max_domains"],m["max_courses"],m["kg_neighbors"])
    ds=RecommendationDataset(b,negative_ratio=2)
    fn=partial(collate_recommendation,max_sessions=4,max_items=8,max_domains=m["max_domains"],max_courses=m["max_courses"])
    batch=next(iter(DataLoader(ds,batch_size=4,collate_fn=fn)))
    model=KHALRS(b.num_users,b.num_entities,b.num_relations,b.num_semesters,b.neighbor_ids,b.neighbor_mask,m)
    pos,neg=model(batch)
    assert pos.shape==(4,) and neg.shape==(4,2)
    assert torch.isfinite(pos).all() and torch.isfinite(neg).all()
