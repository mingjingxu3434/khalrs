from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from functools import partial
from torch.utils.data import DataLoader
from khalrs.config import load_config, seed_everything, resolve_device
from khalrs.data import DataBundle, RecommendationDataset, collate_recommendation
from khalrs.model import KHALRS
from khalrs.trainer import train_recommender

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paper.yaml")
    ap.add_argument("--data", default="data/toy")
    ap.add_argument("--kg-checkpoint", default=None)
    ap.add_argument("--out", default="outputs/run1")
    a = ap.parse_args()
    cfg=load_config(a.config); seed_everything(cfg["seed"]); device=resolve_device(cfg["device"]); m=cfg["model"]
    b=DataBundle(a.data,m["max_sessions"],m["max_items_per_session"],m["max_domains"],m["max_courses"],m["kg_neighbors"])
    model=KHALRS(b.num_users,b.num_entities,b.num_relations,b.num_semesters,b.neighbor_ids,b.neighbor_mask,m).to(device)
    if a.kg_checkpoint:
        model.load_kg_state(a.kg_checkpoint,m.get("freeze_kg_during_rec",False))
    ds=RecommendationDataset(b,cfg["training"]["negative_ratio"],cfg["seed"])
    fn=partial(collate_recommendation,max_sessions=m["max_sessions"],max_items=m["max_items_per_session"],max_domains=m["max_domains"],max_courses=m["max_courses"])
    loader=DataLoader(ds,batch_size=cfg["training"]["batch_size"],shuffle=True,num_workers=cfg["training"]["num_workers"],collate_fn=fn)
    train_recommender(model,b,loader,cfg["training"],cfg["evaluation"],device,a.out)

if __name__ == "__main__":
    main()
