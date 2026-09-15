from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, json, torch
from khalrs.config import load_config, resolve_device
from khalrs.data import DataBundle
from khalrs.model import KHALRS
from khalrs.trainer import evaluate_full_ranking

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/paper.yaml")
    ap.add_argument("--data",default="data/toy")
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--split",choices=["val","test"],default="test")
    a=ap.parse_args()
    cfg=load_config(a.config); device=resolve_device(cfg["device"]); m=cfg["model"]
    b=DataBundle(a.data,m["max_sessions"],m["max_items_per_session"],m["max_domains"],m["max_courses"],m["kg_neighbors"])
    model=KHALRS(b.num_users,b.num_entities,b.num_relations,b.num_semesters,b.neighbor_ids,b.neighbor_mask,m).to(device)
    state=torch.load(a.checkpoint,map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    print(json.dumps(evaluate_full_ranking(model,b,a.split,cfg["evaluation"],device),indent=2))

if __name__ == "__main__":
    main()
