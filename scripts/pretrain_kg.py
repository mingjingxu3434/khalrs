from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from khalrs.config import load_config, seed_everything, resolve_device
from khalrs.data import DataBundle
from khalrs.model import KHALRS
from khalrs.trainer import pretrain_kg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paper.yaml")
    ap.add_argument("--data", default="data/toy")
    ap.add_argument("--out", default="outputs/kg_pretrained.pt")
    a = ap.parse_args()
    cfg = load_config(a.config); seed_everything(cfg["seed"]); device = resolve_device(cfg["device"])
    m = cfg["model"]
    b = DataBundle(a.data,m["max_sessions"],m["max_items_per_session"],m["max_domains"],m["max_courses"],m["kg_neighbors"])
    model = KHALRS(b.num_users,b.num_entities,b.num_relations,b.num_semesters,b.neighbor_ids,b.neighbor_mask,m).to(device)
    pretrain_kg(model,b.triples,b.num_entities,b.entity_types,cfg["kg_pretrain"],device,a.out)

if __name__ == "__main__":
    main()
