from pathlib import Path
import json, itertools
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .kg import KGTripleDataset
from .losses import bpr_loss
from .metrics import ranking_metrics, mean_metrics
from .data import collate_eval

def move_batch(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}

def pretrain_kg(model, triples, num_entities, entity_types, cfg, device, save_path=None):
    ds = KGTripleDataset(triples, num_entities, entity_types)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)
    opt = torch.optim.Adam(model.kg.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    for epoch in range(1, cfg["epochs"]+1):
        model.train()
        total = 0.0
        for pos, neg in loader:
            pos, neg = pos.to(device), neg.to(device)
            loss = model.kg.margin_loss(
                (pos[:,0],pos[:,1],pos[:,2]),
                (neg[:,0],neg[:,1],neg[:,2]),
                cfg["margin"]
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()*len(pos)
        print(f"[KG] epoch={epoch:03d} loss={total/max(len(ds),1):.6f}")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"kg": model.kg.state_dict()}, save_path)

def evaluate_full_ranking(model, bundle, split, cfg, device):
    model.eval()
    rows = []
    all_items = torch.tensor(bundle.item_ids, dtype=torch.long, device=device)
    with torch.no_grad():
        for ex in tqdm(bundle.make_eval_users(split), desc=f"eval:{split}", leave=False):
            batch = collate_eval([ex], bundle.max_sessions, bundle.max_items, bundle.max_domains, bundle.max_courses)
            batch = move_batch(batch, device)
            enc = model.encode_user(batch)
            parts = []
            for i in range(0, len(all_items), cfg.get("score_chunk_size", 2048)):
                cand = all_items[i:i+cfg.get("score_chunk_size", 2048)][None]
                parts.append(model.score_candidates(batch, cand, encoded=enc).squeeze(0))
            scores = torch.cat(parts)
            seen = set(ex["seen"])
            if seen:
                mask = torch.tensor([x in seen for x in bundle.item_ids], dtype=torch.bool, device=device)
                scores[mask] = -torch.inf
            ranked = all_items[torch.argsort(scores, descending=True)].tolist()
            rows.append(ranking_metrics(ranked, ex["relevant"], tuple(cfg["ks"])))
    return mean_metrics(rows)

def train_recommender(model, bundle, train_loader, cfg, eval_cfg, device, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if model.cfg.get("freeze_kg_during_rec", False):
        for p in model.kg.parameters():
            p.requires_grad_(False)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    kg_loader = DataLoader(KGTripleDataset(bundle.triples, bundle.num_entities, bundle.entity_types),
                           batch_size=cfg["batch_size"], shuffle=True)
    kg_iter = itertools.cycle(kg_loader)
    best, bad, hist = -1.0, 0, []

    for epoch in range(1, cfg["epochs"]+1):
        model.train()
        sums = {"loss":0.0,"bpr":0.0,"kg":0.0,"curr":0.0}
        count = 0
        for batch in tqdm(train_loader, desc=f"train:{epoch}", leave=False):
            batch = move_batch(batch, device)
            ps, ns = model(batch)
            rec = bpr_loss(ps, ns)
            _, curr, _ = model.encode_user(batch)
            closs = model.curriculum_loss(batch, curr, cfg["curriculum_margin"])

            kloss = torch.zeros((), device=device)
            if not model.cfg.get("freeze_kg_during_rec", False):
                kp, kn = next(kg_iter)
                kp, kn = kp.to(device), kn.to(device)
                kloss = model.kg.margin_loss(
                    (kp[:,0],kp[:,1],kp[:,2]),
                    (kn[:,0],kn[:,1],kn[:,2]), 1.0
                )

            loss = rec + cfg["lambda_kg"]*kloss + cfg["lambda_curr"]*closs + cfg["lambda_reg"]*model.l2_penalty()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 5.0))
            opt.step()

            bs = len(batch["user_id"])
            count += bs
            for k,v in [("loss",loss),("bpr",rec),("kg",kloss),("curr",closs)]:
                sums[k] += v.item()*bs

        train_stats = {k:v/max(count,1) for k,v in sums.items()}
        val = evaluate_full_ranking(model, bundle, "val", eval_cfg, device)
        hr10 = val.get("HR@10", 0.0)
        hist.append({"epoch":epoch, **train_stats, **{f"val_{k}":v for k,v in val.items()}})
        print(f"[REC] epoch={epoch:03d} loss={train_stats['loss']:.6f} val_HR@10={hr10:.4f}")

        if hr10 > best:
            best, bad = hr10, 0
            torch.save({"model":model.state_dict(),"epoch":epoch,"val":val}, out_dir/"best.pt")
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print("Early stopping.")
                break
        (out_dir/"history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    return hist
