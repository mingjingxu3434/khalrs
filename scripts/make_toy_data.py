from pathlib import Path
import argparse, random
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/toy")
    ap.add_argument("--users", type=int, default=40)
    ap.add_argument("--items", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    items = list(range(1, a.items+1))
    courses = list(range(a.items+1, a.items+9))
    domains = list(range(a.items+9, a.items+13))
    concepts = list(range(a.items+13, a.items+33))
    pd.DataFrame({"item_id":items}).to_csv(out/"items.csv", index=False)

    triples = []
    for item in items:
        triples += [
            (item,0,rng.choice(concepts)),
            (item,1,rng.choice(courses)),
            (item,2,rng.choice(domains)),
        ]
    pd.DataFrame(triples, columns=["head","relation","tail"]).to_csv(out/"kg.tsv", sep="\t", index=False)
    types = [(x,"item") for x in items] + [(x,"course") for x in courses] + [(x,"domain") for x in domains] + [(x,"concept") for x in concepts]
    pd.DataFrame(types, columns=["entity_id","entity_type"]).to_csv(out/"entity_types.csv", index=False)

    ctx, inter, ts = [], [], 0
    for uid in range(1, a.users+1):
        d = rng.choice(domains)
        cs = rng.sample(courses, 2)
        ctx.append((uid, rng.randint(1,4), ";".join(map(str,cs)), str(d)))
        preferred = sorted(set(
            h for h,r,t in triples
            if (r == 1 and t in cs) or (r == 2 and t == d)
        )) or items
        n = rng.randint(12,24)
        for j in range(n):
            ts += 1
            item = rng.choice(preferred if rng.random() < .8 else items)
            inter.append((uid,item,ts,j//4))
    pd.DataFrame(inter, columns=["user_id","item_id","timestamp","session_id"]).to_csv(out/"interactions.csv", index=False)
    pd.DataFrame(ctx, columns=["user_id","semester_id","courses","domains"]).to_csv(out/"user_context.csv", index=False)
    print(out.resolve())

if __name__ == "__main__":
    main()
