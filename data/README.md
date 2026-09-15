# Dataset format

`0` is reserved for padding. All real entity IDs start at 1.

- `interactions.csv`: `user_id,item_id,timestamp,session_id`
- `items.csv`: `item_id`
- `kg.tsv`: tab-separated `head,relation,tail`
- `user_context.csv`: `user_id,semester_id,courses,domains`
  - `courses` and `domains` are semicolon-separated global KG entity IDs.
- `entity_types.csv` (optional): `entity_id,entity_type`

Items, courses, domains, concepts, authors, etc. must use one shared global KG entity ID space.
The loader performs a chronological 70/10/20 split per user and removes users with fewer than 5 interactions.

Run:
```bash
python scripts/make_toy_data.py --out data/toy
```
to generate a valid example.
