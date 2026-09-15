# KHALRS

## Implemented components

- relation-specific projected KG embedding and margin loss;
- three-tier HAE: item -> session -> curriculum-domain;
- user-conditioned session attention;
- candidate-conditioned multi-head cross-attention over bounded 1-hop KG neighborhoods;
- two-layer KG/user fusion gate;
- semester + enrolled-course curriculum encoder;
- `5d -> 2d -> d -> 1` adaptive decoder;
- BPR + KG + curriculum consistency + explicit L2 objective;
- chronological 70/10/20 split;
- HR@K, NDCG@K, Precision@K, Recall@K, MRR full-ranking evaluation.

## Paper defaults

`configs/paper.yaml` uses:
- d=128
- 4 attention heads
- curriculum/semester embedding=64
- dropout=0.3
- Adam lr=1e-3
- weight decay=1e-5
- batch size=2048
- negative ratio=1:4
- early stopping patience=20

## Manuscript ambiguities handled explicitly

1. The Method equations use GELU in the decoder, while Experimental Setup says ReLU.
   Default is GELU. Switch `model.decoder_activation: relu` to test the other interpretation.
2. The Method says KG parameters are fine-tuned jointly, while Limitations says they are frozen.
   Default follows Method (`freeze_kg_during_rec: false`); the flag can reproduce the frozen variant.
3. The setup lists `lambda_1=0.1`, `lambda_2=0.01`, `lambda_3=1e-4` without naming them.
   This implementation maps them to KG, curriculum, and explicit L2 regularization.
4. The processed LibRS-HVS / MIND-Edu / DouBan-Acad files and exact preprocessing pipeline are not
   included in the manuscript, so exact table-number reproduction cannot be guaranteed from the paper text alone.
5. The HAE equations are additive-attention equations, while Experimental Setup mentions 4 attention heads per HAE level. This implementation follows the equations for HAE and uses 4 heads in the explicitly multi-head CKF block.

## Install

```bash
pip install -r requirements.txt
```

## Run toy example

```bash
python scripts/make_toy_data.py --out data/toy
python scripts/pretrain_kg.py --data data/toy --out outputs/kg_pretrained.pt
python scripts/train.py --data data/toy --kg-checkpoint outputs/kg_pretrained.pt --out outputs/toy_run
python scripts/evaluate.py --data data/toy --checkpoint outputs/toy_run/best.pt --split test
```

## Test

```bash
pytest -q
```

## Project tree

```text
khalrs_reproduction/
├── configs/paper.yaml
├── data/README.md
├── khalrs/
│   ├── config.py
│   ├── data.py
│   ├── kg.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── modules.py
│   └── trainer.py
├── scripts/
│   ├── make_toy_data.py
│   ├── pretrain_kg.py
│   ├── train.py
│   └── evaluate.py
├── tests/test_smoke.py
├── requirements.txt
└── README.md
```
