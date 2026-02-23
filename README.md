# recsys-two-stage-pipeline

Purchase-transaction recommendation portfolio: candidate generation (ALS) -> ranking (LTR) -> diversity/serendipity reranking -> A/B test plan.

## Day 2 scope

- Add reproducible dataset download script for UCI Online Retail.
- Add dataset source and license note (CC BY 4.0).

## Download data

```bash
python3 src/data/download.py
```

The script downloads the dataset into `data/raw/` and keeps raw files out of Git tracking.
