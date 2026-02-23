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
구매 트랜잭션 기반 개인화 추천 피드: 후보생성(ALS) -> 랭킹(LTR) -> 다양성/우연성 리랭킹(텍스트 임베딩) -> A/B 테스트 설계 문서까지.

## Problem

- Given a user's purchase history up to time `t`, recommend Top-K items for the next purchase event.
- Offline evaluation uses time-based split to avoid leakage.

## Why this matters (Daangn fit)

- Feed engagement 개선과 함께 다양하고 우연한 발견(serendipity)을 유도할 수 있습니다.
- 데이터 기반 가설 수립 후 온라인 실험(A/B)으로 검증 가능한 형태로 설계합니다.
- 텍스트 임베딩(LLM 활용 가능)을 결합해 개인화 품질을 개선합니다.

## Project status

- Day 1 complete: project bootstrap, environment pinning, README problem draft.

## Planned structure

```text
configs/
data/
  raw/
  processed/
notebooks/
reports/
src/
  data/
  features/
  models/
  eval/
  serve/
tests/
```
