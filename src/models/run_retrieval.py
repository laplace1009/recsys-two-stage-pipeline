#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.features.build_interaction_matrix import (
    build_interaction_matrix,
    print_matrix_stats
)
from src.models.retrieval_als import RetrievalALS
from src.eval.evaluate import run_evaluation, save_results

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATADIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"

def load_config() -> dict:
    """configs/base.yaml 로드."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)
    

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    전처리된 데이터를 로드한다.

    Returns
    -------
    transactions : pd.DataFrame
        transactions_clean.parquet (split 컬럼 포함)
    events : pd.DataFrame
        events.parquet (split, items 컬럼 포함)
    interactions : pd.DataFrame
        interactions.parquet (user_id, item_id, value 컬럼)
    """
    transactions = pd.read_parquet(DATADIR / "transactions_clean.parquet")
    events = pd.read_parquet(DATADIR / "events.parquet")
    interactions = pd.read_parquet(DATADIR / "interactions.parquet")
    return transactions, events, interactions


def load_baseline_results() -> dict[str, dict[str, float]] | None:
    """기존 베이스라인 결과를 로드 (비교 출력용)."""
    path = REPORTS_DIR / "baseline_results.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None

def print_comparison_table(
    all_results: dict[str, dict[str, float]],
) -> None:
    """베이스라인 + ALS 성능 비교 테이블 출력."""
    print("\n" + "=" * 80)
    print("  성능 비교 테이블 (Baselines vs ALS)")
    print("=" * 80)

    # 지표 이름 수집
    first_result = list(all_results.values())[0]
    metric_names = [
        key for key in first_result
        if key != "n_eval_users"
    ]

    # 헤더
    header = f"{'Model':<25}"
    for m in metric_names:
        header += f" {m:>12}"
    print(header)
    print("-" * len(header))

    # 각 모델 결과
    for model_name, results in all_results.items():
        row = f"{model_name:<25}"
        for m in metric_names:
            row += f" {results.get(m, 0.0):>12.4f}"
        print(row)

    print("-" * len(header))
    n_users = first_result.get("n_eval_users", 0)
    print(f"  평가 유저 수: {int(n_users):,}")
    print("=" * 80)


def main() -> None:
    print("=" * 60)
    print("  ALS 후보생성 모델 — 학습 및 평가")
    print("=" * 60)

    # ── 1. 설정 로드 ──
    config = load_config()
    k_list = config.get("evaluation", {}).get("k_list", [10, 20])
    als_config = config.get("als", {})

    # ── 2. 데이터 로드 ──
    transactions, events, interactions = load_data()
    train_tx = transactions[transactions["split"] == "train"]

    print(f"\n[데이터 로드 완료]")
    print(f"  전체 트랜잭션: {len(transactions):,}")
    print(f"  Train 트랜잭션: {len(train_tx):,}")
    print(f"  이벤트 수: {len(events):,}")
    print(f"  상호작용 수: {len(interactions):,}")

    # ── 3. Interaction Matrix 빌드 (Day 8) ──
    print("\n" + "-" * 40)
    print("  Step 1: Interaction Matrix 빌드")
    print("-" * 40)

    im = build_interaction_matrix(interactions)
    print_matrix_stats(im)

    # ── 4. ALS 모델 학습 (Day 9~10) ──
    print("\n" + "-" * 40)
    print("  Step 2: ALS 모델 학습")
    print("-" * 40)

    model = RetrievalALS(
        factors=als_config.get("factors", 64),
        iterations=als_config.get("iterations", 15),
        regularization=als_config.get("regularization", 0.01),
        random_state=als_config.get("random_state", 42),
    )
    model.fit(im)

    # ── 5. 오프라인 평가 ──
    print("\n" + "-" * 40)
    print("  Step 3: 오프라인 평가")
    print("-" * 40)

    results_als = run_evaluation(
        model=model,
        events=events,
        k_list=k_list,
        split="test",
    )

    # ── 6. 결과 저장 ──
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    als_results = {"RetrievalALS": results_als}
    results_path = REPORTS_DIR / "retrieval_results.json"
    with open(results_path, "w") as f:
        json.dump(als_results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ ALS 결과 저장: {results_path}")

    # ── 7. 베이스라인과 비교 ──
    baseline_results = load_baseline_results()
    comparison = {}

    if baseline_results:
        comparison.update(baseline_results)

    comparison["RetrievalALS"] = results_als
    print_comparison_table(comparison)

    # 비교 결과도 저장
    comparison_path = REPORTS_DIR / "comparison_baselines_vs_als.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 비교 결과 저장: {comparison_path}")


if __name__ == "__main__":
    main()