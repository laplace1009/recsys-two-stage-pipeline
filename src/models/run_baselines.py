from __future__ import annotations


from pathlib import Path

import pandas as pd
import yaml

from .baseline_popular import GlobalPopular, RecentPopular
from ..eval.evaluate import run_evaluation, save_results


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"


def load_config() -> dict:
    """configs/base.yaml 로드."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    전처리된 데이터를 로드한다.

    Returns
    -------
    transactions : pd.DataFrame
        transactions_clean.parquet (split 컬럼 포함)
    events : pd.DataFrame
        events.parquet (split, items 컬럼 포함)
    """
    transactions = pd.read_parquet(DATA_DIR / "transactions_clean.parquet")
    events = pd.read_parquet(DATA_DIR / "events.parquet")
    return transactions, events


def print_results_table(all_results: dict[str, dict[str, float]]) -> None:
    """결과 테이블을 보기 좋게 출력."""
    print("\n" + "=" * 70)
    print("  베이스라인 평가 결과 요약")
    print("=" * 70)

    # 지표 이름 수집
    metric_names = [
        key for key in list(all_results.values())[0]
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
    n_users = list(all_results.values())[0].get("n_eval_users", 0)
    print(f"  평가 유저 수: {int(n_users):,}")
    print("=" * 70)


def main() -> None:
    print("=" * 60)
    print("  베이스라인 모델 평가 시작")
    print("=" * 60)

    # 설정 로드
    config = load_config()
    k_list = config.get("evaluation", {}).get("k_list", [10, 20])

    # 데이터 로드
    transactions, events = load_data()
    train_tx = transactions[transactions["split"] == "train"]

    print(f"\n[데이터 로드 완료]")
    print(f"  전체 트랜잭션: {len(transactions):,}")
    print(f"  Train 트랜잭션: {len(train_tx):,}")
    print(f"  이벤트 수: {len(events):,}")

    all_results: dict[str, dict[str, float]] = {}

    # ── 모델 1: Global Popular ──
    print("\n" + "-" * 40)
    print("  모델: GlobalPopular")
    print("-" * 40)
    model_gp = GlobalPopular(user_based=False)
    model_gp.fit(train_tx)

    results_gp = run_evaluation(
        model=model_gp,
        events=events,
        k_list=k_list,
        split="test",
    )
    all_results["GlobalPopular"] = results_gp

    # ── 모델 2: Global Popular (user-based) ──
    print("\n" + "-" * 40)
    print("  모델: GlobalPopular (user-based)")
    print("-" * 40)
    model_gp_ub = GlobalPopular(user_based=True)
    model_gp_ub.fit(train_tx)

    results_gp_ub = run_evaluation(
        model=model_gp_ub,
        events=events,
        k_list=k_list,
        split="test",
    )
    all_results["GlobalPopular_UB"] = results_gp_ub

    # ── 모델 3: Recent Popular (4주) ──
    print("\n" + "-" * 40)
    print("  모델: RecentPopular (4weeks)")
    print("-" * 40)
    model_rp4 = RecentPopular(recent_weeks=4, user_based=False)
    model_rp4.fit(train_tx)

    results_rp4 = run_evaluation(
        model=model_rp4,
        events=events,
        k_list=k_list,
        split="test",
    )
    all_results["RecentPopular_4w"] = results_rp4

    # ── 모델 4: Recent Popular (8주) ──
    print("\n" + "-" * 40)
    print("  모델: RecentPopular (8weeks)")
    print("-" * 40)
    model_rp8 = RecentPopular(recent_weeks=8, user_based=False)
    model_rp8.fit(train_tx)

    results_rp8 = run_evaluation(
        model=model_rp8,
        events=events,
        k_list=k_list,
        split="test",
    )
    all_results["RecentPopular_8w"] = results_rp8

    # 결과 출력 + 저장
    print_results_table(all_results)

    results_path = REPORTS_DIR / "baseline_results.json"
    for model_name, results in all_results.items():
        save_results(results, model_name, results_path)


if __name__ == "__main__":
    main()
