from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import pandas as pd

from src.eval.metrics import evaluate_all


class Recommender(Protocol):
    """모든 추천 모델이 따라야 할 인터페이스."""

    def recommend(self, user_id: str, k: int) -> list[str]:
        """유저에게 Top-K 아이템을 추천한다."""
        ...


# ──────────────────────────────────────────────
# Ground Truth 구성
# ──────────────────────────────────────────────

def build_ground_truth(
    events: pd.DataFrame,
    split: str = "test",
) -> dict[str, set[str]]:
    """
    events 테이블에서 특정 split의 유저별 정답 아이템 집합 구성.

    Parameters
    ----------
    events : pd.DataFrame
        컬럼: user_id, items (list[str]), split
    split : str
        "test" 또는 "valid".

    Returns
    -------
    dict[str, set[str]]
        유저 ID → 정답 아이템 ID 집합.
    """
    split_events = events[events["split"] == split]

    ground_truth: dict[str, set[str]] = {}
    for _, row in split_events.iterrows():
        uid = row["user_id"]
        items = row["items"] if isinstance(row["items"], list) else []
        if uid not in ground_truth:
            ground_truth[uid] = set()
        ground_truth[uid].update(items)

    return ground_truth


def get_train_users(events: pd.DataFrame) -> set[str]:
    """train split에 이력이 있는 유저 집합."""
    return set(events[events["split"] == "train"]["user_id"].unique())


def get_train_items(events: pd.DataFrame) -> set[str]:
    """train split에 등장한 아이템 집합 (추천 후보 풀)."""
    train_events = events[events["split"] == "train"]
    items: set[str] = set()
    for item_list in train_events["items"]:
        if isinstance(item_list, list):
            items.update(item_list)
    return items


# ──────────────────────────────────────────────
# 평가 실행
# ──────────────────────────────────────────────

def run_evaluation(
    model: Recommender,
    events: pd.DataFrame,
    k_list: list[int] | None = None,
    split: str = "test",
    verbose: bool = True,
) -> dict[str, float]:
    """
    모델을 평가하고 결과를 반환한다.

    Parameters
    ----------
    model : Recommender
        recommend(user_id, k) 메서드를 가진 모델.
    events : pd.DataFrame
        전처리된 events 테이블 (split 컬럼 포함).
    k_list : list[int], optional
        평가할 K 값 리스트.
    split : str
        평가 대상 split ("test" 또는 "valid").
    verbose : bool
        진행상황 출력 여부.

    Returns
    -------
    dict[str, float]
        지표명 → 값.
    """
    if k_list is None:
        k_list = [10, 20]

    max_k = max(k_list)

    # 1) Ground truth 구성
    ground_truth = build_ground_truth(events, split=split)

    # 2) train에 이력이 있는 유저만 평가 (콜드스타트 유저 제외)
    train_users = get_train_users(events)
    eval_users = [uid for uid in ground_truth if uid in train_users]

    if verbose:
        print(f"[Evaluation] split={split}")
        print(f"  Ground truth 유저: {len(ground_truth):,}")
        print(f"  Train 이력 유저:   {len(train_users):,}")
        print(f"  평가 대상 유저:    {len(eval_users):,}")

    # 3) 추천 생성
    user_recommendations: dict[str, list[str]] = {}
    for i, uid in enumerate(eval_users):
        rec = model.recommend(uid, max_k)
        user_recommendations[uid] = rec
        if verbose and (i + 1) % 500 == 0:
            print(f"  추천 생성 중... {i + 1}/{len(eval_users)}")

    # 4) 지표 계산
    results = evaluate_all(
        user_recommendations=user_recommendations,
        user_ground_truth={uid: ground_truth[uid] for uid in eval_users},
        k_list=k_list,
    )

    if verbose:
        print(f"\n  === 평가 결과 ({split}) ===")
        for metric, value in results.items():
            if metric == "n_eval_users":
                print(f"  {metric}: {int(value)}")
            else:
                print(f"  {metric}: {value:.4f}")

    return results


def save_results(
    results: dict[str, float],
    model_name: str,
    output_path: Path,
) -> None:
    """평가 결과를 JSON 파일로 저장/추가한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 기존 결과가 있으면 로드
    all_results: dict = {}
    if output_path.exists():
        with open(output_path, "r") as f:
            all_results = json.load(f)

    all_results[model_name] = results

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"  ✓ 결과 저장: {output_path}")
