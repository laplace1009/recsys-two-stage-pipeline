from __future__ import annotations

import numpy as np
from typing import Sequence

def recall_at_k(
    recommended: Sequence[str],
    relevant: set[str],
    k: int,
) -> float:
    """
    Recall@K: 정답 아이템 중 Top-K 추천에 포함된 비율.

    Parameters
    ----------
    recommended : list[str]
        추천된 아이템 ID 리스트 (순서대로, 최대 K개 사용).
    relevant : set[str]
        정답(실제 구매) 아이템 ID 집합.
    k : int
        상위 K개까지만 평가.

    Returns
    -------
    float
        Recall 값 (0.0 ~ 1.0). 정답이 없으면 0.0.
    """
    if not relevant:
        return 0.0
    top_k = set(recommended[:k])
    hits = top_k & relevant
    return len(hits) / len(relevant)

def precision_at_k(
    recommended: Sequence[str],
    relevant: set[str],
    k: int,
) -> float:
    """
    Precision@K: Top-K 추천 중 정답 아이템의 비율.

    Parameters
    ----------
    recommended : list[str]
        추천된 아이템 ID 리스트.
    relevant : set[str]
        정답 아이템 ID 집합.
    k : int
        상위 K개까지만 평가.

    Returns
    -------
    float
        Precision 값 (0.0 ~ 1.0).
    """
    if k == 0:
        return 0.0
    top_k = set(recommended[:k])
    hits = top_k & relevant
    return len(hits) / k

def ndcg_at_k(
    recommended: Sequence[str],
    relevant: set[str],
    k: int,
) -> float:
    """
    NDCG@K: 정규화 할인 누적 이득 (Normalized Discounted Cumulative Gain).

    - 추천 리스트의 **순서**를 반영하여 상위에 정답이 있을수록 높은 점수.
    - Implicit feedback이므로 relevance = binary (0 or 1).
    - DCG  = Σ (rel_i / log2(i+1))  for i=1..K
    - IDCG = 이상적인 순서(정답을 모두 앞에 배치)에서의 DCG

    Parameters
    ----------
    recommended : list[str]
        추천 리스트 (순서 중요).
    relevant : set[str]
        정답 아이템 집합.
    k : int
        상위 K개 평가.

    Returns
    -------
    float
        NDCG 값 (0.0 ~ 1.0). 정답이 없으면 0.0.
    """
    if not relevant:
        return 0.0

    top_k = recommended[:k]
    
    dcg = 0.0
    for i, item in enumerate(top_k, start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 1)

    n_relevant_in_k = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, n_relevant_in_k + 1))

    if idcg == 0.0:
        return 0.0
    
    return dcg / idcg

def hit_rate_at_k(
    recommended: Sequence[str],
    relevant: set[str],
    k: int,
) -> float:
    """
    HitRate@K: 정답 아이템이 Top-K에 1개라도 있으면 1, 아니면 0.

    Parameters
    ----------
    recommended : list[str]
        추천 리스트.
    relevant : set[str]
        정답 아이템 집합.
    k : int
        상위 K개 평가.

    Returns
    -------
    float
        1.0 (hit) 또는 0.0 (miss).
    """
    if not relevant:
        return 0.0
    top_k = set(recommended[:k])
    
    if top_k & relevant:
        return 1.0
    else:
        return 0.0

def mrr_at_k(
    recommended: Sequence[str],
    relevant: set[str],
    k: int,
) -> float:
    """
    MRR@K (Mean Reciprocal Rank): 첫 번째로 맞춘 아이템의 역순위.

    Parameters
    ----------
    recommended : list[str]
        추천 리스트.
    relevant : set[str]
        정답 아이템 집합.
    k : int
        상위 K개 평가.

    Returns
    -------
    float
        첫 hit의 1/rank. 없으면 0.0.
    """
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0

def evaluate_all(
    user_recommendations: dict[str, list[str]],
    user_ground_truth: dict[str, set[str]],
    k_list: list[int] | None = None,
) -> dict[str, float]:
    """
    전체 유저에 대해 Recall@K, NDCG@K, HitRate@K, Precision@K, MRR@K를 계산.

    Parameters
    ----------
    user_recommendations : dict[str, list[str]]
        유저 ID → 추천 아이템 리스트 (순서대로).
    user_ground_truth : dict[str, set[str]]
        유저 ID → 정답(실제 구매) 아이템 집합.
    k_list : list[int], optional
        평가할 K 값 리스트. 기본값 [10, 20].

    Returns
    -------
    dict[str, float]
        지표 이름 → 값. 예: {"Recall@10": 0.12, "NDCG@20": 0.08, ...}
    """

    if k_list is None:
        k_list = [10, 20]

    eval_users = [
        uid for uid in user_ground_truth
        if len(user_ground_truth[uid]) > 0 and uid in user_recommendations
    ]

    if not eval_users:
        return {
            f"{metric}@{k}": 0.0 
            for k in k_list 
            for metric in ["Recall", "NDCG", "HitRate", "Precision", "MRR"]
        }

    results: dict[str, float] = {}

    for k in k_list:
        recalls, ndcgs, hits, precisions, mrrs = [], [], [], [], []

        for uid in eval_users:
            rec = user_recommendations[uid]
            rel = user_ground_truth[uid]

            recalls.append(recall_at_k(rec, rel, k))
            ndcgs.append(ndcg_at_k(rec, rel, k))
            hits.append(hit_rate_at_k(rec, rel, k))
            precisions.append(precision_at_k(rec, rel, k))
            mrrs.append(mrr_at_k(rec, rel, k))
        
        results[f"Recall@{k}"] = np.mean(recalls)
        results[f"NDCG@{k}"] = np.mean(ndcgs)
        results[f"HitRate@{k}"] = np.mean(hits)
        results[f"Precision@{k}"] = np.mean(precisions)
        results[f"MRR@{k}"] = np.mean(mrrs)

    results["n_eval_users"] = len(eval_users)
    return results