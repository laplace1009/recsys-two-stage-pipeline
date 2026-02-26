from __future__ import annotations

from collections import Counter
from typing import Optional

import pandas as pd

class GlobalPopular:
    """
    전체 학습 기간(train split)의 인기 아이템을 추천하는 베이스라인.

    인기도 측정:
      - 기본: 구매 빈도 (해당 아이템이 등장한 라인 수)
      - 옵션: unique user 수 기반 인기도 (user_based=True)

    사용 예:
        model = GlobalPopular()
        model.fit(transactions_train)
        recs = model.recommend("user_123", k=20)
    """

    def __init__(self, user_based: bool = False):
        """
        Parameters
        ----------
        user_based : bool
            True면 unique user 수 기반 인기도,
            False면 단순 구매 빈도(라인 수).
        """
        self.user_based = user_based
        self._popularity_ranking: list[str] = []
        self._user_seen: dict[str, set[str]] = {}

    def fit(self, transactions: pd.DataFrame) -> "GlobalPopular":
        """
        학습: train 데이터에서 인기도 순위를 계산한다.

        Parameters
        ----------
        transactions : pd.DataFrame
            train split의 트랜잭션 데이터.
            필수 컬럼: user_id, item_id

        Returns
        -------
        self
        """
        if self.user_based:
            popularity = (
                transactions.groupby("item_id")["user_id"]
                .nunique()
                .sort_values(ascending=False)
            )
        else:
            popularity = (
                transactions["item_id"]
                .value_counts()
                .sort_values(ascending=False)
            )

        self._popularity_ranking = popularity.index.tolist()

        self._user_seen = (
            transactions.groupby("user_id")["item_id"]
            .apply(set)
            .to_dict()
        )

        return self

    def recommend(self, user_id: str, k: int) -> list[str]:
        """
        유저에게 Top-K 인기 아이템을 추천한다.

        이미 구매한 아이템은 제외하고, 그 다음 인기 아이템으로 채운다.

        Parameters
        ----------
        user_id : str
            추천 대상 유저 ID.
        k : int
            추천 아이템 수.

        Returns
        -------
        list[str]
            추천 아이템 ID 리스트 (최대 K개).
        """
        seen = self._user_seen.get(user_id, set())
        recs: list[str] = []
        for item in self._popularity_ranking:
            if item not in seen:
                recs.append(item)
            if len(recs) >= k:
                break
        return recs

class RecentPopular:
    """
    최근 N주간의 인기 아이템을 추천하는 베이스라인.

    GlobalPopular와 차이점:
      - 전체 기간이 아닌 최근 recent_weeks 동안의 트렌드 반영
      - 시즌성 / 최신 트렌드를 캡처하여 실무에서 더 강력한 경우가 많음

    사용 예:
        model = RecentPopular(recent_weeks=4)
        model.fit(transactions_train)
        recs = model.recommend("user_123", k=20)
    """

    def __init__(
        self,
        recent_weeks: int = 4,
        user_based: bool = False,
    ):
        """
        Parameters
        ----------
        recent_weeks : int
            최근 몇 주간의 데이터를 사용할지.
        user_based : bool
            True면 unique user 수, False면 구매 빈도.
        """
        self.recent_weeks = recent_weeks
        self.user_based = user_based
        self._popularity_ranking: list[str] = []
        self._user_seen: dict[str, set[str]] = {}

    def fit(self, transactions: pd.DataFrame) -> "RecentPopular":
        """
        학습: 최근 N주의 데이터에서 인기도를 계산한다.

        Parameters
        ----------
        transactions : pd.DataFrame
            train split의 트랜잭션 데이터.
            필수 컬럼: user_id, item_id, ts (timestamp)

        Returns
        -------
        self
        """
        max_ts = transactions["ts"].max()
        cutoff = max_ts - pd.Timedelta(weeks=self.recent_weeks)
        recent = transactions[transactions["ts"] >= cutoff]

        if self.user_based:
            popularity = (
                recent.groupby("item_id")["user_id"]
                .nunique()
                .sort_values(ascending=False)
            )
        else:
            popularity = (
                recent["item_id"]
                .value_counts()
                .sort_values(ascending=False)
            )

        self._popularity_ranking = popularity.index.tolist()

        self._user_seen = (
            transactions.groupby("user_id")["item_id"]
            .apply(set)
            .to_dict()
        )

        return self

    def recommend(self, user_id: str, k: int) -> list[str]:
        """
        유저에게 최근 인기 Top-K 아이템을 추천한다.

        Parameters
        ----------
        user_id : str
            추천 대상 유저 ID.
        k : int
            추천 아이템 수.

        Returns
        -------
        list[str]
            추천 아이템 ID 리스트 (최대 K개).
        """
        seen = self._user_seen.get(user_id, set())
        recs: list[str] = []
        for item in self._popularity_ranking:
            if item not in seen:
                recs.append(item)
            if len(recs) >= k:
                break
        return recs