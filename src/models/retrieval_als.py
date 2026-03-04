from __future__ import annotations

import numpy as np
from scipy import sparse

from implicit.als import AlternatingLeastSquares

from src.features.build_interaction_matrix import InteractionMatrix

class RetrievalALS:
    """
    Implicit ALS 기반 후보생성 모델.

    user-item implicit interaction matrix를 행렬분해(Matrix Factorization)하여
    유저별 latent factor를 학습하고, item factor와의 내적으로 Top-K 후보를 생성한다.

    사용 예:
        model = RetrievalALS(factors=64, iterations=15)
        model.fit(interaction_matrix)
        recs = model.recommend("user_123", k=20)
    """

    def __init__(
        self,
        factors: int = 64,
        iterations: int = 15,
        regularization: float = 0.01,
        random_state: int = 42,
        use_gpu: bool = False,
    ):
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.random_state = random_state
        self.use_gpu = use_gpu

        self._model: AlternatingLeastSquares | None = None
        self._interaction_matrix: InteractionMatrix | None = None
        self._user_items: sparse.csr_matrix | None = None

    def fit(self, interaction_matrix: InteractionMatrix) -> "RetrievalALS":
        """
        ALS 모델 학습.

        Parameters
        ----------
        interaction_matrix : InteractionMatrix
            build_interaction_matrix()로 생성한 객체.

        Returns
        -------
        self
        """
        self._interaction_matrix = interaction_matrix

        item_user = interaction_matrix.matrix.T.tocsr()

        self._model = AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            random_state=self.random_state,
            use_gpu=self.use_gpu,
        )

        self._user_items = interaction_matrix.matrix

        print(f"  ALS 학습 시작: factors={self.factors}, "
              f"iterations={self.iterations}, "
              f"regularization={self.regularization}")
        
        self._model.fit(item_user, show_progress=True)

        print(" ALS 학습 완료!")
        return self
    
    def recommend(self, user_id: str, k: int) -> list[str]:
        """
        유저에게 Top-K 아이템 후보를 추천한다.

        이미 구매한 아이템은 자동 제외 (implicit 라이브러리의 filter_already_liked).

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
        if self._model is None or self._interaction_matrix is None:
            raise RuntimeError("fit()을 먼저 호출하세요.")
        
        im = self._interaction_matrix

        if user_id not in im.user_to_idx:
            return []
        
        user_idx = im.user_to_idx[user_id]

        item_ids, scores = self._model.recommend(
            userid=user_idx,
            user_items=self._user_items[user_idx],
            N=k,
            filter_already_liked_items=True,
        )

        recommendations = [
            im.idx_to_item[int(idx)]
            for idx in item_ids
        ]

        return recommendations
    

    def recommend_batch(
        self,
        user_ids: list[str],
        k: int
    ) -> dict[str, list[str]]:
        """
        여러 유저에 대해 일괄 추천 (성능 최적화).

        Parameters
        ----------
        user_ids : list[str]
            추천 대상 유저 ID 리스트.
        k : int
            추천 아이템 수.

        Returns
        -------
        dict[str, list[str]]
            유저 ID → 추천 아이템 리스트.
        """
        if self._model is None or self._interaction_matrix is None:
            raise RuntimeError("fit()을 먼저 호출하세요.")
        
        im = self._interaction_matrix
        results: dict[str, list[str]] = {}

        valid_user_indices = []
        valid_user_ids = []
        for uid in user_ids:
            if uid in im.user_to_idx:
                valid_user_indices.append(im.user_to_idx[uid])
                valid_user_ids.append(uid)
        
        if not valid_user_indices:
            return {uid: [] for uid in user_ids}
        
        user_indices = np.array(valid_user_indices)

        all_item_ids, all_scores = self._model.recommend(
            userid=user_indices,
            user_items=self._user_items[user_indices],
            N=k,
            filter_already_liked_items=True,
        )

        for i, uid in enumerate(valid_user_ids):
            results[uid] = [
                im.idx_to_item[int(idx)]
                for idx in all_item_ids[i]
            ]

        for uid in user_ids:
            if uid not in results:
                results[uid] = []

        return results
        
