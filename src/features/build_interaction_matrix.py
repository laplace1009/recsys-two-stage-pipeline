from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

@dataclass
class InteractionMatrix:
    """
    user-item interaction matrix와 ID 매핑을 함께 보관.

    Attributes
    ----------
    matrix : sparse.csr_matrix
        (n_users, n_items) 크기의 CSR sparse matrix.
        값 = log1p(purchase_count).
    user_to_idx : dict[str, int]
        user_id → 행 인덱스.
    idx_to_user : dict[int, str]
        행 인덱스 → user_id.
    item_to_idx : dict[str, int]
        item_id → 열 인덱스.
    idx_to_item : dict[int, str]
        열 인덱스 → item_id.
    """
    matrix: sparse.csr_matrix
    user_to_idx: dict[str, int] = field(repr=False)
    idx_to_user: dict[int, str] = field(repr=False)
    item_to_idx: dict[str, int] = field(repr=False)
    idx_to_item: dict[int, str] = field(repr=False)

    @property
    def n_users(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_items(self) -> int:
        return self.matrix.shape[1]

    @property
    def n_interactions(self) -> int:
        return self.matrix.nnz

    @property
    def sparsity(self) -> float:
        """비어있는 셀의 비율 (1.0에 가까울수록 sparse)."""
        total = self.n_users * self.n_items
        if total == 0:
            return 0.0
        return 1.0 - (self.n_interactions / total)
    
    @property
    def density(self) -> float:
        """채워진 셀의 비율 (0.0에 가까울수록 sparse)."""
        return 1.0 - self.sparsity
    

def build_interaction_matrix(
    interactions: pd.DataFrame,
    value_col: str = "value",
) -> InteractionMatrix:
    """
    interactions DataFrame → sparse CSR matrix 변환.

    Parameters
    ----------
    interactions : pd.DataFrame
        필수 컬럼: user_id, item_id, value_col
        (split.py의 build_tables_and_split이 생성한 interactions 테이블)
    value_col : str
        가중치 컬럼 이름. 기본값 'value' = log1p(purchase_count).

    Returns
    -------
    InteractionMatrix
        sparse matrix + ID 매핑.
    """
    unique_users = interactions["user_id"].unique()
    unique_items = interactions["item_id"].unique()

    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    item_to_idx = {item_id: idx for idx, item_id in enumerate(unique_items)}

    idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}
    idx_to_item = {idx: item_id for item_id, idx in item_to_idx.items()}

    row_indices = interactions["user_id"].map(user_to_idx).values
    col_indices = interactions["item_id"].map(item_to_idx).values
    values = interactions[value_col].values

    n_users = len(user_to_idx)
    n_items = len(item_to_idx)

    matrix = sparse.csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )

    return InteractionMatrix(
        matrix=matrix,
        user_to_idx=user_to_idx,
        idx_to_user=idx_to_user,
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
    )

def print_matrix_stats(im: InteractionMatrix) -> None:
    """Interaction matrix 통계 요약 출력."""
    print("\n" + "=" * 60)
    print("  Interaction Matrix 통계")
    print("=" * 60)
    print(f"  유저 수:        {im.n_users:,}")
    print(f"  아이템 수:      {im.n_items:,}")
    print(f"  상호작용 수:    {im.n_interactions:,}")
    print(f"  Matrix shape:   {im.matrix.shape}")
    print(f"  Sparsity:       {im.sparsity:.6f} ({im.sparsity * 100:.4f}%)")
    print(f"  Density:        {im.density:.6f} ({im.density * 100:.4f}%)")
    print(f"  값 범위:        [{im.matrix.data.min():.4f}, {im.matrix.data.max():.4f}]")
    print(f"  평균 값:        {im.matrix.data.mean():.4f}")
    print(f"  유저당 평균 아이템: {im.n_interactions / im.n_users:.1f}")
    print(f"  아이템당 평균 유저: {im.n_interactions / im.n_items:.1f}")
    print("=" * 60)



PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

def main() -> None:
    interactions_path = DATA_DIR / "interactions.parquet"

    if not interactions_path.exists():
        print(f"Error: {interactions_path} 파일이 존재하지 않습니다.")
        print("먼저 split.py의 build_tables_and_split 함수를 실행하여 interactions.parquet을 생성하세요.")
        return
    
    print(f"Loading: {interactions_path}")
    interactions = pd.read_parquet(interactions_path)

    print(f"로드 완료: {len(interactions):,} 행")
    print(f"컬럼: {list(interactions.columns)}")

    im = build_interaction_matrix(interactions)
    print_matrix_stats(im)

if __name__ == "__main__":
    main()