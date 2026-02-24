from __future__ import annotations

import pandas as pd
import numpy as np

def filter_min_invoices(df: pd.DataFrame, min_invoices: int = 2) -> pd.DataFrame:
    """유저별 invoice가 min_invoices 미만이면 제거 (평가 불가)."""
    user_inv_count = df.groupby("user_id")["invoice_no"].nunique()
    valid_users = user_inv_count[user_inv_count >= min_invoices].index
    df = df[df["user_id"].isin(valid_users)].copy()
    return df

def build_tables_and_split(
    df: pd.DataFrame,
    test_weeks: int = 8,
    valid_weeks: int = 8
) -> dict[str, pd.DataFrame | dict]:
    """
    transactions_clean, events, interactions 테이블 생성.
    시간 기반 split (OOT): 마지막 test_weeks=test, 그 전 valid_weeks=valid, 나머지=train.
    """
    transcations = df.sort_values(["user_id", "ts"]).reset_index(drop=True)

    max_ts = transcations["ts"].max()
    test_cutoff = max_ts - pd.Timedelta(weeks=test_weeks)
    valid_cutoff = test_cutoff - pd.Timedelta(weeks=valid_weeks)

    transcations["split"] = "train"
    transcations.loc[transcations["ts"] >= valid_cutoff, "split"] = "valid"
    transcations.loc[transcations["ts"] >= test_cutoff, "split"] = "test"

    split_info = {
        "train": {"end": str(valid_cutoff)},
        "valid": {"start": str(valid_cutoff), "end": str(test_cutoff)},
        "test": {"start": str(test_cutoff), "end": str(max_ts)},
    }

    events = (
        transcations.groupby(["invoice_no", "user_id", "split"])
        .agg(
            ts=("ts", "first"),
            items=("item_id", lambda x: sorted(set(x))),
            n_items=("item_id", "nunique"),
            total_amount=("amount", "sum"),
        )
        .reset_index()
    )

    events = events.sort_values(["user_id", "ts"]).reset_index(drop=True)

    train_tx = transcations[transcations["split"] == "train"]
    interactions = (
        train_tx.groupby(["user_id", "item_id"])
        .agg(
            purchase_count=("qty", "sum"),
            total_amount=("amount", "sum"),
            last_ts=("ts", "max")
        )
        .reset_index()
    )

    interactions["value"] = np.log1p(interactions["purchase_count"])

    item_meta = (
        transcations.groupby("item_id")["description_clean"]
        .first()
        .reset_index()
    )

    item_popularity = (
        train_tx.groupby("item_id")
        .agg(
            buy_count=("qty", "sum"),
            user_count=("user_id", "nunique"),
            invoice_count=("invoice_no", "nunique")
        )
        .reset_index()
        .sort_values("buy_count", ascending=False)
    )

    return {
        "transactions_clean": transcations,
        "events": events,
        "interactions": interactions,
        "item_meta": item_meta,
        "item_popularity": item_popularity,
        "split_info": split_info
    }