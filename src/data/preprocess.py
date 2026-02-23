from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data"/ "raw" / "Online Retail.xlsx"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

print(f"Project root: {PROJECT_ROOT}")
print(f"Raw path: {RAW_PATH}")
print(f"Output directory: {OUT_DIR}")

NON_PRODUCT_CODES: set[str] = {
    "POST", "DOT", "M", "m", "D", "C2", "S", "B",
    "BANK CHARGES", "AMAZONFEE", "CRUK", "PADS",
}

NON_PRODUCT_PREFIX: tuple[str, ...] = ("gift_", "DCGS", "DCGSSGIRL", "DCGSSBOY")

def is_product_code(code: str) -> bool:
    if code in NON_PRODUCT_CODES:
        return False
    if any(code.startswith(p) for p in NON_PRODUCT_PREFIX):
        return False
    if re.match(r"^\d{4,6}[a-zA-Z]{0,2}$", code):
        return True
    if code.isdigit():
        return True
    return False

def data_load(path: Path) -> pd.DataFrame:
    """데이터(엑셀) 로드, ID 컬럼을 문자열로 캐스팅, InvoiceDate를 datetime으로 파싱."""
    dtype = {"InvoiceNo": str, "StockCode": str, "CustomerID": str}

    df = pd.read_excel(
        path,
        dtype=dtype,
        parse_dates=["InvoiceDate"],
    )
    return df

def removce_cancellation(df: pd.DataFrame) -> pd.DataFrame:
    """InvoiceNo가 'C'로 시작하는 행 제거 (취소된 주문)"""
    mask = df["InvoiceNo"].str.upper().str.startswith("C")
    df = df[~mask].copy()
    return df

def remove_invalid_qty_price(df: pd.DataFrame) -> pd.DataFrame:
    """Quantity와 UnitPrice가 0 이하인 행 제거 (유효하지 않은 주문)"""
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    return df

def remove_null_customers(df: pd.DataFrame) -> pd.DataFrame:
    """CustomerID가 null인 행 제거 (고객 정보 없는 주문) 개인화 추천에는 user_id가 필수"""
    df = df.dropna(subset=["CustomerID"]).copy()
    return df

def filter_non_products(df: pd.DataFrame) -> pd.DataFrame:
    """비상품 코드 제거 (배송비, 수수료, 상품권 등)"""
    mask = df["StockCode"].apply(is_product_code)
    df = df[mask].copy()
    return df

def normalize_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """strp + 공백 정리 + 대문자 통일"""
    df["description_clean"] = (
        df["Description"]
        .fillna("")
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )
    mode_desc = (
        df.groupby("StockCode")["description_clean"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "")
        .rename("desc_mode")
    )
    df = df.merge(mode_desc, on="StockCode", how="left")
    df["description_clean"] = df["desc_mode"]
    df.drop(columns=["desc_mode"], inplace=True)
    return df

def merge_duplicate_lines(df: pd.DataFrame) -> pd.DataFrame:
    """(invoice_no, user_id, item_id) 기준 qty=sum, amount=sum(qty*price)."""
    df["amount"] = df["Quantity"] * df["UnitPrice"]

    agg_dict = {
        "Quantity": "sum",
        "UnitPrice": "first",
        "amount": "sum",
        "InvoiceDate": "first",
        "Country": "first",
        "description_clean": "first",
    }

    df = (
        df.groupby(["InvoiceNo", "StockCode", "CustomerID"], as_index=False)
        .agg(agg_dict)
    )

    df.rename(
        columns={
            "InvoiceNo": "invoice_no",
            "CustomerID": "user_id",
            "StockCode": "item_id",
            "InvoiceDate": "ts",
            "Quantity": "qty",
            "UnitPrice": "unit_price",
            "Country": "country",
        },
        inplace=True,
    )
    return df

def main() -> None:
    df = data_load(RAW_PATH)
    df = removce_cancellation(df)
    df = remove_invalid_qty_price(df)
    df = remove_null_customers(df)
    df = filter_non_products(df)
    df = merge_duplicate_lines(df)


if __name__ == "__main__":
    main()