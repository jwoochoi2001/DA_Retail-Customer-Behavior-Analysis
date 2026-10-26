"""원본 online_retail_II.xlsx -> 분석용 데이터셋 (제외 기준·건수 기록 포함).

실행: python analysis/01_build_analysis_dataset.py   (프로젝트 루트 기준)
필요: pandas, openpyxl
제외 사유는 아래 순서로 행마다 '첫 번째 해당 사유' 하나만 부여한다(중복 집계 없음).
"""
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw/online_retail_II.xlsx")
RULES = Path("data/raw/transaction_code_rules.csv")
OUT_DATA = Path("data/processed/retail_clean.csv")
OUT_SUM = Path("outputs/tables/exclusion_summary.csv")
OUT_CODE = Path("outputs/tables/exclusion_by_code.csv")

SHEETS = ["Year 2009-2010", "Year 2010-2011"]
OVERLAP_START, OVERLAP_END = "2010-12-01", "2010-12-10"  # 두 시트가 겹치는 구간

df = pd.concat([pd.read_excel(RAW, sheet_name=s).assign(source_sheet=s) for s in SHEETS],
               ignore_index=True)
df["Invoice"] = df["Invoice"].astype(str)
df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()  # gift_/m 등 소문자 표기 통일
df["line_amount"] = df["Quantity"] * df["Price"]
n_raw, amt_raw = len(df), df["line_amount"].sum()

rules = pd.read_csv(RULES, encoding="utf-8-sig")
rules["StockCode"] = rules["StockCode"].str.strip().str.upper()
rule_map = rules[rules["treatment"] == "exclude"].set_index("StockCode")["reason_ko"]

# 1) 시트 간 중복: 2010-12-01~12-09 구간은 두 시트에 동일 행이 들어 있어 앞 시트(2009-2010) 쪽을 제외
in_overlap = df["InvoiceDate"].ge(OVERLAP_START) & df["InvoiceDate"].lt(OVERLAP_END)
reason = pd.Series(pd.NA, index=df.index, dtype="object")
reason[in_overlap & (df["source_sheet"] == SHEETS[0])] = "1.시트 간 중복(2010-12-01~09)"

# 2) 비상품·회계성 코드(배송료, 수수료, 수기/회계 조정, 샘플, 테스트, 상품권)
m = reason.isna() & df["StockCode"].isin(rule_map.index)
reason[m] = "2.비상품·회계조정 코드(규칙표)"

# 3) 비회원(Customer ID 결측)
m = reason.isna() & df["Customer ID"].isna()
reason[m] = "3.비회원(Customer ID 없음)"

# 4) 회원이지만 단가 0 이하(무상·테스트·수기 입력 등 비정상 가격)
m = reason.isna() & (df["Price"] <= 0)
reason[m] = "4.단가 0 이하"

# 5~7) 이상·특수 주문 (1~4 제외 후 남은 구매 주문을 주문 번호 단위로 판단)
live = df[reason.isna()]
is_c = live["Invoice"].str.startswith("C")
buy, cxl = live[~is_c], live[is_c]
order_amt = buy.groupby("Invoice")["line_amount"].sum()
lg = np.log10(order_amt[order_amt > 0])
q1, q3 = lg.quantile([.25, .75])
FENCE = 10 ** (q3 + 3 * (q3 - q1))            # 주문 금액 로그의 IQR x 3 상한(극단값 기준)
SPECIAL_MIN, SPECIAL_RATIO = 5000, 0.9         # 대형 주문 후 취소 매칭 기준
# 같은 고객·같은 상품·정확히 반대 수량·이후 날짜의 취소 행을 해당 주문의 '짝 취소'로 매칭(근사)
cx = cxl[["Customer ID", "StockCode", "Quantity", "InvoiceDate", "line_amount"]].rename(
    columns={"Quantity": "cq", "InvoiceDate": "cdate", "line_amount": "camt"})
cx["cq"] = -cx["cq"]; cx["cidx"] = cx.index
cand = buy[buy["Invoice"].isin(order_amt[order_amt >= SPECIAL_MIN].index)].reset_index()
mt = cand.merge(cx, left_on=["Customer ID", "StockCode", "Quantity"], right_on=["Customer ID", "StockCode", "cq"])
mt = mt[mt["cdate"] >= mt["InvoiceDate"]].drop_duplicates(["Invoice", "cidx"])
matched = mt.groupby("Invoice")["camt"].sum().mul(-1).reindex(order_amt.index).fillna(0)
big = order_amt[order_amt > FENCE].index
special = order_amt[(order_amt >= SPECIAL_MIN) & (order_amt <= FENCE) & (matched / order_amt >= SPECIAL_RATIO)].index
is_buy_row = ~df["Invoice"].str.startswith("C")
r5 = f"5.초대형 주문(주문 금액 > {FENCE:,.0f})"
r6 = f"6.전액 취소된 대형 주문(≥{SPECIAL_MIN:,}, 매칭 취소 ≥{SPECIAL_RATIO:.0%})"
reason[reason.isna() & is_buy_row & df["Invoice"].isin(big)] = r5
reason[reason.isna() & is_buy_row & df["Invoice"].isin(special)] = r6
drop_cancel = mt.loc[mt["Invoice"].isin(big.union(special)), "cidx"].unique()
m = reason.isna() & df.index.isin(drop_cancel)
reason[m] = "7.5·6번 주문과 짝지어진 취소 행"
flag = pd.concat([order_amt[big].rename("order_amount").to_frame().assign(reason=r5),
                  order_amt[special].rename("order_amount").to_frame().assign(reason=r6)])
flag["matched_cancel_amount"] = matched.reindex(flag.index).round(2)
flag["customer_id"] = df.groupby("Invoice")["Customer ID"].first().reindex(flag.index).astype("int64")
flag["invoice_date"] = df.groupby("Invoice")["InvoiceDate"].min().reindex(flag.index)
flag["lines"] = buy.groupby("Invoice").size().reindex(flag.index)
flag["quantity"] = buy.groupby("Invoice")["Quantity"].sum().reindex(flag.index)
flag = flag.sort_values("order_amount", ascending=False).round(2)

df["exclude_reason"] = reason
excl = df[df["exclude_reason"].notna()]
clean = df[df["exclude_reason"].isna()].copy()

# 요약표: 사유별 행 수, 금액 합계
summ = (excl.groupby("exclude_reason")
        .agg(rows=("Invoice", "size"), amount=("line_amount", "sum")).reset_index())
summ.loc[len(summ)] = ["제외 합계", summ["rows"].sum(), summ["amount"].sum()]
summ.loc[len(summ)] = ["최종 분석용 데이터", len(clean), clean["line_amount"].sum()]
summ.loc[len(summ)] = ["원본 합계(두 시트)", n_raw, amt_raw]
summ["amount"] = summ["amount"].round(2)

# 규칙표 코드별 상세
by_code = (excl[excl["exclude_reason"].str.startswith("2.")]
           .groupby("StockCode").agg(rows=("Invoice", "size"), amount=("line_amount", "sum"))
           .join(rule_map).reset_index().sort_values("rows", ascending=False))
by_code["amount"] = by_code["amount"].round(2)

# 분석용 데이터셋
clean["is_cancel"] = clean["Invoice"].str.startswith("C")
final = pd.DataFrame({
    "invoice": clean["Invoice"], "stock_code": clean["StockCode"],
    "description": clean["Description"], "quantity": clean["Quantity"],
    "invoice_date": clean["InvoiceDate"], "price": clean["Price"],
    "customer_id": clean["Customer ID"].astype("int64"), "country": clean["Country"],
    "line_amount": clean["line_amount"].round(2), "is_cancel": clean["is_cancel"],
}).reset_index(drop=True)

# 검증: 행 수·금액 합계 대사
assert summ.loc[summ["exclude_reason"] == "제외 합계", "rows"].iloc[0] + len(final) == n_raw
assert abs(excl["line_amount"].sum() + final["line_amount"].sum() - amt_raw) < 0.01
assert final["customer_id"].notna().all() and (final["price"] > 0).all()
assert (final.loc[~final["is_cancel"], "quantity"] > 0).all()
assert (final.loc[final["is_cancel"], "quantity"] < 0).sum() >= len(final[final["is_cancel"]]) - 1

OUT_FLAG = Path("outputs/tables/excluded_outlier_orders.csv")
for p in (OUT_DATA, OUT_SUM, OUT_CODE, OUT_FLAG):
    p.parent.mkdir(parents=True, exist_ok=True)
flag.to_csv(OUT_FLAG, index_label="invoice", encoding="utf-8-sig")
final.to_csv(OUT_DATA, index=False, encoding="utf-8-sig")
summ.to_csv(OUT_SUM, index=False, encoding="utf-8-sig")
by_code.to_csv(OUT_CODE, index=False, encoding="utf-8-sig")

pd.set_option("display.width", 200); pd.set_option("display.float_format", "{:,.2f}".format)
print(summ.to_string(index=False)); print()
print(by_code.to_string(index=False)); print()
print("분석용:", final.shape, "| 고객 수:", final.customer_id.nunique(),
      "| 기간:", final.invoice_date.min(), "~", final.invoice_date.max())
print("취소(C) 행:", int(final.is_cancel.sum()), "| 주문 수:", final.invoice.nunique())
print(f"이상 주문 상한(로그 IQR x3): {FENCE:,.0f} | 5번 주문 {len(big)}건, 6번 주문 {len(special)}건")
print(flag.head(12).to_string())
print(final.head().to_string())
